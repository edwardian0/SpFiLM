#!/usr/bin/env python3
"""
mask_audit.py -- Unit 1 + Unit 2 of the fundus pipeline.

Purpose
-------
Before any model exists, prove that every (image, disc, cup) triple in every
dataset resolves to the SAME canonical representation:

    disc : bool HxW   full optic disc, INCLUDING the cup region
    cup  : bool HxW   optic cup, a strict subset of disc

Everything downstream (training targets, Dice, vCDR, ROI cropping) is derived
from those two arrays, so if this stage is wrong nothing later can be right.
The REFUGE convention is inverted relative to intuition (background is the
BRIGHTEST value, cup is 0); getting it backwards trains fine and scores zero.

Outputs (written to --out):
    manifest.csv        one row per image: domain, id, image path, mask paths
    audit.csv           manifest + per-image mask statistics + flags
    audit_summary.json  per-domain rollup, including median disc diameter (px)
    contact_sheet_<domain>.png   12 random overlays to check BY EYE

Usage
-----
    python mask_audit.py \
        --refuge-root  datasets/REFUGE \
        --drishti-root datasets/DRISHTI-GS \
        --out          artifacts/data_audit \
        --drishti-thresh 0.75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------
# Loading -> canonical (disc, cup)
# --------------------------------------------------------------------------
def _load_gray(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def load_refuge_mask(mask_path, tol: int = 10):
    """REFUGE BMP convention: 0 = cup, 128 = disc(rim), 255 = background."""
    a = _load_gray(mask_path).astype(np.int16)
    levels = np.array([0, 128, 255], dtype=np.int16)
    snapped = levels[np.abs(a[..., None] - levels[None, None, :]).argmin(-1)]
    offgrid = int((np.abs(a - snapped) > tol).sum())
    cup = snapped == 0
    disc = snapped <= 128           # rim OR cup
    return disc, cup, {"offgrid_px": offgrid}


def load_drishti_mask(od_path, cup_path, thresh: float = 0.75):
    """Drishti-GS ships soft maps (expert agreement in 0..1, stored 0..255).

    The dataset paper evaluates against a 0.75 threshold (>=3 of 4 experts).
    """
    od = _load_gray(od_path).astype(np.float32) / 255.0
    cu = _load_gray(cup_path).astype(np.float32) / 255.0
    return od >= thresh, cu >= thresh, {"od_soft_mean": float(od.mean())}


def load_binary_pair(disc_path, cup_path):
    """Generic: two separate binary files (RIM-ONE-r3, converted RIGA)."""
    return _load_gray(disc_path) > 127, _load_gray(cup_path) > 127, {}


def canonicalise(disc: np.ndarray, cup: np.ndarray):
    """Enforce cup subset of disc. Returns (disc, cup, n_violating_px)."""
    viol = int((cup & ~disc).sum())
    return (disc | cup), cup, viol


# --------------------------------------------------------------------------
# Statistics + flags
# --------------------------------------------------------------------------
def _vertical_extent(m: np.ndarray) -> int:
    rows = np.nonzero(m.any(axis=1))[0]
    return 0 if rows.size == 0 else int(rows.max() - rows.min() + 1)


def mask_stats(disc: np.ndarray, cup: np.ndarray) -> dict:
    H, W = disc.shape
    s = {
        "H": H, "W": W,
        "disc_px": int(disc.sum()),
        "cup_px": int(cup.sum()),
    }
    s["disc_frac"] = s["disc_px"] / (H * W)
    s["cup_disc_area_ratio"] = s["cup_px"] / s["disc_px"] if s["disc_px"] else np.nan

    ys, xs = np.nonzero(disc)
    if ys.size:
        s.update(
            disc_cy=float(ys.mean()), disc_cx=float(xs.mean()),
            disc_y0=int(ys.min()), disc_y1=int(ys.max()),
            disc_x0=int(xs.min()), disc_x1=int(xs.max()),
        )
        s["disc_h"] = s["disc_y1"] - s["disc_y0"] + 1
        s["disc_w"] = s["disc_x1"] - s["disc_x0"] + 1
        s["disc_touches_border"] = bool(
            s["disc_y0"] == 0 or s["disc_x0"] == 0
            or s["disc_y1"] == H - 1 or s["disc_x1"] == W - 1
        )
    else:
        s.update(disc_cy=np.nan, disc_cx=np.nan, disc_h=0, disc_w=0,
                 disc_touches_border=False)

    dv, cv = _vertical_extent(disc), _vertical_extent(cup)
    s["vcdr"] = (cv / dv) if dv else np.nan
    return s


def flags(s: dict) -> dict:
    return {
        "flag_disc_empty": s["disc_px"] == 0,
        "flag_cup_empty": s["cup_px"] == 0,
        "flag_disc_frac_odd": not (1e-4 < s["disc_frac"] < 0.35),
        "flag_vcdr_odd": not (0.05 < (s["vcdr"] if s["vcdr"] == s["vcdr"] else -1) < 1.0),
        "flag_disc_touches_border": bool(s["disc_touches_border"]),
    }


# --------------------------------------------------------------------------
# Dataset indexing
# --------------------------------------------------------------------------
def _valid(p: Path) -> bool:
    return "__MACOSX" not in p.parts and not p.name.startswith("._")


def _stem_map(root: Path, exts) -> dict:
    out = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and _valid(p):
            out.setdefault(p.stem, p)
    return out


def _first_existing(*cands):
    for c in cands:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


def index_refuge(root: Path, include_test: bool = False) -> list[dict]:
    """REFUGE is TWO domains: train400 = Zeiss Visucam, val/test = Canon CR-2."""
    root = Path(root)
    blocks = [
        ("refuge_zeiss",
         _first_existing(root / "Training400", root / "REFUGE-Training400" / "Training400"),
         _first_existing(root / "Annotation-Training400" / "Disc_Cup_Masks")),
        ("refuge_canon_val",
         _first_existing(root / "REFUGE-Validation400" / "REFUGE-Validation400",
                         root / "REFUGE-Validation400"),
         _first_existing(root / "REFUGE-Validation400-GT" / "REFUGE-Validation400-GT" / "Disc_Cup_Masks",
                         root / "REFUGE-Validation400-GT" / "Disc_Cup_Masks")),
    ]
    if include_test:
        blocks.append((
            "refuge_canon_test",
            _first_existing(root / "Test400"),
            _first_existing(root / "REFUGE-Test-GT" / "Disc_Cup_Masks"),
        ))

    rows = []
    for domain, img_dir, mask_dir in blocks:
        if img_dir is None or mask_dir is None:
            print(f"  [skip] {domain}: directory not found")
            continue
        imgs = _stem_map(img_dir, IMG_EXTS - {".bmp"})
        masks = _stem_map(mask_dir, {".bmp"})
        paired = sorted(set(imgs) & set(masks))
        for stem in paired:
            rows.append(dict(domain=domain, image_id=stem, loader="refuge",
                             image_path=str(imgs[stem]),
                             mask_a=str(masks[stem]), mask_b=""))
        print(f"  {domain:20s} images={len(imgs):4d} masks={len(masks):4d} paired={len(paired):4d}")
    return rows


def index_drishti(root: Path) -> list[dict]:
    """Drishti-GS: images under */Images/*, soft maps under */SoftMap/*."""
    root = Path(root)
    imgs, ods, cups = {}, {}, {}
    for p in root.rglob("*.png"):
        if not (p.is_file() and _valid(p)):
            continue
        if p.name.endswith("_ODsegSoftmap.png"):
            ods[p.name[: -len("_ODsegSoftmap.png")]] = p
        elif p.name.endswith("_cupsegSoftmap.png"):
            cups[p.name[: -len("_cupsegSoftmap.png")]] = p
        elif "Images" in p.parts:
            imgs[p.stem] = p

    paired = sorted(set(imgs) & set(ods) & set(cups))
    print(f"  drishti_gs           images={len(imgs):4d} od={len(ods):4d} "
          f"cup={len(cups):4d} paired={len(paired):4d}")
    return [dict(domain="drishti_gs", image_id=s, loader="drishti",
                 image_path=str(imgs[s]), mask_a=str(ods[s]), mask_b=str(cups[s]))
            for s in paired]


# --------------------------------------------------------------------------
# Audit driver
# --------------------------------------------------------------------------
def load_row(row: dict, drishti_thresh: float):
    if row["loader"] == "refuge":
        disc, cup, extra = load_refuge_mask(row["mask_a"])
    elif row["loader"] == "drishti":
        disc, cup, extra = load_drishti_mask(row["mask_a"], row["mask_b"], drishti_thresh)
    else:
        disc, cup, extra = load_binary_pair(row["mask_a"], row["mask_b"])
    disc, cup, viol = canonicalise(disc, cup)
    extra["cup_outside_disc_px"] = viol
    return disc, cup, extra


def audit(rows: list[dict], drishti_thresh: float) -> pd.DataFrame:
    recs = []
    for i, r in enumerate(rows, 1):
        disc, cup, extra = load_row(r, drishti_thresh)
        img_hw = Image.open(r["image_path"]).size[::-1]
        s = mask_stats(disc, cup)
        rec = {**r, **s, **extra, **flags(s)}
        rec["flag_shape_mismatch"] = (img_hw != (s["H"], s["W"]))
        recs.append(rec)
        if i % 100 == 0:
            print(f"    audited {i}/{len(rows)}")
    return pd.DataFrame(recs)


def contact_sheet(df: pd.DataFrame, domain: str, out_png: Path,
                  drishti_thresh: float, n: int = 12, seed: int = 0, thumb: int = 384):
    sub = df[df.domain == domain]
    if sub.empty:
        return
    sub = sub.sample(min(n, len(sub)), random_state=seed)
    cols, rowsn = 4, int(np.ceil(len(sub) / 4))
    fig, axes = plt.subplots(rowsn, cols, figsize=(3.2 * cols, 3.2 * rowsn))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, sub.iterrows()):
        disc, cup, _ = load_row(r.to_dict(), drishti_thresh)
        im = Image.open(r["image_path"]).convert("RGB")
        scale = thumb / max(im.size)
        size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
        arr = np.asarray(im.resize(size, Image.BILINEAR)).astype(np.float32) / 255.0
        d = np.asarray(Image.fromarray(disc.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
        c = np.asarray(Image.fromarray(cup.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
        rim = d & ~c
        arr[rim] = 0.6 * arr[rim] + 0.4 * np.array([0.0, 1.0, 0.0])   # rim  = green
        arr[c] = 0.6 * arr[c] + 0.4 * np.array([1.0, 0.2, 0.0])       # cup  = orange
        ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(f"{r['image_id']}\nvCDR={r['vcdr']:.2f}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(sub):]:
        ax.axis("off")

    fig.suptitle(f"{domain} -- green = rim (disc minus cup), orange = cup", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> dict:
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    out = {}
    for domain, g in df.groupby("domain"):
        out[domain] = {
            "n_images": int(len(g)),
            "image_hw_modes": g.apply(lambda r: f"{r.H}x{r.W}", axis=1).value_counts().head(3).to_dict(),
            "disc_frac_median": round(float(g.disc_frac.median()), 5),
            "disc_diam_px_median": round(float(g[["disc_h", "disc_w"]].max(axis=1).median()), 1),
            "cup_disc_area_ratio_median": round(float(g.cup_disc_area_ratio.median()), 3),
            "vcdr_median": round(float(g.vcdr.median()), 3),
            "cup_outside_disc_px_total": int(g.cup_outside_disc_px.sum()),
            "flags": {c: int(g[c].sum()) for c in flag_cols if g[c].sum() > 0},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refuge-root")
    ap.add_argument("--drishti-root")
    ap.add_argument("--include-refuge-test", action="store_true")
    ap.add_argument("--drishti-thresh", type=float, default=0.75)
    ap.add_argument("--out", default="artifacts/data_audit")
    ap.add_argument("--sheet-seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Indexing:")
    rows = []
    if args.refuge_root:
        rows += index_refuge(args.refuge_root, args.include_refuge_test)
    if args.drishti_root:
        rows += index_drishti(args.drishti_root)
    if not rows:
        raise SystemExit("No image/mask pairs found -- check the --*-root paths.")

    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"\nAuditing {len(rows)} pairs ...")
    df = audit(rows, args.drishti_thresh)
    df.to_csv(out / "audit.csv", index=False)

    summary = summarise(df)
    summary["_config"] = {"drishti_thresh": args.drishti_thresh,
                          "include_refuge_test": args.include_refuge_test}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2))

    for domain in sorted(df.domain.unique()):
        contact_sheet(df, domain, out / f"contact_sheet_{domain}.png",
                      args.drishti_thresh, seed=args.sheet_seed)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote -> {out.resolve()}")
    print("NOW OPEN EVERY contact_sheet_*.png AND LOOK AT IT. "
          "Green must be the rim, orange must sit inside it.")


if __name__ == "__main__":
    main()