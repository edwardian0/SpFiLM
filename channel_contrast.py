#!/usr/bin/env python3
"""How much optic-disc and cup signal each colour channel actually carries.

The domain-shift histograms say how far apart the datasets sit in each channel.
They do not say whether a gap in a given channel *matters*, because they never
look at the structures being segmented. A large gap in a channel that carries no
disc/cup contrast should be harmless; a small gap in the channel the model relies
on should not be.

This measures the second half of that question directly, with no model involved.
For every image it takes three regions - cup, neuroretinal rim (disc minus cup),
and the retina outside the disc but inside the field of view - and reports how
separable they are in each channel:

    d' = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)

which is the region difference in units of the within-region spread, so it is
comparable between channels whose absolute levels differ by a factor of five.
Regions are eroded by one pixel first: cup and rim share a boundary, and at this
working size a boundary ring would otherwise be a noticeable share of the cup.

Sign is kept. A positive disc-vs-retina d' means the disc is the brighter of the
two in that channel.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

from analyze_domain_shift import _write_csv  # noqa: E402
from spfilm.data import FundusRecord, decode_mask_channels  # noqa: E402
from spfilm.global_histograms import (  # noqa: E402
    CHANNELS,
    FOV_LUMINANCE_THRESHOLD,
    LUMA_WEIGHTS,
    load_pixels,
    sample_images,
)
from spfilm.stage3 import (  # noqa: E402
    Stage3ConfigError,
    Stage3DataError,
    Stage3LodoConfig,
    discover_lodo_records,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "domain_shift"
CONTRASTS = ("disc_vs_retina", "cup_vs_rim")
MIN_REGION_PIXELS = 40


def _bands(pixels: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "gray": pixels @ np.asarray(LUMA_WEIGHTS, dtype=np.float32),
        "red": pixels[..., 0],
        "green": pixels[..., 1],
        "blue": pixels[..., 2],
    }


def _dprime(band: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Signed separability of two regions, or NaN if either is too small."""

    if a.sum() < MIN_REGION_PIXELS or b.sum() < MIN_REGION_PIXELS:
        return float("nan")
    left, right = band[a], band[b]
    spread = np.sqrt((left.var() + right.var()) / 2.0)
    if spread <= 0:
        return float("nan")
    return float((left.mean() - right.mean()) / spread)


def _image_contrast(payload: tuple[FundusRecord, int]) -> dict[str, float]:
    record, working_size = payload
    pixels = load_pixels(record.image_path, working_size)
    height, width = pixels.shape[:2]

    # Masks are native resolution; the image was thumbnailed while decoding, so
    # the mask is brought to the image rather than the reverse. Nearest keeps the
    # labels binary.
    masks = decode_mask_channels(record)
    resized = [
        np.asarray(
            Image.fromarray(channel.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        )
        > 127
        for channel in masks
    ]
    disc, cup = resized[0], resized[1]

    luminance = pixels @ np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    fov = luminance > FOV_LUMINANCE_THRESHOLD
    regions = {
        "disc": disc,
        "cup": cup,
        "rim": disc & ~cup,
        "retina": fov & ~disc,
    }
    eroded = {
        name: binary_erosion(mask, iterations=1, border_value=0)
        for name, mask in regions.items()
    }

    row: dict[str, float] = {}
    for channel, band in _bands(pixels).items():
        row[f"disc_vs_retina|{channel}"] = _dprime(
            band, eroded["disc"], eroded["retina"]
        )
        row[f"cup_vs_rim|{channel}"] = _dprime(band, eroded["cup"], eroded["rim"])
        # Absolute level of each region, so a d' near zero can be read as "no
        # difference" rather than "no dynamic range".
        for name in ("cup", "rim", "retina"):
            mask = eroded[name]
            row[f"mean|{name}|{channel}"] = (
                float(band[mask].mean() * 255.0) if mask.sum() else float("nan")
            )
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--working-size",
        type=int,
        default=512,
        help=(
            "Larger than the histogram sweep's 256: the cup is under 0.5%% of the "
            "frame in REFUGE, and needs the pixels to survive erosion"
        ),
    )
    parser.add_argument("--sample", type=int, help="Images per dataset; default all")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = Stage3LodoConfig.from_json(args.config.expanduser().resolve())
        records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
    except (Stage3ConfigError, Stage3DataError, OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for domain in sorted(records_by_domain, key=lambda item: item.value):
        records = list(records_by_domain[domain])
        if args.sample is not None:
            records = sample_images(
                domain.value, records, args.sample, args.sample_seed
            )
        print(
            f"measuring {domain.value}: {len(records)} images at "
            f"{args.working_size}px",
            flush=True,
        )
        payloads = [(record, args.working_size) for record in records]
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                measured = list(pool.map(_image_contrast, payloads, chunksize=4))
        else:
            measured = [_image_contrast(payload) for payload in payloads]

        for contrast in CONTRASTS:
            for channel in CHANNELS:
                values = np.asarray(
                    [row[f"{contrast}|{channel}"] for row in measured], dtype=float
                )
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "domain": domain.value,
                        "contrast": contrast,
                        "channel": channel,
                        "images": len(records),
                        "images_measured": int(finite.size),
                        "dprime_mean": round(float(finite.mean()), 4),
                        "dprime_sd": round(float(finite.std(ddof=1)), 4),
                        "dprime_abs_mean": round(float(np.abs(finite).mean()), 4),
                    }
                )
        for name in ("cup", "rim", "retina"):
            for channel in CHANNELS:
                values = np.asarray(
                    [row[f"mean|{name}|{channel}"] for row in measured], dtype=float
                )
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "domain": domain.value,
                        "contrast": f"level_{name}",
                        "channel": channel,
                        "images": len(records),
                        "images_measured": int(finite.size),
                        "dprime_mean": round(float(finite.mean()), 4),
                        "dprime_sd": round(float(finite.std(ddof=1)), 4),
                        "dprime_abs_mean": round(float(np.abs(finite).mean()), 4),
                    }
                )

    output_dir = args.output_dir.expanduser().resolve()
    path = _write_csv(output_dir / "channel_contrast.csv", rows)

    print()
    print("region separability d' (mean over images; sign kept)")
    for contrast in CONTRASTS:
        print(f"  {contrast}")
        print(f"    {'domain':20s}" + "".join(f"{c:>9s}" for c in CHANNELS))
        for domain_name in sorted({str(row['domain']) for row in rows}):
            selected = {
                str(row["channel"]): row
                for row in rows
                if row["domain"] == domain_name and row["contrast"] == contrast
            }
            print(
                f"    {domain_name:20s}"
                + "".join(f"{selected[c]['dprime_mean']:9.2f}" for c in CHANNELS)
            )
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
