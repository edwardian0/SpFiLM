"""Draw predicted disc/cup contours on original-resolution fundus images.

The domain-shift brief concluded that the shift breaking cross-domain transfer is
geometric rather than photometric, on aggregate statistics alone. This script
exists to put images next to that claim: if the account is right, the failures
should look like boundary and scale errors -- contours the right shape in the
wrong place or at the wrong size -- and not like the model mistaking bright
retina for disc.

Two things separate it from `save_prediction_gallery`, which training calls:
predictions are drawn on the native image rather than the letterboxed 512x512
network input, and instances are chosen by their per-image Dice rather than by
evenly spaced index.

Usage:

    .spfilm/bin/python visualize_predictions.py \
        --run-dir artifacts/runs/single_s3_refuge_zeiss_seed_42_36984711 \
        --target-domain drishti_gs --worst 3 --best 2 \
        --output-dir artifacts/prediction_overlays/zeiss_to_drishti
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from spfilm.data import FundusRecord, FundusSegmentationDataset, decode_mask_channels
from spfilm.metrics import per_image_overlap
from spfilm.model import PlainUNet
from spfilm.visualization import (
    binary_mask_image,
    invert_letterbox,
    overlay_contours,
    overlay_regions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_MARKER = "/datasets/"

# Ground truth keeps the house palette from `_overlay`; the prediction gets a
# contrasting pair so both can share the zoom panel.
TRUTH_COLORS = ((0.1, 1.0, 0.2), (0.1, 0.5, 1.0))
PREDICTION_COLORS = ((1.0, 0.85, 0.1), (1.0, 0.2, 0.85))
STRUCTURES = ("disc", "cup")


class VisualizationError(RuntimeError):
    """A failure that must stop the run rather than yield a misleading figure."""


# --- Run inputs ----------------------------------------------------------------


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise VisualizationError(f"Cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise VisualizationError(f"{path} is not valid JSON: {error}") from error


def _resolve_config(run_dir: Path, checkpoint: dict) -> dict:
    """Prefer the checkpoint's own config, cross-checked against the run dir.

    Pulling weights from CREATE by hand makes it easy to pair a checkpoint with
    the wrong run directory. The two configs disagreeing is the symptom, and a
    figure built from a mismatched pair would be silently wrong, so it is fatal.
    """

    embedded = checkpoint.get("config")
    on_disk_path = run_dir / "resolved_config.json"
    on_disk = _load_json(on_disk_path) if on_disk_path.is_file() else None
    if embedded is None and on_disk is None:
        raise VisualizationError(
            f"No config in the checkpoint and no {on_disk_path}; cannot construct "
            "the model (base_channels defaults to 32 but Stage 3 used 16)"
        )
    if embedded is None:
        return on_disk
    if on_disk is not None:
        for key in ("image_size", "base_channels", "threshold", "experiment_name"):
            mine, theirs = embedded.get(key), on_disk.get(key)
            if mine != theirs:
                raise VisualizationError(
                    f"Checkpoint config disagrees with {on_disk_path} on {key!r}: "
                    f"checkpoint={mine!r}, run dir={theirs!r}. The checkpoint most "
                    "likely belongs to a different run -- check what you pulled."
                )
    return embedded


def _describe_cell(run_dir: Path, target_domain: str | None) -> tuple[str, str]:
    """Return ``(cell, source_label)`` from whichever protocol JSON the run has."""

    target = target_domain or "source test set"
    for filename, key, field in (
        ("single_source_run.json", "single_source", "source_domain"),
        ("fixed_lodo_run.json", "fixed_lodo", "held_out_domain"),
        ("lodo_run.json", "lodo", "held_out_domain"),
    ):
        path = run_dir / filename
        if not path.is_file():
            continue
        payload = _load_json(path).get(key, {})
        domain = payload.get(field)
        if not domain:
            continue
        if field == "source_domain":
            return f"{domain} -> {target}", f"trained on {domain}"
        return f"LODO(held out {domain}) -> {target}", f"LODO, held out {domain}"
    return f"{run_dir.name} -> {target}", run_dir.name


# --- Manifest ------------------------------------------------------------------


def _remap(path_text: str, data_root: Path) -> Path:
    """Rewrite a CREATE dataset path onto the local dataset root."""

    index = path_text.find(DATASET_MARKER)
    if index == -1:
        candidate = Path(path_text)
        if candidate.is_file():
            return candidate
        raise VisualizationError(
            f"Path has no {DATASET_MARKER!r} segment to remap and does not exist "
            f"locally: {path_text}"
        )
    resolved = data_root / path_text[index + len(DATASET_MARKER) :]
    if not resolved.is_file():
        raise VisualizationError(
            f"Missing local file: {resolved}\n  (manifest recorded {path_text})\n"
            f"  Check --data-root, currently {data_root}"
        )
    return resolved


def _record_from_row(row: dict[str, str], data_root: Path) -> FundusRecord:
    mask_paths = [
        _remap(part, data_root) for part in row["mask_paths"].split("|") if part
    ]
    encoding = row["mask_encoding"]
    fields: dict[str, object] = {}
    if len(mask_paths) == 1:
        fields["combined_mask_path"] = mask_paths[0]
    elif len(mask_paths) == 2:
        fields["disc_mask_path"], fields["cup_mask_path"] = mask_paths
    else:
        raise VisualizationError(
            f"{row['sample_id']} has {len(mask_paths)} mask paths; expected 1 or 2"
        )
    record = FundusRecord(
        sample_id=row["sample_id"],
        domain=row["domain"],
        image_path=_remap(row["image_path"], data_root),
        mask_encoding=encoding,
        stratum=row.get("stratum", "all"),
        split_hint=row.get("split"),
        **fields,  # type: ignore[arg-type]
    )
    if encoding == "rim_one_dl_foreground_high":
        # `decode_mask_channels` cross-checks its cup-outside-disc count against the
        # count discovery recorded on the record. The manifest does not carry that
        # number, so re-derive it here. This turns a provenance guard into a no-op
        # for this script -- acceptable, because the guard protects against drift
        # between discovery and training, which a read-only overlay cannot cause.
        disc = np.asarray(Image.open(record.disc_mask_path).convert("L")) >= 128
        cup = np.asarray(Image.open(record.cup_mask_path).convert("L")) >= 128
        record = FundusRecord(
            **{
                **record.__dict__,
                "source_cup_repair_pixels": int(np.count_nonzero(cup & ~disc)),
            }
        )
    return record


def _read_manifest(
    run_dir: Path, data_root: Path, target_domain: str | None
) -> list[FundusRecord]:
    path = run_dir / "split_manifest.csv"
    if not path.is_file():
        raise VisualizationError(f"Run directory has no split manifest: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    if target_domain is not None:
        available = sorted({row["domain"] for row in rows})
        rows = [row for row in rows if row["domain"] == target_domain]
        if not rows:
            raise VisualizationError(
                f"No test rows for domain {target_domain!r} in {path}; "
                f"the manifest's test domains are {available}"
            )
    return [_record_from_row(row, data_root) for row in rows]


# --- Selection -----------------------------------------------------------------


def _read_per_image_metrics(run_dir: Path, target_domain: str | None) -> dict:
    name = (
        "test_per_image_metrics.csv"
        if target_domain is None
        else f"test_{target_domain}_per_image_metrics.csv"
    )
    path = run_dir / name
    if not path.is_file():
        available = sorted(p.name for p in run_dir.glob("test*per_image_metrics.csv"))
        raise VisualizationError(
            f"No per-image metrics at {path}.\n  Available in this run: {available}"
        )
    # Values are per-structure Dice floats, plus a "native_size" tuple where the
    # domain records one.
    table: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            entry = table.setdefault(row["image_id"], {})
            entry[row["structure"]] = float(row["dice"])
            if row.get("native_width") and row.get("native_height"):
                entry["native_size"] = (
                    int(row["native_width"]),
                    int(row["native_height"]),
                )
    if not table:
        raise VisualizationError(f"{path} has no rows")
    return table


def _select(
    metrics: dict[str, dict],
    available: set[str],
    structure: str,
    args: argparse.Namespace,
) -> list[tuple[str, str]]:
    """Return ``(image_id, selector)`` pairs, in the order they were requested."""

    # Sorting on a missing structure would put NaN in the key, and NaN compares
    # false against everything, so the ordering would be quietly arbitrary rather
    # than wrong-looking. Drop those rows instead and say so.
    scored = [
        image_id
        for image_id in metrics
        if image_id in available and structure in metrics[image_id]
    ]
    skipped = len(set(metrics) & available) - len(scored)
    if skipped:
        print(
            f"warning: {skipped} test image(s) have no {structure!r} row in the "
            "per-image metrics and cannot be ranked",
            file=sys.stderr,
        )
    ranked = sorted(scored, key=lambda image_id: (metrics[image_id][structure], image_id))
    if not ranked:
        raise VisualizationError(
            "No image appears in both the per-image metrics and the manifest's "
            "test split -- check --target-domain against the metrics CSV"
        )

    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(image_id: str, selector: str) -> None:
        if image_id not in seen:
            seen.add(image_id)
            chosen.append((image_id, selector))

    for image_id in _split_ids(args.image_ids):
        if image_id not in available:
            raise VisualizationError(
                f"--image-ids named {image_id!r}, which is not in the test split "
                f"for this run/domain"
            )
        add(image_id, "explicit")
    for image_id in ranked[: args.worst]:
        add(image_id, f"worst_{structure}")
    for image_id in reversed(ranked[len(ranked) - args.best :]) if args.best else []:
        add(image_id, f"best_{structure}")
    if args.median:
        middle = len(ranked) // 2
        start = max(0, middle - args.median // 2)
        for image_id in ranked[start : start + args.median]:
            add(image_id, f"median_{structure}")
    if args.random:
        rng = random.Random(args.seed)
        for image_id in rng.sample(ranked, k=min(args.random, len(ranked))):
            add(image_id, f"random_seed{args.seed}")
    if not chosen:
        raise VisualizationError(
            "No instances selected: pass at least one of --worst/--best/--median/"
            "--random/--image-ids"
        )
    return chosen


def _split_ids(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


# --- Geometry read-outs --------------------------------------------------------


def _equivalent_diameter(mask: np.ndarray) -> float:
    """Diameter of the disc of equal area, in pixels of the mask's own grid."""

    return float(2.0 * np.sqrt(np.count_nonzero(mask) / np.pi))


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(columns.mean())


def _dice(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Per-structure Dice using the same arithmetic as the per-image CSV."""

    dice, _ = per_image_overlap(
        torch.from_numpy(np.asarray(prediction, dtype=bool))[None],
        torch.from_numpy(np.asarray(target, dtype=bool))[None],
    )
    return float(dice[0, 0]), float(dice[0, 1])


# --- Figure --------------------------------------------------------------------


def _extent(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Return ``(row_centre, column_centre, span)`` of a mask's bounding box."""

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return (
        (rows.min() + rows.max()) / 2.0,
        (columns.min() + columns.max()) / 2.0,
        float(max(rows.max() - rows.min(), columns.max() - columns.min())),
    )


def _zoom_window(
    truth_disc: np.ndarray,
    prediction_disc: np.ndarray,
    shape: tuple[int, int],
    max_growth: float = 3.0,
) -> tuple[slice, slice]:
    """A square window on the ground-truth disc, widened to show the prediction.

    The truth disc is the anchor because it is always a single well-formed blob.
    Anchoring on the union instead lets one scattered prediction -- exactly what a
    collapsed cross-domain model produces -- open the window to the whole frame and
    show nothing. The window still grows to include a displaced or oversized
    prediction, up to `max_growth`, so a genuine scale error stays visible; beyond
    that the full-frame prediction panel is the place to look.
    """

    height, width = shape
    anchor = _extent(truth_disc) or _extent(prediction_disc)
    if anchor is None:
        return slice(0, height), slice(0, width)
    row_centre, column_centre, span = anchor
    half = max(24.0, span * 0.85)

    predicted = _extent(prediction_disc)
    if predicted is not None:
        reach = max(
            abs(predicted[0] - row_centre) + predicted[2] / 2.0,
            abs(predicted[1] - column_centre) + predicted[2] / 2.0,
        )
        half = max(half, min(reach * 1.15, half * max_growth))

    top = int(max(0, min(height - 1, row_centre - half)))
    bottom = int(max(top + 1, min(height, row_centre + half)))
    left = int(max(0, min(width - 1, column_centre - half)))
    right = int(max(left + 1, min(width, column_centre + half)))
    return slice(top, bottom), slice(left, right)


def _save_figure(
    image: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str,
    contour_width: int,
    dpi: int,
    style: str,
    alpha: float,
) -> None:
    def paint(canvas, masks, colors):
        if style == "contour":
            return overlay_contours(canvas, masks, colors, contour_width)
        return overlay_regions(
            canvas,
            masks,
            colors,
            alpha=alpha,
            contour_width=contour_width if style == "both" else 0,
        )

    truth_panel = paint(image, truth, TRUTH_COLORS)
    prediction_panel = paint(image, prediction, TRUTH_COLORS)
    combined = paint(
        image,
        [truth[0], truth[1], prediction[0], prediction[1]],
        [*TRUTH_COLORS, *PREDICTION_COLORS],
    )
    rows, columns = _zoom_window(truth[0], prediction[0], image.shape[:2])

    height, width = image.shape[:2]
    panel_height = 6.0 * height / width
    figure, axes = plt.subplots(
        2, 2, figsize=(13.0, 1.0 + 2 * min(7.5, max(3.0, panel_height)))
    )
    panels = (
        (image, "original image (native resolution)"),
        (truth_panel, "ground truth: disc green, cup blue"),
        (prediction_panel, "PREDICTION: disc green, cup blue"),
        (
            combined[rows, columns],
            "zoom: truth green/blue vs prediction yellow/magenta",
        ),
    )
    for axis, (panel, panel_title) in zip(axes.flat, panels):
        axis.imshow(np.clip(panel, 0, 1), vmin=0, vmax=1)
        axis.set_title(panel_title, fontsize=9)
        axis.axis("off")
    axes[1, 1].legend(
        handles=[
            Line2D([], [], color=TRUTH_COLORS[0], lw=2, label="truth disc"),
            Line2D([], [], color=TRUTH_COLORS[1], lw=2, label="truth cup"),
            Line2D([], [], color=PREDICTION_COLORS[0], lw=2, label="pred disc"),
            Line2D([], [], color=PREDICTION_COLORS[1], lw=2, label="pred cup"),
        ],
        loc="lower right",
        fontsize=7,
        framealpha=0.85,
    )
    figure.suptitle(title, fontsize=12)
    figure.text(0.5, 0.945, subtitle, ha="center", fontsize=8.5, color="#444444")
    figure.tight_layout(rect=(0, 0, 1, 0.935))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


# --- Main ----------------------------------------------------------------------


def _load_checkpoint(path: Path, run_dir: Path) -> dict:
    if not path.is_file():
        recorded = ""
        for filename in ("single_source_run.json", "fixed_lodo_run.json", "lodo_run.json"):
            candidate = run_dir / filename
            if candidate.is_file():
                remote = _load_json(candidate).get("artifacts", {}).get("checkpoint")
                if remote:
                    recorded = f"\n  This run recorded its weights on CREATE at:\n    {remote}"
                    break
        raise VisualizationError(
            f"No checkpoint at {path}.{recorded}\n"
            "  Runs synced back from CREATE did not bring their weights; pull the "
            "one you need with scp, or run this script on CREATE."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument(
        "--target-domain",
        default=None,
        help="Domain whose per-image CSV drives selection; omit for the run's own "
        "test set (test_per_image_metrics.csv)",
    )
    parser.add_argument("--structure", choices=STRUCTURES, default="disc")
    parser.add_argument("--worst", type=int, default=0)
    parser.add_argument("--best", type=int, default=0)
    parser.add_argument("--median", type=int, default=0)
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--image-ids", default=None, help="Comma-separated sample ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--contour-width",
        type=int,
        default=0,
        help="Contour thickness in native pixels; 0 scales it with image size",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--style",
        choices=("contour", "filled", "both"),
        default="both",
        help="How masks are drawn: outlines only, filled regions, or filled with "
        "an outline on top",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Fill opacity for --style filled/both",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also write the raw binary prediction mask as a separate PNG",
    )
    parser.add_argument(
        "--dice-tolerance",
        type=float,
        default=0.05,
        help="Maximum |recomputed grid Dice - CSV Dice| before the run aborts",
    )
    parser.add_argument(
        "--roundtrip-tolerance",
        type=float,
        default=0.95,
        help="Minimum letterbox round-trip Dice on the ground-truth disc",
    )
    parser.add_argument(
        "--no-fail-checks",
        action="store_true",
        help="Report the sanity checks but do not abort on them",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise VisualizationError(f"--run-dir is not a directory: {run_dir}")
    checkpoint_path = (args.checkpoint or run_dir / "best_model.pt").resolve()
    checkpoint = _load_checkpoint(checkpoint_path, run_dir)
    config = _resolve_config(run_dir, checkpoint)
    image_size = int(config["image_size"])
    threshold = float(config["threshold"])
    base_channels = int(config["base_channels"])
    cell, source_label = _describe_cell(run_dir, args.target_domain)

    if checkpoint.get("channel_order") not in (None, ["disc", "cup"]):
        raise VisualizationError(
            f"Checkpoint channel order is {checkpoint['channel_order']}, but the "
            "overlay palette assumes ['disc', 'cup']"
        )

    model = PlainUNet(base_channels=base_channels)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError) as error:
        raise VisualizationError(
            f"Cannot load weights from {checkpoint_path}: {error}"
        ) from error
    device = torch.device(args.device)
    model.to(device).eval()

    data_root = args.data_root.resolve()
    records = _read_manifest(run_dir, data_root, args.target_domain)
    by_id = {record.sample_id: record for record in records}
    metrics = _read_per_image_metrics(run_dir, args.target_domain)
    selected = _select(metrics, set(by_id), args.structure, args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    failures: list[str] = []

    print(f"cell: {cell}")
    print(f"checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    print(
        f"config: image_size={image_size} threshold={threshold} "
        f"base_channels={base_channels}"
    )
    print(f"selected {len(selected)} instance(s) from {len(by_id)} test images\n")

    for image_id, selector in selected:
        record = by_id[image_id]
        dataset = FundusSegmentationDataset([record], image_size=image_size, augment=False)
        image_tensor, target_tensor, metadata = dataset[0]
        with torch.inference_mode():
            logits = model(image_tensor.unsqueeze(0).to(device))[0].cpu()
        grid_prediction = (torch.sigmoid(logits) >= threshold).numpy()
        grid_target = target_tensor.numpy().astype(bool)

        native_image = Image.open(record.image_path).convert("RGB")
        native_size = native_image.size
        native_truth = decode_mask_channels(record).astype(bool)
        if native_truth.shape[1:] != (native_size[1], native_size[0]):
            raise VisualizationError(
                f"{image_id}: native mask {native_truth.shape[1:]} does not match "
                f"image {(native_size[1], native_size[0])}"
            )
        native_prediction = invert_letterbox(grid_prediction, native_size, image_size)

        # Sanity checks, per the handoff. (1) The grid Dice we recompute must match
        # the number already in the CSV -- that is what proves this checkpoint and
        # this image belong together. (2) Letterboxing the ground truth and
        # inverting it must return the original mask; a systematic loss here means
        # the offset has x and y swapped. (3) Native Dice is reported for context,
        # and is expected to differ slightly: the CSV is computed in the grid.
        grid_dice = _dice(grid_prediction, grid_target)
        native_dice = _dice(native_prediction, native_truth)
        roundtrip = _dice(invert_letterbox(grid_target, native_size, image_size), native_truth)
        csv_dice = metrics[image_id]

        for index, structure in enumerate(STRUCTURES):
            expected = csv_dice.get(structure)
            if expected is None:
                continue
            gap = abs(grid_dice[index] - expected)
            if gap > args.dice_tolerance:
                failures.append(
                    f"{image_id} {structure}: recomputed grid Dice "
                    f"{grid_dice[index]:.4f} vs CSV {expected:.4f} (gap {gap:.4f})"
                )
        recorded = csv_dice.get("native_size")
        if recorded is not None and recorded != native_size:
            failures.append(
                f"{image_id}: image is {native_size} but the run recorded "
                f"{recorded} -- the manifest and the metrics disagree on this image"
            )
        if roundtrip[0] < args.roundtrip_tolerance:
            failures.append(
                f"{image_id}: letterbox round-trip disc Dice {roundtrip[0]:.4f} "
                f"below {args.roundtrip_tolerance} -- inversion geometry is wrong"
            )

        truth_diameter = _equivalent_diameter(native_truth[0])
        prediction_diameter = _equivalent_diameter(native_prediction[0])
        truth_centroid = _centroid(native_truth[0])
        prediction_centroid = _centroid(native_prediction[0])
        displacement = (
            float(
                np.hypot(
                    prediction_centroid[0] - truth_centroid[0],
                    prediction_centroid[1] - truth_centroid[1],
                )
            )
            if truth_centroid and prediction_centroid
            else float("nan")
        )
        scale_ratio = (
            prediction_diameter / truth_diameter if truth_diameter > 0 else float("nan")
        )

        contour_width = args.contour_width or max(
            1, round(max(native_size) / 500.0)
        )
        output_path = output_dir / f"{image_id}__{selector}.png"
        image_array = np.asarray(native_image, dtype=np.float32) / 255.0
        _save_figure(
            image_array,
            native_truth,
            native_prediction,
            output_path,
            title=f"{image_id}  |  {cell}",
            subtitle=(
                f"{source_label} | {record.domain} | {native_size[0]}x{native_size[1]} px | "
                f"CSV Dice disc {csv_dice.get('disc', float('nan')):.3f} "
                f"cup {csv_dice.get('cup', float('nan')):.3f} | "
                f"disc diameter truth {truth_diameter:.0f}px vs pred "
                f"{prediction_diameter:.0f}px (x{scale_ratio:.2f}), "
                f"centroid offset {displacement:.0f}px"
            ),
            contour_width=contour_width,
            dpi=args.dpi,
            style=args.style,
            alpha=args.alpha,
        )
        mask_path = ""
        if args.save_masks:
            # The prediction as a plain label map, no fundus underneath, next to the
            # ground truth in the same frame for a like-for-like shape comparison.
            mask_path = str((output_dir / f"{image_id}__{selector}_masks.png").name)
            side_by_side = np.concatenate(
                [
                    binary_mask_image(native_truth, TRUTH_COLORS),
                    binary_mask_image(native_prediction, TRUTH_COLORS),
                ],
                axis=1,
            )
            Image.fromarray((side_by_side * 255).astype(np.uint8)).save(
                output_dir / mask_path
            )

        print(
            f"  {output_path.name}  csv_disc={csv_dice.get('disc', float('nan')):.3f} "
            f"grid_disc={grid_dice[0]:.3f} native_disc={native_dice[0]:.3f} "
            f"roundtrip={roundtrip[0]:.4f} scale=x{scale_ratio:.2f}"
        )
        index_rows.append(
            {
                "image_id": image_id,
                "selector": selector,
                "cell": cell,
                "run_dir": run_dir.name,
                "checkpoint": str(checkpoint_path),
                "domain": record.domain,
                "stratum": record.stratum,
                "native_width": native_size[0],
                "native_height": native_size[1],
                "csv_dice_disc": csv_dice.get("disc", ""),
                "csv_dice_cup": csv_dice.get("cup", ""),
                "grid_dice_disc": round(grid_dice[0], 6),
                "grid_dice_cup": round(grid_dice[1], 6),
                "native_dice_disc": round(native_dice[0], 6),
                "native_dice_cup": round(native_dice[1], 6),
                "roundtrip_dice_disc": round(roundtrip[0], 6),
                "roundtrip_dice_cup": round(roundtrip[1], 6),
                "truth_disc_diameter_px": round(truth_diameter, 2),
                "pred_disc_diameter_px": round(prediction_diameter, 2),
                "disc_scale_ratio": round(scale_ratio, 4),
                "disc_centroid_offset_px": round(displacement, 2),
                "image_path": str(record.image_path),
                "png": output_path.name,
                "masks_png": mask_path,
            }
        )

    index_path = output_dir / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"\nwrote {len(index_rows)} figure(s) and {index_path}")

    if failures:
        sys.stdout.flush()  # keep the failure list after the per-image log when piped
        message = "sanity checks failed:\n  " + "\n  ".join(failures)
        if args.no_fail_checks:
            print(f"\nWARNING: {message}", file=sys.stderr)
        else:
            raise VisualizationError(
                message + "\n(pass --no-fail-checks to write the figures anyway)"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VisualizationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
