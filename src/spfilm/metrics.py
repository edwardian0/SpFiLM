from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


CHANNEL_NAMES = ("disc", "cup")

PER_IMAGE_FIELDNAMES = (
    "image_id",
    "structure",
    "dice",
    "iou",
    "hd95",
    "acc",
    "tp",
    "fp",
    "fn",
    "tn",
)

def metric_frame(image_size: int) -> str:
    """Self-documenting statement of the grid every metric below is measured in.
    1 letterbox pixel = 2124/512 ≈ 4.15 native pixels.
    """

    return (
        f"metrics computed on the {image_size}px aspect-preserving letterboxed "
        "full-image grid (whole fundus resized to fit and centre-pasted on a "
        f"{image_size}x{image_size} canvas, so part of the vertical extent is zero "
        "padding); predictions are never resampled back to native resolution, so "
        f"HD95 is in {image_size}x{image_size} letterboxed-grid pixels, not "
        "millimetres and not native pixels"
    )

DEGENERATE_POLICY = {
    "empty_prediction_non_empty_target": (
        "Dice and IoU are ~0 from the smoothed overlap formula and are kept in the "
        "means; HD95 is undefined (no predicted surface) so the image is excluded "
        "from the HD95 mean and counted in hd95_excluded_empty_prediction"
    ),
    "empty_target": (
        "Dice and IoU are ~0 unless the prediction is also empty; HD95 is undefined "
        "(no reference surface) so the image is excluded from the HD95 mean and "
        "counted in hd95_excluded_empty_target"
    ),
    "both_empty": (
        "Dice and IoU are exactly 1.0 (agreement on absence) and are kept in the "
        "means; HD95 is undefined and the image is excluded from the HD95 mean and "
        "counted in hd95_excluded_both_empty"
    ),
    "csv_encoding": (
        "an excluded HD95 is written to the per-image CSV as the literal 'nan' and "
        "is dropped before any mean, std, or median is taken; no inf ever reaches a "
        "summary and no undefined HD95 is silently reported as 0"
    ),
}


def per_image_overlap(
    predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image, per-channel Dice and IoU tensors."""

    predictions = predictions.to(torch.bool)
    targets = targets.to(torch.bool)
    dimensions = (2, 3)
    intersection = (predictions & targets).sum(dim=dimensions).to(torch.float64)
    predicted = predictions.sum(dim=dimensions).to(torch.float64)
    actual = targets.sum(dim=dimensions).to(torch.float64)
    dice = (2 * intersection + smooth) / (predicted + actual + smooth)
    union = predicted + actual - intersection
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def per_image_confusion(
    predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-image, per-channel TP, FP, FN and TN pixel counts."""

    predictions = predictions.to(torch.bool)
    targets = targets.to(torch.bool)
    dimensions = (2, 3)
    true_positive = (predictions & targets).sum(dim=dimensions)
    false_positive = (predictions & ~targets).sum(dim=dimensions)
    false_negative = (~predictions & targets).sum(dim=dimensions)
    true_negative = (~predictions & ~targets).sum(dim=dimensions)
    return true_positive, false_positive, false_negative, true_negative


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    """Mask pixels that touch the background under 4-connectivity."""

    footprint = generate_binary_structure(mask.ndim, 1)
    return mask ^ binary_erosion(mask, structure=footprint, iterations=1)


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance in grid pixels.

    Returns NaN when either mask is empty: a surface distance needs two surfaces,
    so the value is undefined rather than zero. Callers must exclude NaN before
    averaging (see ``DEGENERATE_POLICY``).
    """

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() or not target.any():
        return math.nan

    # Distances only involve pixels of the two masks, so cropping to their shared
    # bounding box (padded by one background pixel so erosion sees the same
    # neighbourhood as it would in the full frame) is exact and much cheaper.
    union = prediction | target
    rows, columns = np.nonzero(union)
    row_slice = slice(max(rows.min() - 1, 0), rows.max() + 2)
    column_slice = slice(max(columns.min() - 1, 0), columns.max() + 2)
    prediction = prediction[row_slice, column_slice]
    target = target[row_slice, column_slice]

    prediction_surface = _binary_surface(prediction)
    target_surface = _binary_surface(target)
    to_target = distance_transform_edt(~target_surface)[prediction_surface]
    to_prediction = distance_transform_edt(~prediction_surface)[target_surface]
    return float(
        max(np.percentile(to_target, 95), np.percentile(to_prediction, 95))
    )


@dataclass
class OverlapAccumulator:
    threshold: float = 0.5
    dice_batches: list[torch.Tensor] = field(default_factory=list)
    iou_batches: list[torch.Tensor] = field(default_factory=list)
    rows: list[dict[str, object]] = field(default_factory=list)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        image_ids: Sequence[str] | None = None,
    ) -> None:
        # Thresholded masks move to the CPU once: HD95 needs numpy arrays anyway,
        # and per_image_overlap accumulates in float64, which MPS cannot hold.
        predictions = (torch.sigmoid(logits) >= self.threshold).cpu()
        targets = (targets >= 0.5).cpu()
        dice, iou = per_image_overlap(predictions, targets)
        self.dice_batches.append(dice)
        self.iou_batches.append(iou)

        true_positive, false_positive, false_negative, true_negative = (
            per_image_confusion(predictions, targets)
        )
        prediction_masks = predictions.numpy()
        target_masks = targets.numpy()
        if image_ids is None:
            offset = len(self.rows) // len(CHANNEL_NAMES)
            image_ids = [
                f"sample_{offset + index:05d}" for index in range(dice.shape[0])
            ]
        if len(image_ids) != dice.shape[0]:
            raise ValueError(
                f"Got {len(image_ids)} image ids for {dice.shape[0]} images"
            )
        for index, image_id in enumerate(image_ids):
            for channel, name in enumerate(CHANNEL_NAMES):
                pixels = int(prediction_masks[index, channel].size)
                correct = int(true_positive[index, channel]) + int(
                    true_negative[index, channel]
                )
                self.rows.append(
                    {
                        "image_id": image_id,
                        "structure": name,
                        "dice": float(dice[index, channel]),
                        "iou": float(iou[index, channel]),
                        "hd95": hd95(
                            prediction_masks[index, channel],
                            target_masks[index, channel],
                        ),
                        "acc": correct / pixels,
                        "tp": int(true_positive[index, channel]),
                        "fp": int(false_positive[index, channel]),
                        "fn": int(false_negative[index, channel]),
                        "tn": int(true_negative[index, channel]),
                    }
                )

    def write_per_image_csv(self, output_path: str | Path) -> Path:
        """Write one row per image and structure; this file is the summary source."""

        if not self.rows:
            raise RuntimeError("Cannot write per-image metrics before any samples")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PER_IMAGE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        return output_path

    def compute(self) -> dict[str, dict[str, float | int | None]]:
        if not self.rows:
            raise RuntimeError("Cannot compute metrics before any samples were added")
        return summarise_per_image_rows(self.rows)


def summarise_per_image_rows(
    rows: Sequence[dict[str, object]]
) -> dict[str, dict[str, float | int | None]]:
    """Reduce per-image rows to per-structure summaries; disc and cup stay separate."""

    result: dict[str, dict[str, float | int | None]] = {}
    for name in CHANNEL_NAMES:
        structure_rows = [row for row in rows if row["structure"] == name]
        if not structure_rows:
            continue
        dice = np.array([float(row["dice"]) for row in structure_rows])
        iou = np.array([float(row["iou"]) for row in structure_rows])
        accuracy = np.array([float(row["acc"]) for row in structure_rows])
        distances = np.array([float(row["hd95"]) for row in structure_rows])
        finite = distances[np.isfinite(distances)]
        empty_prediction = np.array(
            [int(row["tp"]) + int(row["fp"]) == 0 for row in structure_rows]
        )
        empty_target = np.array(
            [int(row["tp"]) + int(row["fn"]) == 0 for row in structure_rows]
        )
        result[name] = {
            "dice_mean": float(np.mean(dice)),
            "dice_std": float(np.std(dice, ddof=0)),
            "dice_median": float(np.median(dice)),
            "iou_mean": float(np.mean(iou)),
            "iou_std": float(np.std(iou, ddof=0)),
            "accuracy_mean": float(np.mean(accuracy)),
            "accuracy_std": float(np.std(accuracy, ddof=0)),
            "hd95_mean": float(np.mean(finite)) if finite.size else None,
            "hd95_std": float(np.std(finite, ddof=0)) if finite.size else None,
            "hd95_median": float(np.median(finite)) if finite.size else None,
            "hd95_sample_count": int(finite.size),
            "hd95_excluded_count": int(distances.size - finite.size),
            "hd95_excluded_empty_prediction": int(
                np.sum(empty_prediction & ~empty_target)
            ),
            "hd95_excluded_empty_target": int(np.sum(empty_target & ~empty_prediction)),
            "hd95_excluded_both_empty": int(np.sum(empty_prediction & empty_target)),
            "tp_total": int(sum(int(row["tp"]) for row in structure_rows)),
            "fp_total": int(sum(int(row["fp"]) for row in structure_rows)),
            "fn_total": int(sum(int(row["fn"]) for row in structure_rows)),
            "tn_total": int(sum(int(row["tn"]) for row in structure_rows)),
            "sample_count": len(structure_rows),
        }
    return result


def read_per_image_csv(csv_path: str | Path) -> list[dict[str, object]]:
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarise_per_image_csv(
    csv_path: str | Path
) -> dict[str, dict[str, float | int | None]]:
    """Summarise straight from the written CSV so the report and the file agree."""

    return summarise_per_image_rows(read_per_image_csv(csv_path))

