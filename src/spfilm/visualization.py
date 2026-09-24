from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import FundusRecord, FundusSegmentationDataset


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def _overlay(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    overlay = np.clip(image.copy(), 0, 1)
    overlay[_boundary(masks[0])] = (0.1, 1.0, 0.2)
    overlay[_boundary(masks[1])] = (0.1, 0.5, 1.0)
    return overlay


def save_mask_contact_sheet(
    records: Sequence[FundusRecord],
    output_path: str | Path,
    count: int = 12,
    seed: int = 42,
    image_size: int = 320,
) -> Path:
    """Save image, normalized channels, and overlay for a visual mask audit."""

    if not records:
        raise ValueError("Cannot inspect an empty dataset")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected = rng.sample(list(records), k=min(count, len(records)))
    dataset = FundusSegmentationDataset(selected, image_size=image_size, augment=False)
    samples_per_row = 2
    rows = math.ceil(len(selected) / samples_per_row)
    columns = samples_per_row * 4
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")

    for index in range(len(dataset)):
        image_tensor, mask_tensor, metadata = dataset[index]
        image = image_tensor.permute(1, 2, 0).numpy()
        masks = mask_tensor.numpy()
        row = index // samples_per_row
        offset = (index % samples_per_row) * 4
        panels = (
            (image, "image", None),
            (masks[0], "disc", "gray"),
            (masks[1], "cup", "gray"),
            (_overlay(image, masks), "overlay", None),
        )
        for panel_index, (panel, title, color_map) in enumerate(panels):
            axis = axes[row, offset + panel_index]
            axis.imshow(panel, cmap=color_map, vmin=0, vmax=1)
            axis.set_title(
                f"{metadata['sample_id']} - {title}" if panel_index == 0 else title,
                fontsize=9,
            )
            axis.axis("off")

    figure.suptitle(
        "Normalized mask audit - green: disc boundary, blue: cup boundary",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


# --- Training curves ------------------------------------------------------------
#
# The engine rewrites ``history.csv`` after every epoch and redraws
# ``training_curves.png`` every few epochs, so a run can be watched while it trains;
# ``plot_training_curves.py`` renders the same figure from the CSV on demand, for a
# run that is still going, was preempted, or has finished.

HISTORY_FILENAME = "history.csv"
TRAINING_CURVES_FILENAME = "training_curves.png"
HISTORY_REQUIRED_COLUMNS = ("epoch", "train_loss", "val_loss", "val_disc_dice", "val_cup_dice")


def load_history(path: str | Path) -> list[dict[str, float]]:
    """Read a run's ``history.csv`` back into the engine's per-epoch rows."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [{key: float(value) for key, value in row.items()} for row in reader]
    if not rows:
        raise ValueError(f"{path} holds no completed epochs")
    missing = [column for column in HISTORY_REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks history columns {missing}")
    return rows


def best_epoch_from_history(history: Sequence[Mapping[str, float]]) -> int | None:
    """The epoch the engine checkpointed as best.

    The engine resets ``epochs_without_improvement`` to zero exactly when it
    saves ``best_model.pt``, so the last row with a zero counter is that epoch
    under the engine's own min-delta rule -- not merely the argmin of val loss.
    Older histories without the column give ``None``.
    """

    if not history or "epochs_without_improvement" not in history[0]:
        return None
    best = [row["epoch"] for row in history if row["epochs_without_improvement"] == 0]
    return int(best[-1]) if best else None


def early_stop_epoch_from_history(history: Sequence[Mapping[str, float]]) -> int | None:
    """The epoch the early-stopping rule first fired, or ``None`` if it has not."""

    if not history or "would_have_stopped_at_epoch" not in history[-1]:
        return None
    value = history[-1]["would_have_stopped_at_epoch"]
    return int(value) if value >= 0 else None


def _mark_epochs(axis, best_epoch: int | None, stop_epoch: int | None) -> None:
    if best_epoch is not None:
        axis.axvline(
            best_epoch, color="0.35", linestyle="--", linewidth=1,
            label=f"best val loss (epoch {best_epoch})",
        )
    if stop_epoch is not None:
        axis.axvline(
            stop_epoch, color="tab:red", linestyle=":", linewidth=1,
            label=f"early-stop rule fired (epoch {stop_epoch})",
        )


def save_training_curves(
    history: Sequence[Mapping[str, float]],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Draw loss, validation Dice and (when logged) learning rate against epoch.

    Marks the best epoch and the epoch the early-stopping rule fired when the
    history carries those columns. Works on a partial history, so it is safe to
    call mid-training.
    """

    if not history:
        raise ValueError("Cannot draw training curves from an empty history")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    best_epoch = best_epoch_from_history(history)
    stop_epoch = early_stop_epoch_from_history(history)
    has_learning_rate = "learning_rate" in history[0]
    panels = 3 if has_learning_rate else 2
    figure, axes = plt.subplots(1, panels, figsize=(6 * panels, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    _mark_epochs(axes[0], best_epoch, stop_epoch)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE + soft Dice")
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, [row["val_disc_dice"] for row in history], label="disc Dice")
    axes[1].plot(epochs, [row["val_cup_dice"] for row in history], label="cup Dice")
    _mark_epochs(axes[1], best_epoch, stop_epoch)
    axes[1].set(title="Validation Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1].legend(fontsize=8)

    if has_learning_rate:
        axes[2].plot(epochs, [row["learning_rate"] for row in history], color="tab:green")
        axes[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR", yscale="log")

    if title:
        figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


@torch.inference_mode()
def save_prediction_gallery(
    model: torch.nn.Module,
    dataset: FundusSegmentationDataset,
    device: torch.device,
    output_path: str | Path,
    threshold: float = 0.5,
    count: int = 6,
    predict: Callable[[torch.Tensor, Mapping[str, Any]], torch.Tensor] | None = None,
) -> Path:
    """Overlay targets and predictions for a few dataset items.

    ``predict`` maps a ``(1, 3, H, W)`` batch and its batched metadata to logits;
    a conditioned arm passes one that also chooses the domain code, so the
    gallery shows the prediction the model actually makes at test time. When it
    is omitted the model is called on the images alone.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_indices = np.linspace(
        0, max(0, len(dataset) - 1), num=min(count, len(dataset)), dtype=int
    )
    figure, axes = plt.subplots(
        len(selected_indices), 4, figsize=(14, 3.5 * len(selected_indices)), squeeze=False
    )
    model.eval()
    for row, index in enumerate(selected_indices):
        image_tensor, target_tensor, metadata = dataset[int(index)]
        batch = image_tensor.unsqueeze(0).to(device)
        if predict is None:
            logits = model(batch)[0].cpu()
        else:
            batched_metadata = {key: [value] for key, value in metadata.items()}
            logits = predict(batch, batched_metadata)[0].cpu()
        prediction = (torch.sigmoid(logits) >= threshold).numpy()
        image = image_tensor.permute(1, 2, 0).numpy()
        target = target_tensor.numpy()
        target_overlay = _overlay(image, target)
        prediction_overlay = _overlay(image, prediction)

        false_positive = prediction & ~target.astype(bool)
        false_negative = target.astype(bool) & ~prediction
        error = image.copy()
        error[np.any(false_positive, axis=0)] = (1.0, 0.1, 0.1)
        error[np.any(false_negative, axis=0)] = (0.1, 1.0, 1.0)
        panels = (
            (image, f"{metadata['sample_id']} - image"),
            (target_overlay, "target contours"),
            (prediction_overlay, "prediction contours"),
            (error, "errors: red FP, cyan FN"),
        )
        for column, (panel, title) in enumerate(panels):
            axes[row, column].imshow(panel, vmin=0, vmax=1)
            axes[row, column].set_title(title, fontsize=9)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path



# --- Native-resolution prediction overlays -------------------------------------
#
# `save_prediction_gallery` above draws the letterboxed network input. The helpers
# below invert that letterbox so a prediction can be drawn on the original image at
# its own resolution. They are a separate path on purpose: training still calls the
# gallery, and the two must not drift into each other.


def letterbox_geometry(
    native_size: tuple[int, int], image_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(resized_wh, offset_xy)`` for ``data._resize_and_pad``.

    This mirrors that function's arithmetic exactly rather than reading the
    ``letterbox_scale`` the dataset reports, so rounding cannot differ between the
    forward and inverse transform. Both tuples are in PIL ``(x, y)`` order; numpy
    indexing needs them swapped.
    """

    width, height = native_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Native size must be positive, got {native_size}")
    scale = image_size / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    offset = ((image_size - resized[0]) // 2, (image_size - resized[1]) // 2)
    return resized, offset


def invert_letterbox(
    masks: np.ndarray, native_size: tuple[int, int], image_size: int
) -> np.ndarray:
    """Map letterboxed ``[C, S, S]`` binary masks back onto the native image grid.

    Nearest-neighbour throughout: bilinear on a binary mask yields fractional
    values and a contour that drifts by a pixel or two.
    """

    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[1] != image_size or masks.shape[2] != image_size:
        raise ValueError(
            f"Expected [C, {image_size}, {image_size}] masks, got {masks.shape}"
        )
    (resized_width, resized_height), (offset_x, offset_y) = letterbox_geometry(
        native_size, image_size
    )
    width, height = native_size
    native = np.zeros((masks.shape[0], height, width), dtype=bool)
    for channel in range(masks.shape[0]):
        cropped = masks[channel].astype(bool)[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ]
        native[channel] = (
            np.asarray(
                Image.fromarray(cropped.astype(np.uint8) * 255, mode="L").resize(
                    (width, height), Image.Resampling.NEAREST
                )
            )
            > 127
        )
    return native


def thicken(mask: np.ndarray, width: int) -> np.ndarray:
    """Dilate a 1px contour so it survives being drawn at figure scale.

    A native fundus image can be 2000px wide inside a 7in axis; an undilated
    contour disappears into resampling.
    """

    grown = np.asarray(mask, dtype=bool)
    for _ in range(max(0, width - 1)):
        expanded = grown.copy()
        expanded[1:, :] |= grown[:-1, :]
        expanded[:-1, :] |= grown[1:, :]
        expanded[:, 1:] |= grown[:, :-1]
        expanded[:, :-1] |= grown[:, 1:]
        grown = expanded
    return grown


def overlay_contours(
    image: np.ndarray,
    masks: np.ndarray,
    colors: Sequence[tuple[float, float, float]],
    contour_width: int = 1,
) -> np.ndarray:
    """Paint each mask's boundary onto ``image`` in the matching colour.

    The colourless generalisation of ``_overlay``: same ``_boundary``, but the
    caller picks the palette so ground truth and prediction can share one panel.
    """

    overlay = np.clip(np.asarray(image, dtype=np.float32).copy(), 0, 1)
    if len(colors) != len(masks):
        raise ValueError(f"Need one colour per mask, got {len(colors)}/{len(masks)}")
    for mask, color in zip(masks, colors):
        overlay[thicken(_boundary(mask), contour_width)] = color
    return overlay


def overlay_regions(
    image: np.ndarray,
    masks: np.ndarray,
    colors: Sequence[tuple[float, float, float]],
    alpha: float = 0.4,
    contour_width: int = 0,
) -> np.ndarray:
    """Alpha-blend filled mask regions onto ``image``, optionally edged.

    Contours show where a boundary sits; a filled region shows how much area a
    prediction claims, which is what makes a scale error read at a glance. Masks
    are painted in order, so a cup drawn after its disc sits on top of it.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    overlay = np.clip(np.asarray(image, dtype=np.float32).copy(), 0, 1)
    if len(colors) != len(masks):
        raise ValueError(f"Need one colour per mask, got {len(colors)}/{len(masks)}")
    for mask, color in zip(masks, colors):
        selected = np.asarray(mask, dtype=bool)
        overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * np.asarray(
            color, dtype=np.float32
        )
    if contour_width:
        for mask, color in zip(masks, colors):
            overlay[thicken(_boundary(mask), contour_width)] = color
    return overlay


def binary_mask_image(masks: np.ndarray, colors: Sequence) -> np.ndarray:
    """Render masks as flat colour on black -- the segmentation with no fundus.

    Stripping the image away is the honest way to compare two shapes: nothing of
    the retina is left to flatter or excuse the mask's outline.
    """

    if len(colors) != len(masks):
        raise ValueError(f"Need one colour per mask, got {len(colors)}/{len(masks)}")
    canvas = np.zeros((*np.asarray(masks[0]).shape, 3), dtype=np.float32)
    for mask, color in zip(masks, colors):
        canvas[np.asarray(mask, dtype=bool)] = color
    return canvas
