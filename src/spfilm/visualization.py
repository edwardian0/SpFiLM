from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def save_training_curves(history: list[dict[str, float]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="BCE + soft Dice")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["val_disc_dice"] for row in history],
        label="disc Dice",
    )
    axes[1].plot(
        epochs,
        [row["val_cup_dice"] for row in history],
        label="cup Dice",
    )
    axes[1].set(title="Validation Dice", xlabel="Epoch", ylabel="Dice", ylim=(0, 1))
    axes[1].legend()
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
) -> Path:
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
        logits = model(image_tensor.unsqueeze(0).to(device))[0].cpu()
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

