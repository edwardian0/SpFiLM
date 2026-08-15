from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


CHANNEL_NAMES = ("disc", "cup")


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


@dataclass
class OverlapAccumulator:
    threshold: float = 0.5
    dice_batches: list[torch.Tensor] = field(default_factory=list)
    iou_batches: list[torch.Tensor] = field(default_factory=list)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = torch.sigmoid(logits) >= self.threshold
        dice, iou = per_image_overlap(predictions, targets >= 0.5)
        self.dice_batches.append(dice.cpu())
        self.iou_batches.append(iou.cpu())

    def compute(self) -> dict[str, dict[str, float | int]]:
        if not self.dice_batches:
            raise RuntimeError("Cannot compute metrics before any samples were added")
        dice = torch.cat(self.dice_batches).numpy()
        iou = torch.cat(self.iou_batches).numpy()
        result: dict[str, dict[str, float | int]] = {}
        for channel, name in enumerate(CHANNEL_NAMES):
            result[name] = {
                "dice_mean": float(np.mean(dice[:, channel])),
                "dice_std": float(np.std(dice[:, channel], ddof=0)),
                "dice_median": float(np.median(dice[:, channel])),
                "iou_mean": float(np.mean(iou[:, channel])),
                "iou_std": float(np.std(iou[:, channel], ddof=0)),
                "sample_count": int(dice.shape[0]),
            }
        return result

