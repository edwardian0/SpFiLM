from __future__ import annotations

import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    """Equal-weight BCE and soft Dice over the disc and cup channels."""

    def __init__(
        self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-6
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = (0, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        channel_dice = (2 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1 - channel_dice.mean()
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * dice_loss

