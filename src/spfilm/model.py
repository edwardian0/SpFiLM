from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """The plain U-Net building block used in the learning implementations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # InstanceNorm, not BatchNorm: SpFiLM will modulate normalized features, so
        # the baseline has to normalize the same way for the arms to be comparable.
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.convolutions(inputs)
        return skip, self.pool(skip)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.convolutions = DoubleConv(in_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(
                inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.convolutions(torch.cat((skip, inputs), dim=1))


class PlainUNet(nn.Module):
    """A standard 2D U-Net with no FiLM or SpFiLM conditioning."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16
        self.down1 = DownBlock(in_channels, c1)
        self.down2 = DownBlock(c1, c2)
        self.down3 = DownBlock(c2, c3)
        self.down4 = DownBlock(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)
        self.up1 = UpBlock(c5, c4)
        self.up2 = UpBlock(c4, c3)
        self.up3 = UpBlock(c3, c2)
        self.up4 = UpBlock(c2, c1)
        self.output = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1, pooled1 = self.down1(inputs)
        skip2, pooled2 = self.down2(pooled1)
        skip3, pooled3 = self.down3(pooled2)
        skip4, pooled4 = self.down4(pooled3)
        features = self.bottleneck(pooled4)
        features = self.up1(features, skip4)
        features = self.up2(features, skip3)
        features = self.up3(features, skip2)
        features = self.up4(features, skip1)
        return self.output(features)

