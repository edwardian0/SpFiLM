from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .film.global_film import (
    DEFAULT_CLAMP,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DomainOneHot,
    GlobalFiLM,
)


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



class ConditionedUNet(nn.Module):
    """``PlainUNet`` with Global FiLM after each encoder block.

    The backbone is a real ``PlainUNet`` instance, not a copy of its layers, so
    the two arms share code and their parameter counts differ by exactly the
    FiLM generators. FiLM is applied to each ``DoubleConv`` output in the encoder
    (the tensor that feeds both the skip connection and the pooling) and to the
    bottleneck, following the SpFiLM draft's "after each convolutional block in
    the encoding part". ``film_levels`` counts conditioned levels from the
    shallowest; ``5`` conditions every encoder level including the bottleneck.
    """

    ENCODER_LEVELS = 5

    def __init__(
        self,
        num_domains: int,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
        film_levels: int = ENCODER_LEVELS,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        clamp: float = DEFAULT_CLAMP,
    ) -> None:
        super().__init__()
        if not 1 <= film_levels <= self.ENCODER_LEVELS:
            raise ValueError(
                f"film_levels must be in [1, {self.ENCODER_LEVELS}], got {film_levels}"
            )
        if not 1 <= num_domains <= embedding_dim:
            raise ValueError(
                f"num_domains must be in [1, {embedding_dim}], got {num_domains}"
            )
        self.num_domains = num_domains
        self.film_levels = film_levels
        self.backbone = PlainUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
        self.one_hot = DomainOneHot(embedding_dim)
        c1 = base_channels
        widths = (c1, c1 * 2, c1 * 4, c1 * 8, c1 * 16)
        self.films = nn.ModuleList(
            GlobalFiLM(width, embedding_dim, hidden_dim, clamp)
            for width in widths[:film_levels]
        )

    def _film(self, level: int, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if level < self.film_levels:
            return self.films[level](features, embedding)
        return features

    def forward(self, inputs: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor:
        if domain_index.numel() and int(domain_index.max()) >= self.num_domains:
            raise ValueError(
                f"domain_index {int(domain_index.max())} is outside the "
                f"{self.num_domains} trained codes"
            )
        net = self.backbone
        embedding = self.one_hot(domain_index)
        skip1 = self._film(0, net.down1.convolutions(inputs), embedding)
        skip2 = self._film(1, net.down2.convolutions(net.down1.pool(skip1)), embedding)
        skip3 = self._film(2, net.down3.convolutions(net.down2.pool(skip2)), embedding)
        skip4 = self._film(3, net.down4.convolutions(net.down3.pool(skip3)), embedding)
        features = self._film(4, net.bottleneck(net.down4.pool(skip4)), embedding)
        features = net.up1(features, skip4)
        features = net.up2(features, skip3)
        features = net.up3(features, skip2)
        features = net.up4(features, skip1)
        return net.output(features)


ARMS = ("plain", "global_film")


def build_model(
    arm: str,
    base_channels: int,
    num_domains: int | None = None,
    film_levels: int = ConditionedUNet.ENCODER_LEVELS,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    clamp: float = DEFAULT_CLAMP,
) -> nn.Module:
    """The one place that maps an experimental arm to a network."""

    if arm == "plain":
        return PlainUNet(base_channels=base_channels)
    if arm == "global_film":
        if num_domains is None:
            raise ValueError("global_film needs the number of source domains")
        return ConditionedUNet(
            num_domains=num_domains,
            base_channels=base_channels,
            film_levels=film_levels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            clamp=clamp,
        )
    if arm == "spatial_film":
        return SpatialFiLMUNet
    
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")

class SpatialFiLMUNet():
    pass
