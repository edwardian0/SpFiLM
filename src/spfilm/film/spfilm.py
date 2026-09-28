"""Rank-K Spatial FiLM layers for two-dimensional feature maps.

Spatial FiLM augments the global per-channel scale and shift with independent,
image-conditioned scale and shift basis maps. Domain-conditioned coefficients
mix those bases into dense fields before applying ``(1 + gamma) * F + beta``.
With rank zero there are no basis modules and the layer reduces numerically to
``GlobalFiLM``. This is the 2D adaptation of the reference 3D layer: it uses
``Conv2d``/``InstanceNorm2d`` and bilinear interpolation. Modulation runs in
float32 under autocast, and the assembled fields are clamped before use.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_RANK = 8
DEFAULT_BASIS_HIDDEN_CHANNELS = 16


class SpatialBasis(nn.Module):
    """Generate K smooth image-conditioned basis maps in [-1, 1].

    An instance supplies either φ (scale) or ψ (shift), independently of the domain code.
    """

    def __init__(
        self,
        rank: int,
        in_channels: int = 3,
        hidden_channels: int = DEFAULT_BASIS_HIDDEN_CHANNELS,
    ) -> None:
        super().__init__()

        self.rank = rank
        if self.rank < 1:
            raise ValueError("rank must be positive")

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(hidden_channels, rank, 3, padding=1, bias=False),
            nn.InstanceNorm2d(rank, affine=True),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return ``rank`` smooth basis maps at the input image resolution."""

        height, width = image.shape[-2:]
        basis_maps = self.layer(image)
        return F.interpolate(
            basis_maps,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )


class SpatialFiLM(nn.Module):
    """Rank-K spatially varying FiLM for one feature map."""

    def __init__(
        self,
        num_channels: int,
        rank: int = DEFAULT_RANK,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        clamp: float = 5.0,
        in_channels: int = 3,
        basis_hidden_channels: int = 16,
        fov_gating: bool = False,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if clamp <= 0:
            raise ValueError("clamp must be positive")

        self.num_channels = num_channels
        self.rank = rank
        self.clamp = float(clamp)
        self.fov_gating = fov_gating

        if rank == 0:
            self.basis_gamma = None
            self.basis_beta = None
        else:
            self.basis_gamma = SpatialBasis(
                rank=rank,
                in_channels=in_channels,
                hidden_channels=basis_hidden_channels,
            )
            self.basis_beta = SpatialBasis(
                rank=rank,
                in_channels=in_channels,
                hidden_channels=basis_hidden_channels,
            )

        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels * (1 + rank)),
        )

    def coefficients(
        self,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``gamma_bar, A, beta_bar, B`` in the documented MLP layout."""

        if embedding.dim() != 2:
            raise ValueError(
                f"embedding must be (N, embedding_dim), got "
                f"{tuple(embedding.shape)}"
            )

        with torch.autocast(device_type=embedding.device.type, enabled=False):
            parameters = self.generator(embedding.float())

        batch_size = parameters.shape[0]
        channels = self.num_channels
        rank = self.rank
        index = 0

        gamma_bar = parameters[:, index:index + channels]
        index += channels
        A = parameters[:, index:index + channels * rank]
        A = A.reshape(batch_size, channels, rank)
        index += channels * rank

        beta_bar = parameters[:, index:index + channels]
        index += channels

        B = parameters[:, index:index + channels * rank]
        B = B.reshape(batch_size, channels, rank)

        return gamma_bar, A, beta_bar, B

    def fields(
        self,
        image: torch.Tensor,
        embedding: torch.Tensor,
        size: tuple[int, int],
        fov_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble and clamp the dense gamma and beta modulation fields."""

        if image.dim() != 4:
            raise ValueError(
                f"image must be a 4-D tensor shaped (N, C, H, W), got "
                f"{tuple(image.shape)}"
            )

        if tuple(image.shape[-2:]) != tuple(size):
            raise ValueError(
                f"image spatial size {tuple(image.shape[-2:])} does not match "
                f"{tuple(size)}"
            )

        gamma_bar, A, beta_bar, B = self.coefficients(embedding)

        batch_size = embedding.shape[0]
        height, width = size

        gamma_global = gamma_bar.reshape(batch_size, self.num_channels, 1, 1)
        beta_global = beta_bar.reshape(batch_size, self.num_channels, 1, 1)

        if self.rank == 0:
            gamma = gamma_global.expand(-1, -1, height, width)
            beta = beta_global.expand(-1, -1, height, width)
        else:
            phi = self.basis_gamma(image.float())
            psi = self.basis_beta(image.float())

            phi_flat = phi.flatten(start_dim=2)
            psi_flat = psi.flatten(start_dim=2)

            gamma_spatial = torch.bmm(A, phi_flat).reshape(
                batch_size,
                self.num_channels,
                height,
                width,
            )
            beta_spatial = torch.bmm(B, psi_flat).reshape(
                batch_size,
                self.num_channels,
                height,
                width,
            )

            gamma = gamma_global + gamma_spatial
            beta = beta_global + beta_spatial

        gamma = gamma.clamp(-self.clamp, self.clamp)
        beta = beta.clamp(-self.clamp, self.clamp)

        # Gating is deliberately ignored unless the layer was configured to use it.
        if self.fov_gating and fov_mask is not None:
            if fov_mask.dim() != 4:
                raise ValueError(
                    f"fov_mask must be (N, 1, H, W), got {tuple(fov_mask.shape)}"
                )
            if fov_mask.shape[1] != 1:
                raise ValueError(
                    f"fov_mask must have exactly one channel, got "
                    f"{fov_mask.shape[1]}"
                )
            if fov_mask.shape[0] != gamma.shape[0]:
                raise ValueError(
                    f"batch mismatch: {gamma.shape[0]} fields but "
                    f"{fov_mask.shape[0]} masks"
                )

            resized_mask = F.interpolate(
                fov_mask.float(),
                size=size,
                mode="nearest",
            )

            gamma = gamma * resized_mask
            beta = beta * resized_mask

        return gamma, beta

    def forward(
        self,
        features: torch.Tensor,
        image: torch.Tensor,
        embedding: torch.Tensor,
        fov_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply spatial FiLM in float32 and preserve the feature dtype."""

        if features.dim() != 4 or features.shape[1] != self.num_channels:
            raise ValueError(
                f"features must be (N, {self.num_channels}, H, W), got "
                f"{tuple(features.shape)}"
            )
        if image.dim() != 4:
            raise ValueError(
                f"image must be a 4-D tensor shaped (N, C, H, W), got "
                f"{tuple(image.shape)}"
            )
        if embedding.dim() != 2:
            raise ValueError(
                f"embedding must be (N, embedding_dim), got "
                f"{tuple(embedding.shape)}"
            )
        if not (features.shape[0] == image.shape[0] == embedding.shape[0]):
            raise ValueError(
                f"batch mismatch: {features.shape[0]} feature maps, "
                f"{image.shape[0]} images, and {embedding.shape[0]} embeddings"
            )
        if tuple(features.shape[-2:]) != tuple(image.shape[-2:]):
            raise ValueError(
                f"image spatial size {tuple(image.shape[-2:])} does not match "
                f"feature spatial size {tuple(features.shape[-2:])}"
            )

        with torch.autocast(device_type=features.device.type, enabled=False):
            gamma, beta = self.fields(
                image=image,
                embedding=embedding,
                size=tuple(features.shape[-2:]),
                fov_mask=fov_mask,
            )
            features_f = features.float()
            modulated = (1.0 + gamma) * features_f + beta

        return modulated.to(features.dtype)
