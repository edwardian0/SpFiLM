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
            raise ValueError("Rank is less than 1.")

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

        # FOV Gating
        if self.fov_gating and fov_mask is not None:
            if tuple(fov_mask.shape[-2:]) != tuple(fov_mask.size()):
                raise ValueError(
                    f"FOV mask must be (N, embedding_dim), got "
                    f"{tuple(embedding.shape)}"
                )

            if self.num_channels != 1:
                raise ValueError()

            if 
        return gamma, beta
