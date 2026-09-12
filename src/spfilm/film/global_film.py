"""Global (channel-wise) FiLM, the non-spatial conditioning baseline.

This is the layer the project brief calls "Global FiLM" and the SpFiLM draft calls
channel-wise FiLM (its Sec. 2.1, eq. 1):

    FiLM(F_c) = (1 + gamma_c(s)) * F_c + beta_c(s)

One scale and one shift per channel, broadcast to every pixel; that broadcast is
what makes it "global". ``s`` is a frozen one-hot embedding of the domain with no
learned parameters, and a three-layer MLP (256, 256, 2C) maps it to gamma and
beta, exactly as in the draft and in Perez et al.'s reference code (whose
``gamma_baseline=1`` is the ``1 +`` here). gamma and beta are clamped to
[-clamp, clamp] as the draft does for numerical stability.

SpFiLM with K=0 must reproduce this layer numerically; keep them aligned.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_CLAMP = 5.0


class DomainOneHot(nn.Module):
    """Frozen one-hot embedding: domain index ``i`` maps to the ``i``-th basis vector.

    With at most a handful of domains only the first few of the ``embedding_dim``
    basis vectors are ever used, which is what the draft does with its 64-d code.
    The identity matrix is a buffer, so the embedding contributes no parameters.
    """

    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.register_buffer("basis", torch.eye(embedding_dim), persistent=False)

    def forward(self, domain_index: torch.Tensor) -> torch.Tensor:
        if domain_index.dim() != 1:
            raise ValueError(
                f"domain_index must be a 1-D tensor of indices, got shape "
                f"{tuple(domain_index.shape)}"
            )
        if domain_index.numel() and (
            int(domain_index.min()) < 0
            or int(domain_index.max()) >= self.embedding_dim
        ):
            raise ValueError(
                f"domain_index must lie in [0, {self.embedding_dim}), got "
                f"[{int(domain_index.min())}, {int(domain_index.max())}]"
            )
        return self.basis[domain_index.long()]


class GlobalFiLM(nn.Module):
    """Channel-wise FiLM for one feature map; owns its own gamma/beta MLP."""

    def __init__(
        self,
        num_channels: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        clamp: float = DEFAULT_CLAMP,
    ) -> None:
        super().__init__()
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if clamp <= 0:
            raise ValueError("clamp must be positive")
        self.num_channels = num_channels
        self.clamp = float(clamp)
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )

    def gamma_beta(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the clamped per-channel (gamma, beta), each ``(N, C)``."""

        if embedding.dim() != 2:
            raise ValueError(
                f"embedding must be (N, embedding_dim), got {tuple(embedding.shape)}"
            )
        parameters = self.generator(embedding)
        gamma, beta = torch.split(parameters, self.num_channels, dim=1)
        return (
            gamma.clamp(-self.clamp, self.clamp),
            beta.clamp(-self.clamp, self.clamp),
        )

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4 or features.shape[1] != self.num_channels:
            raise ValueError(
                f"features must be (N, {self.num_channels}, H, W), got "
                f"{tuple(features.shape)}"
            )
        if features.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"batch mismatch: {features.shape[0]} feature maps but "
                f"{embedding.shape[0]} embeddings"
            )
        gamma, beta = self.gamma_beta(embedding)
        gamma = gamma.to(features.dtype)[:, :, None, None]
        beta = beta.to(features.dtype)[:, :, None, None]
        return (1.0 + gamma) * features + beta
