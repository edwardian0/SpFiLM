"""How a conditioned model gets its domain code at train, validation, and test time.

Training and validation images come from the source domains, so their code is
the true one (``OracleCondition``). Under leave-one-domain-out the test images
come from a domain the model has never seen, so there is no valid code for
them. The policy agreed with the supervisor (2026-09-12) is: *supply the code of
the source domain whose distribution the image is closest to*. That is the
``NearestDomainSelector``: a label-free appearance descriptor of the input is
compared with per-source-domain reference statistics fitted on the fold's
training images, and the nearest domain's code is used. The selector is part of
the forward pass -- one decision per image, nothing pooled over the test set --
so a held-out result stays purely inductive.

The descriptor is the per-channel mean and standard deviation of the RGB values
inside the retinal field of view, computed on the exact tensor the network
receives (after letterboxing, in [0, 1]). It is deliberately the same family of
statistic as the earlier domain-shift diagnosis, which found the cross-domain
intensity gap to be almost entirely a per-channel gain and offset, and it masks
the black surround with the same luminance threshold ``global_histograms`` uses
so that letterbox padding cannot dominate it.

Both arms of the comparison (Global FiLM now, SpFiLM later) must use this same
selector with the same reference statistics, so that they differ only in K.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..global_histograms import FOV_LUMINANCE_THRESHOLD, LUMA_WEIGHTS

DESCRIPTOR_NAMES = (
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
)
DESCRIPTOR_POLICY = (
    "per-channel RGB mean and std over field-of-view pixels "
    f"(Rec. 601 luminance > {FOV_LUMINANCE_THRESHOLD}) of the letterboxed [0, 1] "
    "input tensor"
)
SELECTION_POLICY = (
    "per image: standardise each descriptor dimension by the pooled "
    "within-domain training spread (diagonal LDA), take the Euclidean distance "
    "to each source domain's training centroid, and use the nearest domain's code"
)
SCHEMA_VERSION = 1


class ConditioningError(ValueError):
    """A domain code was requested that the model was never trained with."""


@dataclass(frozen=True)
class DomainVocabulary:
    """The source domains a conditioned model was trained on, in code order.

    The vocabulary is fold-local: a model trained without RIM-ONE-DL has no code
    for it. That is deliberate. The FiLM generator's weights for an untrained
    code are random, so the only safe codes are the ones that saw gradients, and
    keeping the vocabulary to those makes selecting anything else impossible.
    """

    domains: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError("DomainVocabulary needs at least one domain")
        if len(set(self.domains)) != len(self.domains):
            raise ValueError(f"DomainVocabulary has duplicates: {self.domains}")
        if tuple(sorted(self.domains)) != self.domains:
            raise ValueError(
                "DomainVocabulary must be sorted so codes are reproducible: "
                f"{self.domains}"
            )

    @classmethod
    def from_domains(cls, domains: Iterable[str]) -> "DomainVocabulary":
        return cls(tuple(sorted(set(str(domain) for domain in domains))))

    def __len__(self) -> int:
        return len(self.domains)

    def index_of(self, domain: str) -> int:
        try:
            return self.domains.index(domain)
        except ValueError:
            raise ConditioningError(
                f"Domain {domain!r} has no code; the model was trained on "
                f"{list(self.domains)}"
            ) from None

    def indices(self, domains: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(
            [self.index_of(str(domain)) for domain in domains],
            dtype=torch.long,
            device=device,
        )

    def to_json(self) -> list[str]:
        return list(self.domains)

    @classmethod
    def from_json(cls, payload: object) -> "DomainVocabulary":
        if not isinstance(payload, list) or not all(
            isinstance(item, str) for item in payload
        ):
            raise ValueError("DomainVocabulary payload must be a list of strings")
        return cls(tuple(payload))


def fov_descriptor(images: torch.Tensor) -> torch.Tensor:
    """Per-image ``(N, 6)`` field-of-view RGB mean and std of a ``(N, 3, H, W)`` batch.

    An image whose mask is empty (no pixel above the threshold, which no real
    fundus image produces) falls back to all its pixels rather than yielding NaN.
    """

    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be (N, 3, H, W), got {tuple(images.shape)}")
    pixels = images.float()
    weights = torch.tensor(LUMA_WEIGHTS, dtype=pixels.dtype, device=pixels.device)
    luminance = torch.einsum("nchw,c->nhw", pixels, weights)
    mask = (luminance > FOV_LUMINANCE_THRESHOLD).to(pixels.dtype)
    counts = mask.sum(dim=(1, 2))
    empty = counts <= 0
    if bool(empty.any()):
        mask = torch.where(empty[:, None, None], torch.ones_like(mask), mask)
        counts = mask.sum(dim=(1, 2))
    mask = mask[:, None, :, :]
    means = (pixels * mask).sum(dim=(2, 3)) / counts[:, None]
    centred = (pixels - means[:, :, None, None]) * mask
    variances = (centred * centred).sum(dim=(2, 3)) / counts[:, None]
    stds = variances.clamp_min(0.0).sqrt()
    return torch.cat([means, stds], dim=1)


@dataclass(frozen=True)
class NearestDomainSelector:
    """Nearest-source-domain rule fitted on one fold's training images."""

    vocabulary: DomainVocabulary
    centroids: torch.Tensor  # (K, 6) mean descriptor per domain
    spreads: torch.Tensor  # (K, 6) descriptor std within each domain
    shift: torch.Tensor  # (6,) pooled mean; cancels in the argmin, kept for reporting
    scale: torch.Tensor  # (6,) pooled within-domain std used for standardising
    fitted_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = (len(self.vocabulary), len(DESCRIPTOR_NAMES))
        for name in ("centroids", "spreads"):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")
        for name in ("shift", "scale"):
            value = getattr(self, name)
            if tuple(value.shape) != (len(DESCRIPTOR_NAMES),):
                raise ValueError(
                    f"{name} must be ({len(DESCRIPTOR_NAMES)},), got {tuple(value.shape)}"
                )
        if len(self.fitted_counts) != len(self.vocabulary):
            raise ValueError("fitted_counts must have one entry per vocabulary domain")
        if any(count <= 0 for count in self.fitted_counts):
            raise ValueError(
                "Every vocabulary domain must contribute training images to the "
                f"selector, got counts {self.fitted_counts}"
            )
        if bool((self.scale <= 0).any()) or not bool(torch.isfinite(self.scale).all()):
            raise ValueError("scale must be finite and positive in every dimension")

    @classmethod
    def fit(
        cls,
        descriptors: torch.Tensor,
        domains: Sequence[str],
        vocabulary: DomainVocabulary,
    ) -> "NearestDomainSelector":
        """Fit from already-computed descriptors ``(M, 6)`` and their true domains."""

        if descriptors.dim() != 2 or descriptors.shape[1] != len(DESCRIPTOR_NAMES):
            raise ValueError(
                f"descriptors must be (M, {len(DESCRIPTOR_NAMES)}), got "
                f"{tuple(descriptors.shape)}"
            )
        if descriptors.shape[0] != len(domains):
            raise ValueError(
                f"{descriptors.shape[0]} descriptors but {len(domains)} domains"
            )
        descriptors = descriptors.detach().float().cpu()
        labels = vocabulary.indices(domains)
        centroids = []
        spreads = []
        counts = []
        for index in range(len(vocabulary)):
            members = descriptors[labels == index]
            counts.append(int(members.shape[0]))
            if members.shape[0] == 0:
                # Rejected by __post_init__ through fitted_counts; keep the
                # arithmetic finite until then.
                centroids.append(torch.full((len(DESCRIPTOR_NAMES),), float("nan")))
                spreads.append(torch.zeros(len(DESCRIPTOR_NAMES)))
                continue
            centroids.append(members.mean(dim=0))
            spreads.append(
                members.std(dim=0, unbiased=False)
                if members.shape[0] > 1
                else torch.zeros(len(DESCRIPTOR_NAMES))
            )
        shift = descriptors.mean(dim=0)
        # Standardise by the pooled *within-domain* spread, not the total spread.
        # The total spread of a dimension that separates the domains is mostly
        # between-domain variance, so dividing by it would shrink exactly the
        # dimensions that carry the signal while a noisy, uninformative dimension
        # with a small total spread would be blown up to unit variance. Weighting
        # each dimension by its within-domain noise is the diagonal-LDA metric.
        count_tensor = torch.tensor(counts, dtype=torch.float32)
        within_variance = (
            torch.stack(spreads) ** 2 * count_tensor[:, None]
        ).sum(dim=0) / count_tensor.sum()
        scale = within_variance.clamp_min(0.0).sqrt()
        # A dimension with no within-domain spread (every training image agrees)
        # cannot be standardised; fall back to the total spread, then to unit
        # scale, rather than dividing by zero.
        total = descriptors.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, total)
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        return cls(
            vocabulary=vocabulary,
            centroids=torch.stack(centroids),
            spreads=torch.stack(spreads),
            shift=shift,
            scale=scale,
            fitted_counts=tuple(counts),
        )

    @classmethod
    def fit_from_loader(
        cls, loader: Iterable[tuple[torch.Tensor, Any, Mapping[str, Any]]], vocabulary: DomainVocabulary
    ) -> "NearestDomainSelector":
        """Fit from a loader over un-augmented training images."""

        descriptors: list[torch.Tensor] = []
        domains: list[str] = []
        for images, _targets, metadata in loader:
            descriptors.append(fov_descriptor(images).cpu())
            domains.extend(str(value) for value in metadata["domain"])
        if not descriptors:
            raise ValueError("Selector loader yielded no images")
        return cls.fit(torch.cat(descriptors), domains, vocabulary)

    def distances(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Z-scored Euclidean distance ``(N, K)`` from each descriptor to each centroid."""

        descriptors = descriptors.float()
        shift = self.shift.to(descriptors.device)
        scale = self.scale.to(descriptors.device)
        centroids = self.centroids.to(descriptors.device)
        standardised = (descriptors - shift) / scale
        reference = (centroids - shift) / scale
        return torch.cdist(standardised, reference)

    def select(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(indices (N,), distances (N, K), descriptors (N, 6))``."""

        descriptors = fov_descriptor(images)
        distances = self.distances(descriptors)
        return distances.argmin(dim=1), distances, descriptors

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "descriptor": list(DESCRIPTOR_NAMES),
            "descriptor_policy": DESCRIPTOR_POLICY,
            "selection_policy": SELECTION_POLICY,
            "fov_luminance_threshold": FOV_LUMINANCE_THRESHOLD,
            "luma_weights": list(LUMA_WEIGHTS),
            "vocabulary": self.vocabulary.to_json(),
            "fitted_counts": list(self.fitted_counts),
            "centroids": self.centroids.tolist(),
            "spreads": self.spreads.tolist(),
            "shift": self.shift.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "NearestDomainSelector":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported selector schema {payload.get('schema_version')!r}"
            )
        if list(payload.get("descriptor", [])) != list(DESCRIPTOR_NAMES):
            raise ValueError("Selector payload was fitted with a different descriptor")
        return cls(
            vocabulary=DomainVocabulary.from_json(payload["vocabulary"]),
            centroids=torch.tensor(payload["centroids"], dtype=torch.float32),
            spreads=torch.tensor(payload["spreads"], dtype=torch.float32),
            shift=torch.tensor(payload["shift"], dtype=torch.float32),
            scale=torch.tensor(payload["scale"], dtype=torch.float32),
            fitted_counts=tuple(int(value) for value in payload["fitted_counts"]),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NearestDomainSelector":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ConditionResult:
    """The codes used for one batch, plus what the selector saw when it chose them."""

    indices: torch.Tensor  # (N,) long
    source: str  # "oracle", "nearest_domain", or "fixed:<domain>"
    distances: torch.Tensor | None = None  # (N, K) when a selector ran
    descriptors: torch.Tensor | None = None  # (N, 6) when a selector ran


class OracleCondition:
    """The true source-domain code; refuses any domain outside the vocabulary."""

    source = "oracle"

    def __init__(self, vocabulary: DomainVocabulary) -> None:
        self.vocabulary = vocabulary

    def __call__(
        self, images: torch.Tensor, metadata: Mapping[str, Any]
    ) -> ConditionResult:
        domains = [str(value) for value in metadata["domain"]]
        if len(domains) != images.shape[0]:
            raise ConditioningError(
                f"{len(domains)} domains for a batch of {images.shape[0]} images"
            )
        return ConditionResult(
            indices=self.vocabulary.indices(domains, device=images.device),
            source=self.source,
        )


class NearestCondition:
    """The nearest source domain's code, chosen per image by the selector."""

    source = "nearest_domain"

    def __init__(self, selector: NearestDomainSelector) -> None:
        self.selector = selector
        self.vocabulary = selector.vocabulary

    def __call__(
        self, images: torch.Tensor, metadata: Mapping[str, Any]
    ) -> ConditionResult:
        del metadata  # the rule is a function of the image alone
        indices, distances, descriptors = self.selector.select(images)
        return ConditionResult(
            indices=indices.to(images.device),
            source=self.source,
            distances=distances,
            descriptors=descriptors,
        )


class FixedCondition:
    """One source domain's code for every image; used for the fixed-code sweep."""

    def __init__(self, vocabulary: DomainVocabulary, domain: str) -> None:
        self.vocabulary = vocabulary
        self.domain = domain
        self.index = vocabulary.index_of(domain)
        self.source = f"fixed:{domain}"

    def __call__(
        self, images: torch.Tensor, metadata: Mapping[str, Any]
    ) -> ConditionResult:
        del metadata
        return ConditionResult(
            indices=torch.full(
                (images.shape[0],), self.index, dtype=torch.long, device=images.device
            ),
            source=self.source,
        )
