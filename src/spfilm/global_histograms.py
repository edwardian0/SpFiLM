"""Pixel-intensity histograms accumulated across a whole acquisition domain.

This is the one implementation of intensity accumulation in the project.
``analyze_domain_shift.py`` builds its four-domain comparison on top of it, so
the conventions below are fixed in a single place and every figure in the
write-up agrees about what it is measuring.

Two populations of pixels are accumulated for every image, because they answer
different questions:

``all``
    Every pixel of the source image. REFUGE and Drishti-GS store a circular
    retinal field of view inside a rectangular frame, so a large share of their
    pixels are the black surround. RIM-ONE-DL ships square optic-nerve-head
    crops with almost no surround. The ``all`` histograms therefore show the
    *framing* difference, which is real and is what the network's input tensor
    contains.

``fov``
    Only pixels inside the retinal field of view. This removes the framing
    difference and isolates the *photometric* difference: illumination, camera
    response, and pigmentation.

A domain gap in ``fov`` means the cameras genuinely disagree about colour. A gap
that exists only in ``all`` means the images are cropped differently. The two
have different remedies, so they are never pooled here.

Histograms are stored as **densities**, not raw counts: per-image counts are
summed, then divided by the domain's total pixels and the bin width, so every
curve integrates to one and domains of very different size stay comparable.
:meth:`DomainHistograms.counts` recovers the underlying counts when a raw
"total pixels" axis is wanted.

PIL is used rather than OpenCV deliberately: Pillow is a declared dependency of
this package, OpenCV is not, and an unguarded ``import cv2`` here would make the
whole ``spfilm`` package fail to import on CREATE.
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch off this path
    from .data import FundusRecord


BIN_COUNT = 256
CHANNELS = ("gray", "red", "green", "blue")
POPULATIONS = ("all", "fov")
DEFAULT_WORKING_SIZE = 256
# Below this luminance a pixel is the black surround outside the retinal disc
# rather than tissue. Fundus surrounds sit within a few counts of zero, so the
# threshold is far from any real retinal intensity and the choice is not
# delicate; callers record it alongside their figures so they can be reproduced.
FOV_LUMINANCE_THRESHOLD = 0.10
# Rec. 601 luma. The green channel dominates it, which suits fundus photography:
# green carries most of the vessel and rim contrast.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
DOMAIN_COLOURS = {
    "refuge_zeiss": "#1f77b4",
    "refuge_canon_val": "#17becf",
    "drishti_gs": "#d62728",
    "rim_one_dl": "#2ca02c",
}


class HistogramError(ValueError):
    """An image could not be read, or a domain accumulated no pixels."""


def bin_edges() -> np.ndarray:
    """Bin boundaries over the normalised intensity range [0, 1]."""

    return np.linspace(0.0, 1.0, BIN_COUNT + 1)


def bin_centres() -> np.ndarray:
    edges = bin_edges()
    return (edges[:-1] + edges[1:]) / 2.0


@dataclass(frozen=True)
class DomainHistograms:
    """Normalised intensity densities for one domain, per channel and population."""

    domain: str
    image_count: int
    pixel_counts: dict[str, int]
    densities: dict[str, dict[str, np.ndarray]]

    def density(self, population: str, channel: str) -> np.ndarray:
        return self.densities[population][channel]

    def counts(self, population: str, channel: str) -> np.ndarray:
        """Recover the accumulated pixel counts behind a density curve.

        Every pixel of the population contributes to each channel, so the counts
        for a channel sum to ``pixel_counts[population]``. Densities are the
        stored form because they are what makes domains comparable; counts are
        derived here for figures that want an absolute "total pixels" axis.
        """

        width = 1.0 / BIN_COUNT
        return self.density(population, channel) * width * self.pixel_counts[population]

    def proportions(self, population: str, channel: str) -> np.ndarray:
        """Fraction of the population's pixels falling in each bin; sums to 1.

        This is the form to plot. ``density`` integrates to one *over the [0, 1]
        intensity axis*, so on a 0-255 axis its area reads as 255 and its height
        depends on the bin count; ``counts`` is not normalised at all and scales
        with how many images the domain happens to have. A proportion is
        directly comparable between domains and needs no axis caveat.
        """

        width = 1.0 / BIN_COUNT
        return self.density(population, channel) * width


def plot_domain_overlay(
    histograms: Sequence[DomainHistograms],
    output_path: str | Path,
    population: str = "fov",
    channel: str = "gray",
    annotations: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    """Draw every domain's intensity histogram on one axes, to show the shift.

    The per-domain figures answer "what does this domain look like"; this one
    answers "how far apart are they", which needs the curves on a shared axis
    where the horizontal offsets between peaks are read directly.

    Curves are proportions, so the four are comparable despite the domains
    differing in image count and frame size — an un-normalised overlay would
    order the curves by dataset size and show nothing about the shift.

    ``annotations`` adds a per-domain suffix to the legend (the caller passes
    distance-to-rest, which turns "these differ" into "by this much"). Distances
    are not computed here: this module owns the histograms, not the metrics.
    """

    if population not in POPULATIONS:
        raise HistogramError(f"unknown population {population!r}")
    if channel not in CHANNELS:
        raise HistogramError(f"unknown channel {channel!r}")
    if not histograms:
        raise HistogramError("no histograms to overlay")

    annotations = annotations or {}
    intensity = bin_centres() * 255.0
    ordered = sorted(histograms, key=lambda item: item.domain)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(11, 6))
    for histogram in ordered:
        shares = histogram.proportions(population, channel)
        colour = DOMAIN_COLOURS.get(histogram.domain)
        label = f"{histogram.domain}  (n={histogram.image_count}"
        suffix = annotations.get(histogram.domain)
        label += f", {suffix})" if suffix else ")"
        axes.plot(intensity, shares, color=colour, linewidth=1.9, label=label)
        # A light wash under each curve keeps the four readable where they
        # cross, without the solid fill that would hide whichever is drawn last.
        axes.fill_between(intensity, shares, color=colour, alpha=0.07, linewidth=0)

    if population == "all":
        # The black surround puts a spike two orders of magnitude above the
        # tissue distribution at zero; on a linear axis it flattens everything.
        axes.set_yscale("log")

    axes.set_title(
        title
        or (
            f"Pixel-intensity shift across acquisition domains "
            f"({channel}, {population} pixels)"
        ),
        fontsize=13,
    )
    axes.set_xlabel("Pixel intensity (0-255)")
    axes.set_ylabel("Fraction of pixels")
    axes.set_xlim(0.0, 255.0)
    axes.margins(y=0.05)
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.legend(frameon=False, loc="upper right")
    if population == "fov":
        figure.text(
            0.5,
            -0.01,
            f"Field-of-view pixels only (luminance > {FOV_LUMINANCE_THRESHOLD}); "
            "the cliff near 25 is that threshold, not the data.",
            ha="center",
            fontsize=9,
            color="0.35",
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def sample_images(
    domain: str,
    items: Sequence[object],
    count: int,
    seed: int = 0,
) -> list[object]:
    """Take ``count`` items from one domain at random, without replacement.

    Domains differ in size by nearly five to one (Drishti-GS has 101 images
    against RIM-ONE-DL's 485). Densities already remove that from the *shape* of
    a histogram, but not from how much sampling noise each curve carries, so an
    equal-sized draw per domain makes the curves fairer to compare by eye.

    Taking the head of the list instead would not be a random sample: the
    records arrive in filename order, which tracks release prefix and, in some
    releases, diagnosis class. The draw is seeded per domain, so a domain's
    sample does not shift when another domain's record count changes, and any
    figure can be reproduced from the seed recorded beside it. Domains with
    fewer than ``count`` images are returned whole.
    """

    if count <= 0:
        raise HistogramError(f"sample count must be positive, got {count}")
    if len(items) <= count:
        return list(items)
    rng = random.Random(f"{seed}:{domain}")
    # Sample indices and restore the original order, so the selection is a
    # subset of the sequence rather than a reshuffling of it.
    chosen = sorted(rng.sample(range(len(items)), count))
    return [items[index] for index in chosen]


def load_pixels(
    image_path: str | Path, working_size: int = DEFAULT_WORKING_SIZE
) -> np.ndarray:
    """Decode one image to a small RGB float array in [0, 1].

    ``draft`` lets the JPEG decoder downscale while decoding, which is what makes
    it affordable to sweep every REFUGE image at full dataset size. The intensity
    *distribution* is what is being estimated, and it is insensitive to this
    resampling; the exact pixel grid is not needed.
    """

    try:
        with Image.open(image_path) as image:
            image.draft("RGB", (working_size, working_size))
            image = image.convert("RGB")
            image.thumbnail((working_size, working_size), Image.BILINEAR)
            return np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as error:
        # Skipping unreadable files in silence would produce a plausible-looking
        # histogram over whatever happened to decode, which is worse than a stop.
        raise HistogramError(f"could not read image {image_path}: {error}") from error


def image_histograms(
    image_path: str | Path, working_size: int = DEFAULT_WORKING_SIZE
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-image counts keyed ``"<population>/<channel>"``, plus pixels per population."""

    pixels = load_pixels(image_path, working_size)
    luminance = pixels @ np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    fov = luminance > FOV_LUMINANCE_THRESHOLD
    edges = bin_edges()

    bands = {
        "gray": luminance,
        "red": pixels[..., 0],
        "green": pixels[..., 1],
        "blue": pixels[..., 2],
    }
    counts: dict[str, np.ndarray] = {}
    for population in POPULATIONS:
        for channel, band in bands.items():
            values = band if population == "all" else band[fov]
            counts[f"{population}/{channel}"] = np.histogram(values, bins=edges)[
                0
            ].astype(np.float64)
    pixel_counts = {"all": int(luminance.size), "fov": int(fov.sum())}
    return counts, pixel_counts


def _worker(payload: tuple[str, int]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    image_path, working_size = payload
    return image_histograms(Path(image_path), working_size)


def accumulate_paths(
    domain: str,
    image_paths: Sequence[str | Path],
    working_size: int = DEFAULT_WORKING_SIZE,
    workers: int = 1,
) -> DomainHistograms:
    """Sum per-image histograms over one domain, then normalise once at the end.

    Summing counts and normalising afterwards weights each domain by its pixels,
    not by its images, so a domain is not skewed by having a handful of unusually
    large frames.
    """

    if not image_paths:
        raise HistogramError(f"{domain} has no images to accumulate")

    totals: dict[str, np.ndarray] = {
        f"{population}/{channel}": np.zeros(BIN_COUNT, dtype=np.float64)
        for population in POPULATIONS
        for channel in CHANNELS
    }
    pixel_totals = {population: 0 for population in POPULATIONS}
    payloads = [(str(path), working_size) for path in image_paths]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results: Iterable[tuple[dict[str, np.ndarray], dict[str, int]]] = pool.map(
                _worker, payloads, chunksize=8
            )
            collected = list(results)
    else:
        collected = [_worker(payload) for payload in payloads]

    for counts, pixel_counts in collected:
        for key, value in counts.items():
            totals[key] += value
        for population, value in pixel_counts.items():
            pixel_totals[population] += value

    densities: dict[str, dict[str, np.ndarray]] = {
        population: {} for population in POPULATIONS
    }
    width = 1.0 / BIN_COUNT
    for population in POPULATIONS:
        for channel in CHANNELS:
            counts = totals[f"{population}/{channel}"]
            total = counts.sum()
            if total <= 0:
                raise HistogramError(
                    f"{domain} {population}/{channel} accumulated no pixels"
                )
            # A density, not a count: the curves are comparable across domains
            # of very different size, and each integrates to one.
            densities[population][channel] = counts / total / width

    return DomainHistograms(
        domain=domain,
        image_count=len(payloads),
        pixel_counts=pixel_totals,
        densities=densities,
    )


def accumulate_domain(
    domain: str,
    records: Sequence["FundusRecord"],
    working_size: int = DEFAULT_WORKING_SIZE,
    workers: int = 1,
) -> DomainHistograms:
    """:func:`accumulate_paths` over the images of a sequence of records."""

    return accumulate_paths(
        domain,
        [record.image_path for record in records],
        working_size=working_size,
        workers=workers,
    )


def plot_global_histogram(
    histogram: DomainHistograms,
    output_path: str | Path,
    population: str = "fov",
    channel: str = "gray",
    normalise: bool = True,
) -> Path:
    """Draw one domain's intensity histogram and save it to ``output_path``.

    This is the single-dataset figure, as opposed to the four-domain comparison
    in ``analyze_domain_shift.py``. ``population`` defaults to ``"fov"`` because
    the black surround in ``"all"`` puts a spike at zero that is one to two
    orders of magnitude taller than the tissue distribution and flattens
    everything else on a linear axis.

    The curve is normalised by default: each point is the fraction of that
    domain's pixels in the bin, so domains with different image counts and
    different frame sizes can be laid side by side. Pass ``normalise=False``
    for the raw pixel counts.
    """

    if population not in POPULATIONS:
        raise HistogramError(f"unknown population {population!r}")
    if channel not in CHANNELS:
        raise HistogramError(f"unknown channel {channel!r}")

    values = (
        histogram.proportions(population, channel)
        if normalise
        else histogram.counts(population, channel)
    )
    intensity = bin_centres() * 255.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.fill_between(intensity, values, color="black", alpha=0.15, linewidth=0)
    axes.plot(intensity, values, color="black", linewidth=1.0)
    axes.set_title(
        f"{histogram.domain}: {channel} intensity over "
        f"{histogram.image_count} images ({population} pixels)"
    )
    axes.set_xlabel("Pixel intensity (0-255)")
    axes.set_ylabel(
        "Fraction of pixels" if normalise else "Total pixel count"
    )
    axes.set_xlim(0.0, 255.0)
    axes.grid(True, linestyle="--", alpha=0.6)
    figure.tight_layout()
    # savefig before close, and never plt.show(): show() can clear the figure and
    # is useless headless. The rest of the repo renders through Agg to a path.
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path
