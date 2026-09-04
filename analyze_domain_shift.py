#!/usr/bin/env python3
"""Quantify the acquisition shift between the four fundus domains.

Stage 3 measures how far segmentation performance falls when a model meets an
unseen acquisition domain. This script characterises the shift itself, before any
model is involved, as the difference between the domains' pixel-intensity
distributions.

Two populations of pixels are reported for every domain, because they answer
different questions:

``all``
    Every pixel of the source image. REFUGE and Drishti-GS store a circular
    retinal field of view inside a rectangular frame, so a large share of their
    pixels are the black surround. RIM-ONE-DL ships square optic-nerve-head crops
    with almost no surround. The ``all`` histograms therefore show the *framing*
    difference, which is real and is what the network's input tensor contains.

``fov``
    Only pixels inside the retinal field of view. This removes the framing
    difference and isolates the *photometric* difference: illumination, camera
    response, and pigmentation.

A domain gap in ``fov`` means the cameras genuinely disagree about colour. A gap
that exists only in ``all`` means the images are cropped differently. The two
have different remedies, so they are never pooled here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from spfilm.data import FundusRecord, decode_mask_channels  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.stage3 import (  # noqa: E402
    Stage3ConfigError,
    Stage3DataError,
    Stage3LodoConfig,
    discover_lodo_records,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "domain_shift"

BIN_COUNT = 256
CHANNELS = ("gray", "red", "green", "blue")
POPULATIONS = ("all", "fov")
# Below this luminance a pixel is the black surround outside the retinal disc
# rather than tissue. Fundus surrounds sit within a few counts of zero, so the
# threshold is far from any real retinal intensity and the choice is not
# delicate; it is recorded in the JSON so the figure can be reproduced.
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


@dataclass(frozen=True)
class DomainHistograms:
    """Normalised intensity densities for one domain, per channel and population."""

    domain: str
    image_count: int
    pixel_counts: dict[str, int]
    densities: dict[str, dict[str, np.ndarray]]

    def density(self, population: str, channel: str) -> np.ndarray:
        return self.densities[population][channel]


def _bin_edges() -> np.ndarray:
    return np.linspace(0.0, 1.0, BIN_COUNT + 1)


def _bin_centres() -> np.ndarray:
    edges = _bin_edges()
    return (edges[:-1] + edges[1:]) / 2.0


def _load_pixels(image_path: Path, working_size: int) -> np.ndarray:
    """Decode one image to a small RGB float array in [0, 1].

    ``draft`` lets the JPEG decoder downscale while decoding, which is what makes
    it affordable to sweep every REFUGE image at full dataset size. The intensity
    *distribution* is what is being estimated, and it is insensitive to this
    resampling; the exact pixel grid is not needed.
    """

    with Image.open(image_path) as image:
        image.draft("RGB", (working_size, working_size))
        image = image.convert("RGB")
        image.thumbnail((working_size, working_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _image_histograms(
    image_path: Path, working_size: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    pixels = _load_pixels(image_path, working_size)
    luminance = pixels @ np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    fov = luminance > FOV_LUMINANCE_THRESHOLD
    edges = _bin_edges()

    bands = {
        "gray": luminance,
        "red": pixels[..., 0],
        "green": pixels[..., 1],
        "blue": pixels[..., 2],
    }
    counts: dict[str, np.ndarray] = {}
    for population in POPULATIONS:
        selector = Ellipsis if population == "all" else fov
        for channel, band in bands.items():
            values = band if population == "all" else band[selector]
            counts[f"{population}/{channel}"] = np.histogram(
                values, bins=edges
            )[0].astype(np.float64)
    pixel_counts = {
        "all": int(luminance.size),
        "fov": int(fov.sum()),
    }
    return counts, pixel_counts


def _worker(payload: tuple[str, int]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    image_path, working_size = payload
    return _image_histograms(Path(image_path), working_size)


def accumulate_domain(
    domain: str,
    records: Sequence[FundusRecord],
    working_size: int,
    workers: int,
) -> DomainHistograms:
    """Sum per-image histograms, then normalise once at the end.

    Summing counts and normalising afterwards weights each domain by its pixels,
    not by its images, so a domain is not skewed by having a handful of unusually
    large frames.
    """

    totals: dict[str, np.ndarray] = {
        f"{population}/{channel}": np.zeros(BIN_COUNT, dtype=np.float64)
        for population in POPULATIONS
        for channel in CHANNELS
    }
    pixel_totals = {population: 0 for population in POPULATIONS}
    payloads = [(str(record.image_path), working_size) for record in records]

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
                raise Stage3DataError(
                    f"{domain} {population}/{channel} accumulated no pixels"
                )
            # A density, not a count: the curves are comparable across domains
            # of very different size, and each integrates to one.
            densities[population][channel] = counts / total / width

    return DomainHistograms(
        domain=domain,
        image_count=len(records),
        pixel_counts=pixel_totals,
        densities=densities,
    )


def _record_geometry(record: FundusRecord) -> dict[str, float]:
    """Measure how large the target structures are relative to the frame.

    The training pipeline letterboxes each source image by its longest edge, so
    a structure's size in the tensor the network sees is its size relative to
    that edge. A domain whose discs occupy a different fraction of the frame
    presents the network with objects at a scale it has never been trained on,
    which no amount of intensity matching would fix.
    """

    masks = decode_mask_channels(record)
    disc = masks[0].astype(bool)
    cup = masks[1].astype(bool)
    height, width = disc.shape
    pixels = float(height * width)
    disc_area = float(disc.sum())
    cup_area = float(cup.sum())
    long_edge = float(max(height, width))
    return {
        "disc_area_fraction": disc_area / pixels,
        "cup_area_fraction": cup_area / pixels,
        "cup_to_disc_area_ratio": (cup_area / disc_area) if disc_area > 0 else float("nan"),
        # Diameter of a circle with the same area, as a fraction of the edge the
        # letterbox scales by.
        "disc_diameter_fraction": (
            float(np.sqrt(4.0 * disc_area / np.pi)) / long_edge if long_edge > 0 else float("nan")
        ),
        "aspect_ratio": width / height if height > 0 else float("nan"),
        "long_edge_pixels": long_edge,
    }


GEOMETRY_FIELDS = (
    "disc_area_fraction",
    "cup_area_fraction",
    "cup_to_disc_area_ratio",
    "disc_diameter_fraction",
)


def accumulate_geometry(
    domain: str,
    records: Sequence[FundusRecord],
    workers: int,
) -> dict[str, np.ndarray]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            measured = list(pool.map(_record_geometry, records, chunksize=4))
    else:
        measured = [_record_geometry(record) for record in records]
    if not measured:
        raise Stage3DataError(f"{domain} produced no geometry measurements")
    return {
        field: np.asarray([row[field] for row in measured], dtype=np.float64)
        for field in (*GEOMETRY_FIELDS, "aspect_ratio", "long_edge_pixels")
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: float("nan") for name in ("mean", "std", "p1", "p25", "p50", "p75", "p99")}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        **{
            f"p{level}": float(np.percentile(finite, level))
            for level in (1, 25, 50, 75, 99)
        },
    }


def plot_geometry(
    geometry: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> Path:
    domains = sorted(geometry)
    labels = {
        "disc_area_fraction": "Disc area / frame area",
        "cup_area_fraction": "Cup area / frame area",
        "disc_diameter_fraction": "Disc diameter / longest edge",
        "cup_to_disc_area_ratio": "Cup area / disc area",
    }
    figure, axes = plt.subplots(1, len(GEOMETRY_FIELDS), figsize=(17, 4.4), squeeze=False)
    for column, field in enumerate(GEOMETRY_FIELDS):
        axis = axes[0, column]
        data = [
            geometry[domain][field][np.isfinite(geometry[domain][field])]
            for domain in domains
        ]
        parts = axis.boxplot(
            data,
            tick_labels=domains,
            patch_artist=True,
            widths=0.6,
            flierprops={"markersize": 2, "alpha": 0.4},
        )
        for patch, domain in zip(parts["boxes"], domains):
            patch.set_facecolor(DOMAIN_COLOURS.get(domain, "#888888"))
            patch.set_alpha(0.65)
        for median in parts["medians"]:
            median.set_color("black")
        axis.set_title(labels[field], fontsize=11)
        axis.tick_params(axis="x", rotation=35, labelsize=9)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        # Disc and cup area fractions differ by an order of magnitude between the
        # cropped and the full-frame domains, which a linear axis would compress
        # into a line at zero for three of the four.
        if field in {"disc_area_fraction", "cup_area_fraction"}:
            axis.set_yscale("log")
    figure.suptitle(
        "Target structure scale relative to the frame, by acquisition domain",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _probabilities(density: np.ndarray) -> np.ndarray:
    return density / density.sum()


def wasserstein_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Earth mover's distance on the [0, 1] intensity axis.

    Reported in intensity units, so 0.05 means the two domains differ by about
    5% of the full dynamic range once the cheapest transport is used. Unlike a
    divergence it stays finite and interpretable when the supports differ.
    """

    left_cdf = np.cumsum(_probabilities(left))
    right_cdf = np.cumsum(_probabilities(right))
    # Uniform bins, so the integral of |F - G| is the bin width times the sum.
    return float(np.sum(np.abs(left_cdf - right_cdf)) / BIN_COUNT)


def jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Square root of the JS divergence in bits: a bounded metric in [0, 1]."""

    p = _probabilities(left)
    q = _probabilities(right)
    m = (p + q) / 2.0

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    divergence = (_kl(p, m) + _kl(q, m)) / 2.0
    return float(np.sqrt(max(divergence, 0.0)))


def hellinger_distance(left: np.ndarray, right: np.ndarray) -> float:
    p = _probabilities(left)
    q = _probabilities(right)
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))


def summarise(density: np.ndarray) -> dict[str, float]:
    centres = _bin_centres()
    probabilities = _probabilities(density)
    mean = float(np.sum(centres * probabilities))
    variance = float(np.sum(((centres - mean) ** 2) * probabilities))
    cdf = np.cumsum(probabilities)
    percentiles = {
        f"p{int(level * 100)}": float(centres[int(np.searchsorted(cdf, level))])
        for level in (0.01, 0.25, 0.50, 0.75, 0.99)
    }
    return {"mean": mean, "std": float(np.sqrt(variance)), **percentiles}


def pooled_density(
    histograms: Sequence[DomainHistograms],
    population: str,
    channel: str,
) -> np.ndarray:
    """Weight each domain's density by its pixel count, as pooled training would."""

    weights = np.asarray(
        [histogram.pixel_counts[population] for histogram in histograms],
        dtype=np.float64,
    )
    stacked = np.stack(
        [histogram.density(population, channel) for histogram in histograms]
    )
    return np.average(stacked, axis=0, weights=weights)


def plot_histograms(
    histograms: Sequence[DomainHistograms],
    output_path: Path,
) -> Path:
    figure, axes = plt.subplots(
        len(CHANNELS),
        len(POPULATIONS),
        figsize=(13, 3.1 * len(CHANNELS)),
        squeeze=False,
        sharex=True,
    )
    centres = _bin_centres()
    titles = {
        "all": "All pixels, log density (includes the black surround)",
        "fov": f"Retinal field of view only (luminance > {FOV_LUMINANCE_THRESHOLD})",
    }
    for row, channel in enumerate(CHANNELS):
        for column, population in enumerate(POPULATIONS):
            axis = axes[row, column]
            for histogram in histograms:
                axis.plot(
                    centres,
                    histogram.density(population, channel),
                    label=histogram.domain,
                    color=DOMAIN_COLOURS.get(histogram.domain),
                    linewidth=1.6,
                )
            axis.set_ylabel(f"{channel}\ndensity" if column == 0 else "")
            axis.set_xlim(0.0, 1.0)
            axis.margins(y=0.05)
            if population == "all":
                # The black surround puts a spike of two orders of magnitude at
                # zero. On a linear axis it flattens every retinal intensity into
                # the baseline, so the panel would show only the framing artefact.
                axis.set_yscale("log")
            if row == 0:
                axis.set_title(titles[population], fontsize=11)
            if row == len(CHANNELS) - 1:
                axis.set_xlabel("Normalised intensity")
            axis.grid(alpha=0.25, linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(histograms),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "Normalised pixel-intensity distributions by acquisition domain",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_leave_one_out_distance(
    histograms: Sequence[DomainHistograms],
    output_path: Path,
    channel: str = "gray",
) -> Path:
    """Each domain's distance from the pooled distribution of the other three.

    This is the quantity the leave-one-domain-out result should track: in that
    protocol a model sees the pooled remainder and is scored on the domain left
    out, so the further a domain sits from the rest, the larger the shift it is
    asked to absorb.
    """

    figure, axes = plt.subplots(1, len(POPULATIONS), figsize=(11, 4), squeeze=False)
    for column, population in enumerate(POPULATIONS):
        axis = axes[0, column]
        names: list[str] = []
        values: list[float] = []
        for histogram in histograms:
            others = [item for item in histograms if item.domain != histogram.domain]
            rest = pooled_density(others, population, channel)
            names.append(histogram.domain)
            values.append(
                wasserstein_distance(histogram.density(population, channel), rest)
            )
        order = np.argsort(values)
        axis.barh(
            [names[index] for index in order],
            [values[index] for index in order],
            color=[DOMAIN_COLOURS.get(names[index]) for index in order],
        )
        axis.set_xlabel("Wasserstein-1 distance (intensity units)")
        axis.set_title(
            f"{population} pixels, {channel} channel",
            fontsize=11,
        )
        axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    figure.suptitle(
        "Distance from each domain to the pooled remaining three", fontsize=13
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def build_report(histograms: Sequence[DomainHistograms]) -> dict[str, object]:
    domains = [histogram.domain for histogram in histograms]
    by_domain = {histogram.domain: histogram for histogram in histograms}

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for histogram in histograms:
        summaries[histogram.domain] = {
            f"{population}/{channel}": summarise(
                histogram.density(population, channel)
            )
            for population in POPULATIONS
            for channel in CHANNELS
        }

    pairwise: list[dict[str, object]] = []
    for index, left in enumerate(domains):
        for right in domains[index + 1 :]:
            for population in POPULATIONS:
                for channel in CHANNELS:
                    a = by_domain[left].density(population, channel)
                    b = by_domain[right].density(population, channel)
                    pairwise.append(
                        {
                            "domain_a": left,
                            "domain_b": right,
                            "population": population,
                            "channel": channel,
                            "wasserstein_1": wasserstein_distance(a, b),
                            "jensen_shannon": jensen_shannon_distance(a, b),
                            "hellinger": hellinger_distance(a, b),
                        }
                    )

    leave_one_out: list[dict[str, object]] = []
    for histogram in histograms:
        others = [item for item in histograms if item.domain != histogram.domain]
        for population in POPULATIONS:
            for channel in CHANNELS:
                rest = pooled_density(others, population, channel)
                own = histogram.density(population, channel)
                leave_one_out.append(
                    {
                        "domain": histogram.domain,
                        "population": population,
                        "channel": channel,
                        "wasserstein_1_to_rest": wasserstein_distance(own, rest),
                        "jensen_shannon_to_rest": jensen_shannon_distance(own, rest),
                        "hellinger_to_rest": hellinger_distance(own, rest),
                    }
                )

    return {
        "method": {
            "bin_count": BIN_COUNT,
            "intensity_range": [0.0, 1.0],
            "fov_luminance_threshold": FOV_LUMINANCE_THRESHOLD,
            "luma_weights": list(LUMA_WEIGHTS),
            "normalisation": (
                "per-image counts summed per domain, then divided by the domain's "
                "total pixels and the bin width, so each curve is a density that "
                "integrates to one"
            ),
            "populations": {
                "all": "every pixel of the source image, including the black surround",
                "fov": (
                    "pixels whose Rec. 601 luminance exceeds the threshold, which "
                    "removes the black surround and isolates the photometric shift"
                ),
            },
            "distances": {
                "wasserstein_1": "intensity units on [0, 1]; lower is more similar",
                "jensen_shannon": "bounded metric in [0, 1], square root of the JS divergence in bits",
                "hellinger": "bounded metric in [0, 1]",
            },
        },
        "domains": {
            histogram.domain: {
                "image_count": histogram.image_count,
                "pixel_counts": histogram.pixel_counts,
                "fov_pixel_fraction": (
                    histogram.pixel_counts["fov"] / histogram.pixel_counts["all"]
                ),
                "summaries": summaries[histogram.domain],
            }
            for histogram in histograms
        },
        "pairwise_distances": pairwise,
        "leave_one_out_distances": leave_one_out,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"Refusing to write an empty table to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summary_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain, payload in sorted(report["domains"].items()):  # type: ignore[union-attr]
        for key, statistics in sorted(payload["summaries"].items()):
            population, channel = key.split("/")
            rows.append(
                {
                    "domain": domain,
                    "population": population,
                    "channel": channel,
                    "images": payload["image_count"],
                    "fov_pixel_fraction": round(payload["fov_pixel_fraction"], 6),
                    **{
                        name: round(value, 6)
                        for name, value in statistics.items()
                    },
                }
            )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalised pixel-intensity distributions across the four fundus "
            "acquisition domains, with the distances between them"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Stage 3 JSON config; only its domain discovery settings are used",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the figures, tables, and JSON report",
    )
    parser.add_argument(
        "--working-size",
        type=int,
        default=256,
        help="Longest edge each image is decoded to before counting pixels",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Use only the first N images per domain (for a quick rehearsal)",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip mask decoding and report only the intensity distributions",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel image decoders",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = Stage3LodoConfig.from_json(args.config.expanduser().resolve())
        records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
    except (Stage3ConfigError, Stage3DataError, OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    histograms: list[DomainHistograms] = []
    geometry: dict[str, dict[str, np.ndarray]] = {}
    for domain in sorted(records_by_domain, key=lambda item: item.value):
        records = records_by_domain[domain]
        if args.limit is not None:
            records = records[: args.limit]
        print(
            f"scanning {domain.value}: {len(records)} images "
            f"at {args.working_size}px with {args.workers} workers",
            flush=True,
        )
        histograms.append(
            accumulate_domain(
                domain.value, records, args.working_size, args.workers
            )
        )
        if not args.skip_geometry:
            print(f"measuring {domain.value} mask geometry", flush=True)
            geometry[domain.value] = accumulate_geometry(
                domain.value, records, args.workers
            )

    report = build_report(histograms)
    report["method"]["working_size"] = args.working_size  # type: ignore[index]
    report["method"]["image_limit"] = args.limit  # type: ignore[index]

    if geometry:
        report["geometry"] = {
            domain: {
                field: _distribution(values[field])
                for field in (*GEOMETRY_FIELDS, "aspect_ratio", "long_edge_pixels")
            }
            for domain, values in sorted(geometry.items())
        }
        report["method"]["geometry"] = (  # type: ignore[index]
            "structure areas measured on the decoded native-resolution masks; "
            "disc_diameter_fraction is the equal-area circle diameter divided by "
            "the image's longest edge, which is the edge the training pipeline "
            "letterboxes by, so it is the scale the network actually sees"
        )

    output_dir = args.output_dir.expanduser().resolve()
    figure_path = plot_histograms(
        histograms, output_dir / "intensity_histograms.png"
    )
    distance_figure_path = plot_leave_one_out_distance(
        histograms, output_dir / "leave_one_out_distance.png"
    )
    report_path = _write_json(output_dir / "domain_shift.json", report)
    summary_path = _write_csv(
        output_dir / "domain_intensity_summary.csv", _summary_rows(report)
    )
    pairwise_path = _write_csv(
        output_dir / "domain_pairwise_distances.csv",
        [
            {
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in row.items()
            }
            for row in report["pairwise_distances"]  # type: ignore[union-attr]
        ],
    )
    leave_one_out_path = _write_csv(
        output_dir / "domain_leave_one_out_distances.csv",
        [
            {
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in row.items()
            }
            for row in report["leave_one_out_distances"]  # type: ignore[union-attr]
        ],
    )

    geometry_figure_path = None
    geometry_table_path = None
    if geometry:
        geometry_figure_path = plot_geometry(
            geometry, output_dir / "structure_scale.png"
        )
        geometry_table_path = _write_csv(
            output_dir / "domain_structure_scale.csv",
            [
                {
                    "domain": domain,
                    "field": field,
                    **{
                        name: round(value, 6)
                        for name, value in statistics.items()
                    },
                }
                for domain, fields in sorted(report["geometry"].items())  # type: ignore[union-attr]
                for field, statistics in fields.items()
            ],
        )

    print()
    print("field-of-view pixel fraction (1.0 means no black surround):")
    for histogram in histograms:
        fraction = histogram.pixel_counts["fov"] / histogram.pixel_counts["all"]
        print(
            f"  {histogram.domain:<18} {fraction:6.3f}  "
            f"({histogram.image_count} images)"
        )

    print()
    print("distance from each domain to the pooled other three (gray channel):")
    for population in POPULATIONS:
        print(f"  {population} pixels:")
        rows = [
            row
            for row in report["leave_one_out_distances"]  # type: ignore[union-attr]
            if row["population"] == population and row["channel"] == "gray"
        ]
        for row in sorted(rows, key=lambda item: item["wasserstein_1_to_rest"]):
            print(
                f"    {row['domain']:<18} W1={row['wasserstein_1_to_rest']:.4f}  "
                f"JS={row['jensen_shannon_to_rest']:.4f}"
            )

    if geometry:
        print()
        print("target structure scale relative to the frame (median):")
        print(
            f"  {'domain':<18} {'disc/frame':>11} {'cup/frame':>10} "
            f"{'disc dia/edge':>14} {'cup/disc':>9}"
        )
        for domain in sorted(report["geometry"]):  # type: ignore[union-attr]
            fields = report["geometry"][domain]  # type: ignore[index]
            print(
                f"  {domain:<18} "
                f"{fields['disc_area_fraction']['p50']:>11.4f} "
                f"{fields['cup_area_fraction']['p50']:>10.4f} "
                f"{fields['disc_diameter_fraction']['p50']:>14.4f} "
                f"{fields['cup_to_disc_area_ratio']['p50']:>9.4f}"
            )

    print()
    for path in (
        figure_path,
        distance_figure_path,
        geometry_figure_path,
        report_path,
        summary_path,
        pairwise_path,
        leave_one_out_path,
        geometry_table_path,
    ):
        if path is not None:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
