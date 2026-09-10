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
import zipfile
from xml.etree import ElementTree
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spfilm.data import FundusRecord, decode_mask_channels  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.stage3 import (  # noqa: E402
    Stage3ConfigError,
    Stage3DataError,
    Stage3LodoConfig,
    discover_lodo_records,
)
from spfilm.global_histograms import (  # noqa: E402
    BIN_COUNT,
    CHANNELS,
    DOMAIN_COLOURS,
    FOV_LUMINANCE_THRESHOLD,
    LUMA_WEIGHTS,
    POPULATIONS,
    DomainHistograms,
    accumulate_domain,
    bin_centres as _bin_centres,
    plot_domain_overlay,
    plot_global_histogram,
    sample_images,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "domain_shift"

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


def _rebin(density: np.ndarray, bins: int) -> np.ndarray:
    """Coarsen a density to ``bins`` bars by averaging equal groups of fine bins."""

    if BIN_COUNT % bins:
        raise ValueError(f"{BIN_COUNT} bins is not divisible into {bins} bars")
    return density.reshape(bins, BIN_COUNT // bins).mean(axis=1)


def plot_histogram_bars(
    histograms: Sequence[DomainHistograms],
    output_path: Path,
    population: str,
    bins: int = 64,
) -> Path:
    """One barred histogram per domain and channel, on shared axes.

    The overlay figure compares the four domains on one pair of axes, which is
    the right form for judging how far apart they sit but reads as a line chart.
    This is the plain histogram of the same counts: one panel per domain, real
    bars, and a shared y-limit down each column so the panels can be compared by
    eye rather than by reading the tick labels.
    """

    edges = np.linspace(0.0, 1.0, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    width = 1.0 / bins
    figure, axes = plt.subplots(
        len(histograms),
        len(CHANNELS),
        figsize=(4.0 * len(CHANNELS), 2.6 * len(histograms)),
        squeeze=False,
        sharex=True,
    )
    for column, channel in enumerate(CHANNELS):
        coarse = [
            _rebin(histogram.density(population, channel), bins)
            for histogram in histograms
        ]
        ceiling = max(float(values.max()) for values in coarse) * 1.08
        for row, (histogram, values) in enumerate(zip(histograms, coarse)):
            axis = axes[row, column]
            axis.bar(
                centres,
                values,
                width=width,
                color=DOMAIN_COLOURS.get(histogram.domain, "#888888"),
                edgecolor="white",
                linewidth=0.3,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, ceiling)
            axis.grid(axis="y", alpha=0.2, linewidth=0.5)
            if row == 0:
                axis.set_title(channel, fontsize=12)
            if column == 0:
                axis.set_ylabel(f"{histogram.domain}\ndensity", fontsize=9)
            if row == len(histograms) - 1:
                axis.set_xlabel("Normalised intensity")
    scope = (
        "retinal field of view only"
        if population == "fov"
        else "all pixels, including the black surround"
    )
    figure.suptitle(
        f"Pixel-intensity histograms by domain and channel ({scope}, {bins} bins)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_densities(
    histograms: Sequence[DomainHistograms],
    output_path: Path,
    geometry: dict[str, dict[str, np.ndarray]] | None = None,
) -> Path:
    """Cache the accumulated densities so figures can be redrawn without rescanning."""

    payload: dict[str, np.ndarray] = {}
    for domain, fields in (geometry or {}).items():
        for field, values in fields.items():
            payload[f"{domain}|geom|{field}"] = values
    for histogram in histograms:
        payload[f"{histogram.domain}|image_count"] = np.asarray(
            [histogram.image_count]
        )
        for population in POPULATIONS:
            payload[f"{histogram.domain}|pixels|{population}"] = np.asarray(
                [histogram.pixel_counts[population]]
            )
            for channel in CHANNELS:
                payload[f"{histogram.domain}|{population}|{channel}"] = (
                    histogram.density(population, channel)
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_densities(
    input_path: Path,
) -> tuple[list[DomainHistograms], dict[str, dict[str, np.ndarray]]]:
    """Rebuild accumulated histograms from the cache written by a full scan."""

    with np.load(input_path) as archive:
        domains = sorted(
            {key.split("|", 1)[0] for key in archive.files if "|" in key}
        )
        geometry = {
            domain: {
                key.split("|geom|")[1]: archive[key]
                for key in archive.files
                if key.startswith(f"{domain}|geom|")
            }
            for domain in domains
        }
        histograms = [
            DomainHistograms(
                domain=domain,
                image_count=int(archive[f"{domain}|image_count"][0]),
                pixel_counts={
                    population: int(archive[f"{domain}|pixels|{population}"][0])
                    for population in POPULATIONS
                },
                densities={
                    population: {
                        channel: archive[f"{domain}|{population}|{channel}"]
                        for channel in CHANNELS
                    }
                    for population in POPULATIONS
                },
            )
            for domain in domains
        ]
    return histograms, {
        domain: fields for domain, fields in geometry.items() if fields
    }


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


GLAUCOMA = "glaucoma"
NON_GLAUCOMA = "non_glaucoma"
DIAGNOSIS_CLASSES = (GLAUCOMA, NON_GLAUCOMA)
# The four domains record the same fact in three different vocabularies, and one
# of them does not record it on the record at all. Normalising here keeps that
# mess out of the figures.
_GLAUCOMA_WORDS = {"glaucoma", "glaucomatous", "1"}
_NON_GLAUCOMA_WORDS = {"normal", "non_glaucoma", "non-glaucoma", "healthy", "0"}
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _normalise_diagnosis(value: object) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in _GLAUCOMA_WORDS:
        return GLAUCOMA
    if key in _NON_GLAUCOMA_WORDS:
        return NON_GLAUCOMA
    return None


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    """Read the first worksheet of an .xlsx as rows of strings.

    An .xlsx is a zip of XML, so this needs only the standard library. Adding
    openpyxl for one static label file would put an undeclared dependency in
    front of every run, which is the trap OpenCV already set in this project.
    """

    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t"))
                for item in root
            ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.iter(f"{_XLSX_NS}row"):
            cells: list[str] = []
            for cell in row.iter(f"{_XLSX_NS}c"):
                value = cell.find(f"{_XLSX_NS}v")
                text = "" if value is None else (value.text or "")
                if cell.get("t") == "s" and text.isdigit():
                    text = shared[int(text)]
                cells.append(text)
            rows.append(cells)
        return rows


def refuge_validation_labels(
    config: Stage3LodoConfig, project_root: Path
) -> dict[str, str]:
    """Glaucoma labels for REFUGE-Validation400, keyed by image stem.

    Unlike the other three domains, this one carries no diagnosis on the record:
    the loader groups it under a single ``refuge_validation400`` stratum. The
    labels do exist, in the ``Glaucoma Label`` column of the fovea spreadsheet
    that ships beside the masks, so they are read from there rather than leaving
    a quarter of the data out of the comparison.
    """

    for domain_config in config.domains:
        if domain_config.domain.value != "refuge_canon_val":
            continue
        if not domain_config.mask_subdir:
            break
        root = (project_root / domain_config.data_root).resolve()
        spreadsheet = root / Path(domain_config.mask_subdir).parent / "Fovea_locations.xlsx"
        if not spreadsheet.is_file():
            raise Stage3DataError(f"no REFUGE validation labels at {spreadsheet}")
        rows = _read_xlsx_rows(spreadsheet)
        header = rows[0]
        try:
            name_column = header.index("ImgName")
            label_column = header.index("Glaucoma Label")
        except ValueError as error:
            raise Stage3DataError(
                f"{spreadsheet} has no 'Glaucoma Label' column: {header}"
            ) from error
        labels: dict[str, str] = {}
        for row in rows[1:]:
            if len(row) <= max(name_column, label_column) or not row[name_column]:
                continue
            label = _normalise_diagnosis(row[label_column])
            if label is not None:
                labels[Path(row[name_column]).stem] = label
        return labels
    return {}


def diagnosis_of(record: FundusRecord, extra_labels: dict[str, str]) -> str | None:
    """Classify one record as glaucoma or not, or ``None`` if it is unlabelled."""

    for candidate in (record.diagnosis_class, record.stratum):
        label = _normalise_diagnosis(candidate)
        if label is not None:
            return label
    return extra_labels.get(record.image_path.stem)


def group_by_diagnosis(
    records: Sequence[FundusRecord], extra_labels: dict[str, str]
) -> tuple[dict[str, list[FundusRecord]], int]:
    grouped: dict[str, list[FundusRecord]] = {
        label: [] for label in DIAGNOSIS_CLASSES
    }
    unlabelled = 0
    for record in records:
        label = diagnosis_of(record, extra_labels)
        if label is None:
            unlabelled += 1
        else:
            grouped[label].append(record)
    return grouped, unlabelled


def _diagnosis_caps(
    grouped_by_domain: dict[str, dict[str, list[FundusRecord]]],
    balance: str,
    sample: int | None,
) -> dict[str, int | None]:
    """How many images each (domain, class) subset may contribute.

    Densities already make curves of different size comparable in *shape* — an
    unequal draw does not bias where a curve sits, only how noisy it looks. What
    balancing buys is that every curve carries the same sampling noise, so a
    wobble in one cannot be mistaken for a real difference from another. The cost
    is real: ``global`` throws away 329 of REFUGE's 360 non-glaucoma images to
    match Drishti's 31, making every curve as noisy as the worst one.

    ``per-class``
        Each figure is internally balanced: every domain contributes the smallest
        subset available for that class. The glaucoma and non-glaucoma figures
        may still differ from each other.
    ``global``
        One cap across both classes, so all eight curves match.
    ``off``
        Use whatever ``--sample`` says, or everything.
    """

    sizes = {
        label: [
            len(grouped[label])
            for grouped in grouped_by_domain.values()
            if grouped[label]
        ]
        for label in DIAGNOSIS_CLASSES
    }
    if balance == "off":
        return {label: sample for label in DIAGNOSIS_CLASSES}
    if balance == "global":
        every = [size for values in sizes.values() for size in values]
        floor = min(every) if every else None
        caps = {label: floor for label in DIAGNOSIS_CLASSES}
    else:
        caps = {
            label: (min(values) if values else None)
            for label, values in sizes.items()
        }
    if sample is not None:
        caps = {
            label: (sample if cap is None else min(cap, sample))
            for label, cap in caps.items()
        }
    return caps



def _diagnosis_artifacts(
    args: argparse.Namespace,
    histograms: Sequence[DomainHistograms],
    geometry: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    label: str,
    populations: Sequence[str],
) -> list[Path]:
    """Write, for one diagnosis class, everything the pooled run writes.

    The split used to emit one overlay per class in a single channel. That is
    enough to answer "does diagnosis move the curve", but not "does it move the
    curve the same way in every channel", which is the question the RGB input
    actually poses. Producing the full artifact family per class puts the two
    subsets on the same footing as the pooled domains, so a distance read from
    one directory means the same thing as a distance read from the other.

    The single-channel overlay keeps its original filename as well as gaining a
    channel-qualified one: existing write-ups cite the old path.
    """

    report = build_report(histograms)
    report["method"]["working_size"] = args.working_size  # type: ignore[index]
    report["method"]["image_limit"] = args.limit  # type: ignore[index]
    report["method"]["sample_per_domain"] = args.sample  # type: ignore[index]
    report["method"]["sample_seed"] = args.sample_seed  # type: ignore[index]
    report["method"]["diagnosis"] = label  # type: ignore[index]
    report["method"]["balance_diagnosis"] = args.balance_diagnosis  # type: ignore[index]
    if geometry:
        report["geometry"] = {
            domain: {
                field: _distribution(values[field])
                for field in (*GEOMETRY_FIELDS, "aspect_ratio", "long_edge_pixels")
            }
            for domain, values in sorted(geometry.items())
        }

    written: list[Path] = [
        save_densities(histograms, output_dir / f"densities_{label}.npz", geometry),
        _write_json(output_dir / f"domain_shift_{label}.json", report),
        _write_csv(
            output_dir / f"domain_intensity_summary_{label}.csv",
            _summary_rows(report),
        ),
        _write_csv(
            output_dir / f"domain_pairwise_distances_{label}.csv",
            [
                {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
                for row in report["pairwise_distances"]  # type: ignore[union-attr]
            ],
        ),
        _write_csv(
            output_dir / f"domain_leave_one_out_distances_{label}.csv",
            [
                {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
                for row in report["leave_one_out_distances"]  # type: ignore[union-attr]
            ],
        ),
        plot_histograms(
            histograms, output_dir / f"intensity_histograms_{label}.png"
        ),
        plot_leave_one_out_distance(
            histograms, output_dir / f"leave_one_out_distance_{label}.png"
        ),
    ]
    written.extend(
        plot_histogram_bars(
            histograms,
            output_dir / f"intensity_histograms_bars_{population}_{label}.png",
            population,
            bins=args.bars,
        )
        for population in POPULATIONS
    )
    written.extend(
        plot_global_histogram(
            histogram,
            output_dir
            / f"global_histogram_{histogram.domain}_{population}_{label}.png",
            population=population,
        )
        for population in populations
        for histogram in sorted(histograms, key=lambda item: item.domain)
    )

    titles = {
        GLAUCOMA: "Glaucoma images only",
        NON_GLAUCOMA: "Non-glaucoma images only",
    }
    for population in populations:
        for channel in CHANNELS:
            annotations = {}
            for histogram in histograms:
                others = [item for item in histograms if item is not histogram]
                if not others:
                    continue
                rest = pooled_density(others, population, channel)
                annotations[histogram.domain] = (
                    "W₁ to rest "
                    f"{wasserstein_distance(histogram.density(population, channel), rest):.4f}"
                )
            title = (
                f"{titles[label]}: pixel-intensity shift across domains "
                f"({channel}, {population} pixels)"
            )
            paths = [output_dir / f"intensity_overlay_{population}_{channel}_{label}.png"]
            if channel == args.overlay_channel:
                paths.append(output_dir / f"intensity_overlay_{population}_{label}.png")
            written.extend(
                plot_domain_overlay(
                    histograms,
                    path,
                    population=population,
                    channel=channel,
                    annotations=annotations,
                    title=title,
                )
                for path in paths
            )

    if geometry:
        written.append(
            plot_geometry(geometry, output_dir / f"structure_scale_{label}.png")
        )
        written.append(
            _write_csv(
                output_dir / f"domain_structure_scale_{label}.csv",
                [
                    {
                        "domain": domain,
                        "field": field,
                        **{name: round(value, 6) for name, value in statistics.items()},
                    }
                    for domain, fields in sorted(report["geometry"].items())  # type: ignore[union-attr]
                    for field, statistics in fields.items()
                ],
            )
        )
    return written


def run_diagnosis_split(
    args: argparse.Namespace,
    config: Stage3LodoConfig,
    records_by_domain: dict[Domain, Sequence[FundusRecord]],
    output_dir: Path,
    populations: Sequence[str],
) -> tuple[list[Path], list[dict[str, object]], list[dict[str, object]]]:
    """One overlay per diagnosis class, so the shift can be read within each."""

    extra_labels = refuge_validation_labels(config, PROJECT_ROOT)
    by_class: dict[str, list[DomainHistograms]] = {
        label: [] for label in DIAGNOSIS_CLASSES
    }
    geometry_by_class: dict[str, dict[str, dict[str, np.ndarray]]] = {
        label: {} for label in DIAGNOSIS_CLASSES
    }
    rows: list[dict[str, object]] = []
    distances: list[dict[str, object]] = []

    # Group every domain first: the balanced caps are a property of the whole
    # split, so nothing can be scanned until all the subset sizes are known.
    grouped_by_domain: dict[str, dict[str, list[FundusRecord]]] = {}
    for domain in sorted(records_by_domain, key=lambda item: item.value):
        records = records_by_domain[domain]
        if args.limit is not None:
            records = records[: args.limit]
        grouped, unlabelled = group_by_diagnosis(records, extra_labels)
        grouped_by_domain[domain.value] = grouped
        if unlabelled:
            print(
                f"NOTE: {domain.value}: {unlabelled} of {len(records)} images "
                "carry no diagnosis label and are left out of both figures",
                flush=True,
            )

    caps = _diagnosis_caps(grouped_by_domain, args.balance_diagnosis, args.sample)
    for label, cap in sorted(caps.items()):
        if cap is not None:
            print(f"balancing {label} to n={cap} per domain", flush=True)

    for domain_name, grouped in grouped_by_domain.items():
        for label in DIAGNOSIS_CLASSES:
            subset = grouped[label]
            available = len(subset)
            if not subset:
                print(
                    f"NOTE: {domain_name} has no {label} images; it is absent "
                    "from that figure",
                    flush=True,
                )
                continue
            cap = caps[label]
            if cap is not None:
                subset = sample_images(
                    f"{domain_name}:{label}", subset, cap, args.sample_seed
                )
            print(
                f"scanning {domain_name} {label}: {len(subset)} of {available} "
                f"images at {args.working_size}px",
                flush=True,
            )
            histogram = accumulate_domain(
                domain_name, subset, args.working_size, args.workers
            )
            by_class[label].append(histogram)
            if not args.skip_geometry:
                print(f"measuring {domain_name} {label} mask geometry", flush=True)
                geometry_by_class[label][domain_name] = accumulate_geometry(
                    domain_name, subset, args.workers
                )
            rows.append(
                {
                    "domain": domain_name,
                    "diagnosis": label,
                    "images_available": available,
                    "images_used": len(subset),
                }
            )

    # The question is whether diagnosis moves the distribution as much as
    # acquisition does. A within-domain comparison answers it directly and,
    # unlike a distance to the pooled rest, does not depend on how the domains
    # happen to be weighted against each other. It is reported per channel
    # because a diagnosis that shifted only one channel would be invisible in
    # the luminance projection, which weights green at 0.587.
    for population in populations:
        for channel in CHANNELS:
            paired = {
                label: {h.domain: h for h in by_class[label]}
                for label in DIAGNOSIS_CLASSES
            }
            for domain_name in sorted(paired[GLAUCOMA]):
                if domain_name not in paired[NON_GLAUCOMA]:
                    continue
                distance = wasserstein_distance(
                    paired[GLAUCOMA][domain_name].density(population, channel),
                    paired[NON_GLAUCOMA][domain_name].density(population, channel),
                )
                distances.append(
                    {
                        "domain": domain_name,
                        "population": population,
                        "channel": channel,
                        "wasserstein_1_glaucoma_vs_non": round(distance, 6),
                    }
                )

    figure_paths: list[Path] = []
    for label in DIAGNOSIS_CLASSES:
        if not by_class[label]:
            continue
        figure_paths.extend(
            _diagnosis_artifacts(
                args,
                by_class[label],
                geometry_by_class[label],
                output_dir,
                label,
                populations,
            )
        )
    return figure_paths, rows, distances



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
        "--sample",
        type=int,
        help=(
            "Use a random sample of N images per domain instead of all of them, "
            "so every domain contributes an equally noisy curve. Domains with "
            "fewer than N images are used whole"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for --sample, so a sampled run can be reproduced exactly",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Redraw figures from a previous scan's densities.npz without rescanning",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=64,
        help="Bar count for the histogram figures; must divide 256",
    )
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip mask decoding and report only the intensity distributions",
    )
    parser.add_argument(
        "--split-diagnosis",
        action="store_true",
        help=(
            "Emit one domain overlay per diagnosis class (glaucoma and "
            "non-glaucoma) instead of the pooled analysis"
        ),
    )
    parser.add_argument(
        "--balance-diagnosis",
        choices=("off", "per-class", "global"),
        default="off",
        help=(
            "Cap every domain's diagnosis subset to the smallest one, so the "
            "curves carry equal sampling noise. 'per-class' balances within each "
            "figure, 'global' uses one cap across both"
        ),
    )
    parser.add_argument(
        "--overlay-channel",
        choices=CHANNELS,
        default="gray",
        help="Channel for the single-axes domain overlay figure",
    )
    parser.add_argument(
        "--global-population",
        choices=(*POPULATIONS, "both"),
        default="fov",
        help=(
            "Pixel population for the per-domain global histograms. 'fov' is the "
            "default because the black surround in 'all' puts a spike at zero "
            "that flattens the tissue distribution on a linear axis"
        ),
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

    sampling = args.sample is not None
    if sampling and args.output_dir == DEFAULT_OUTPUT_DIR:
        # Keep the full-sweep artifacts intact: the W1 distances quoted in the
        # write-up come from all 1,386 images, not from a sample.
        args.output_dir = DEFAULT_OUTPUT_DIR.with_name(
            f"{DEFAULT_OUTPUT_DIR.name}_sample{args.sample}"
        )
    if args.split_diagnosis:
        if args.from_cache:
            print(
                "FATAL: --split-diagnosis needs a scan; the cache stores pooled "
                "domains only",
                file=sys.stderr,
            )
            return 2
        output_dir = args.output_dir.expanduser().resolve()
        populations = (
            POPULATIONS
            if args.global_population == "both"
            else (args.global_population,)
        )
        try:
            figure_paths, rows, distances = run_diagnosis_split(
                args, config, records_by_domain, output_dir, populations
            )
        except (Stage3DataError, OSError, ValueError) as error:
            print(f"FATAL: {error}", file=sys.stderr)
            return 2
        counts_path = _write_csv(output_dir / "diagnosis_split_counts.csv", rows)
        distance_path = _write_csv(
            output_dir / "diagnosis_within_domain_distance.csv", distances
        )
        print()
        print("within-domain glaucoma vs non-glaucoma (W1), for scale against")
        print("the between-domain distances in the pooled run:")
        print(f"  {'domain':18s}" + "".join(f"{channel:>9s}" for channel in CHANNELS))
        by_domain: dict[str, dict[str, float]] = {}
        for row in distances:
            if row["population"] != "fov":
                continue
            by_domain.setdefault(str(row["domain"]), {})[str(row["channel"])] = float(
                row["wasserstein_1_glaucoma_vs_non"]
            )
        for domain_name, channels in sorted(by_domain.items()):
            print(
                f"  {domain_name:18s}"
                + "".join(f"{channels.get(channel, float('nan')):9.4f}" for channel in CHANNELS)
            )
        print("  (fov pixels; the full table, both populations, is in the CSV)")
        print()
        for path in (*figure_paths, counts_path, distance_path):
            print(f"wrote {path}")
        return 0

    cache_name = (
        f"densities_sample{args.sample}_seed{args.sample_seed}.npz"
        if sampling
        else "densities.npz"
    )
    cache_path = args.output_dir.expanduser().resolve() / cache_name
    histograms: list[DomainHistograms] = []
    geometry: dict[str, dict[str, np.ndarray]] = {}
    if args.from_cache:
        if not cache_path.is_file():
            print(f"FATAL: no cached scan at {cache_path}", file=sys.stderr)
            return 2
        histograms, geometry = load_densities(cache_path)
        print(f"redrawing from {cache_path} without rescanning", flush=True)
    for domain in sorted(records_by_domain, key=lambda item: item.value):
        if args.from_cache:
            break
        records = records_by_domain[domain]
        if args.limit is not None:
            records = records[: args.limit]
        if sampling:
            available = len(records)
            records = sample_images(
                domain.value, records, args.sample, args.sample_seed
            )
            if available < args.sample:
                print(
                    f"NOTE: {domain.value} has only {available} images, "
                    f"fewer than the requested sample of {args.sample}",
                    flush=True,
                )
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
    report["method"]["sample_per_domain"] = args.sample  # type: ignore[index]
    report["method"]["sample_seed"] = (  # type: ignore[index]
        args.sample_seed if sampling else None
    )

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
    if not args.from_cache:
        save_densities(histograms, cache_path, geometry)
    bar_figure_paths = [
        plot_histogram_bars(
            histograms,
            output_dir / f"intensity_histograms_bars_{population}.png",
            population,
            bins=args.bars,
        )
        for population in POPULATIONS
    ]
    figure_path = plot_histograms(
        histograms, output_dir / "intensity_histograms.png"
    )
    distance_figure_path = plot_leave_one_out_distance(
        histograms, output_dir / "leave_one_out_distance.png"
    )
    global_populations = (
        POPULATIONS
        if args.global_population == "both"
        else (args.global_population,)
    )
    global_figure_paths = [
        plot_global_histogram(
            histogram,
            output_dir / f"global_histogram_{histogram.domain}_{population}.png",
            population=population,
        )
        for population in global_populations
        for histogram in sorted(histograms, key=lambda item: item.domain)
    ]
    overlay_figure_paths = []
    for population in global_populations:
        # One overlay per channel, because the network is fed RGB and the
        # luminance projection can hide a channel-specific gap: two domains
        # whose gray curves coincide may still be several times further apart
        # in red or blue, and luma weights green at 0.587 so a red-and-blue
        # disagreement can cancel out of it entirely.
        for channel in CHANNELS:
            # Distance from each domain to the pooled other three, so the legend
            # says how far apart the curves are rather than leaving it to the eye.
            annotations = {}
            for histogram in histograms:
                others = [item for item in histograms if item is not histogram]
                if not others:
                    continue
                rest = pooled_density(others, population, channel)
                distance = wasserstein_distance(
                    histogram.density(population, channel), rest
                )
                annotations[histogram.domain] = f"W\u2081 to rest {distance:.4f}"
            paths = [output_dir / f"intensity_overlay_{population}_{channel}.png"]
            if channel == args.overlay_channel:
                # The unqualified name is cited by the write-ups; keep it.
                paths.append(output_dir / f"intensity_overlay_{population}.png")
            overlay_figure_paths.extend(
                plot_domain_overlay(
                    histograms,
                    path,
                    population=population,
                    channel=channel,
                    annotations=annotations,
                )
                for path in paths
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
        *bar_figure_paths,
        *global_figure_paths,
        *overlay_figure_paths,
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
