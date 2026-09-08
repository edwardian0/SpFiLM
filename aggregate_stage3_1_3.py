#!/usr/bin/env python3
"""Aggregate the Stage 3 train-on-one, test-on-three runs into a per-domain report.

The companion tool ``aggregate_stage3.py`` reduces the leave-one-domain-out arm,
where each run yields one held-out score. This arm yields *three* scores per run,
one for each unseen target domain, and they must never be averaged together: the
whole point of the protocol is the spread between them, and RIM-ONE-DL's HD95 is
in a different coordinate frame from the others, so a pooled distance would mix
units.

Accordingly this tool reads only ``test_by_domain`` from each run. It refuses to
read ``test_pooled``, and it recomputes every summary from the per-image metric
CSVs rather than trusting any stored number, so the report and the files on disk
cannot disagree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from spfilm.lodo import Domain  # noqa: E402
from spfilm.metrics import CHANNEL_NAMES, summarise_per_image_csv  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SINGLE_SOURCE_PROTOCOL_NAME,
    SingleSourceManifest,
    load_single_source_manifest,
)


TEST_METRICS_NAME = "test_metrics.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "splits" / "single_source" / "single_source_manifest.json"
DEFAULT_RUN_ROOTS = (PROJECT_ROOT / "artifacts",)
DEFAULT_EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CONFIDENCE_LEVEL = 0.95
SEED_METRICS = ("dice", "iou", "hd95")
METRIC_LABELS = {"dice": "Dice", "iou": "IoU", "hd95": "HD95"}
HD95_UNIT_NATIVE = "native pixels"
HD95_UNIT_GRID = "letterboxed-grid pixels"
NATIVE_HD95_DOMAIN = "rim_one_dl"
SUMMARY_RELATIVE_TOLERANCE = 1e-9
SUMMARY_ABSOLUTE_TOLERANCE = 1e-12
TODO = "<!-- TODO: written by hand; the tool does not infer this. -->"


class Stage3SingleSourceReportError(ValueError):
    """Raised when the discovered runs cannot support an honest report."""


# --------------------------------------------------------------------------
# Stage A: load one run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunIdentity:
    """The provenance that must agree across every run in one report."""

    arm: str
    source_domain: Domain
    run_seed: int
    manifest_sha256: str
    parent_manifest_sha256: str
    config_sha256: str
    git_revision: str
    completed_at_utc: str
    train_budget: int
    val_budget: int
    test_budget: int | None


@dataclass(frozen=True)
class SingleSourceRun:
    """One completed run and the three target evaluations it produced."""

    identity: RunIdentity
    directory: Path
    metrics_path: Path
    target_domains: tuple[Domain, ...]
    per_image_csv: Mapping[Domain, Path]
    stored: Mapping[Domain, Mapping[str, Any]]
    hd95_unit: Mapping[Domain, str]

    @property
    def label(self) -> str:
        return f"{self.identity.source_domain.value}/seed_{self.identity.run_seed}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage3SingleSourceReportError(f"Cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Stage3SingleSourceReportError(f"{path} must hold a JSON object")
    return payload


def _require(payload: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise Stage3SingleSourceReportError(f"{context} is missing {key!r}")
    return payload[key]


def _require_sha256(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = _require(payload, key, context)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage3SingleSourceReportError(
            f"{context} {key} must be a lowercase hex SHA-256 digest"
        )
    return value


def _parse_domain(value: object, context: str) -> Domain:
    if not isinstance(value, str):
        raise Stage3SingleSourceReportError(f"{context} must be a domain string")
    try:
        return Domain(value)
    except ValueError as error:
        raise Stage3SingleSourceReportError(
            f"Unknown domain in {context}: {value!r}"
        ) from error


def build_run(metrics_path: Path) -> SingleSourceRun | None:
    """Load one run, or return None when the file is not from this arm.

    A directory that merely sits under the search root is not evidence. Anything
    that claims to be from this arm but is incomplete raises instead of being
    skipped, because a silently dropped run is a silently wrong mean.
    """

    payload = _read_json(metrics_path)
    metadata = payload.get("single_source")
    if not isinstance(metadata, dict):
        return None
    context = str(metrics_path)
    if _require(metadata, "protocol", context) != SINGLE_SOURCE_PROTOCOL_NAME:
        return None
    if metadata.get("smoke_rehearsal") is True:
        return None
    if metadata.get("scientific_result") is not True:
        raise Stage3SingleSourceReportError(
            f"{context} is not marked scientific_result; refusing to report it"
        )
    if "test_pooled" in payload and "test_by_domain" not in payload:
        raise Stage3SingleSourceReportError(
            f"{context} has no test_by_domain block; it predates the per-domain "
            "reporting schema and must be re-run"
        )
    if "test" in payload:
        raise Stage3SingleSourceReportError(
            f"{context} carries a bare 'test' key from the superseded schema; "
            "re-run it so the per-domain results are unambiguous"
        )

    source_domain = _parse_domain(
        _require(metadata, "source_domain", context), f"{context} source_domain"
    )
    target_domains = tuple(
        _parse_domain(value, f"{context} target_domains")
        for value in _require(metadata, "target_domains", context)
    )
    if source_domain in target_domains:
        raise Stage3SingleSourceReportError(
            f"{context} lists its source domain as a target"
        )
    if len(set(target_domains)) != len(target_domains):
        raise Stage3SingleSourceReportError(f"{context} repeats a target domain")

    by_domain = _require(payload, "test_by_domain", context)
    if not isinstance(by_domain, dict):
        raise Stage3SingleSourceReportError(f"{context} test_by_domain must be an object")
    if set(by_domain) != {domain.value for domain in target_domains}:
        raise Stage3SingleSourceReportError(
            f"{context} test_by_domain does not cover exactly its target domains"
        )

    budget = _require(metadata, "budget", context)
    directory = metrics_path.parent
    per_image: dict[Domain, Path] = {}
    stored: dict[Domain, Mapping[str, Any]] = {}
    hd95_unit: dict[Domain, str] = {}
    for domain in target_domains:
        block = by_domain[domain.value]
        if not isinstance(block, dict):
            raise Stage3SingleSourceReportError(
                f"{context} test_by_domain.{domain.value} must be an object"
            )
        csv_path = directory / f"test_{domain.value}_per_image_metrics.csv"
        if not csv_path.is_file():
            raise Stage3SingleSourceReportError(
                f"{context} names {domain.value} but {csv_path.name} is missing"
            )
        unit = block.get("hd95_unit", HD95_UNIT_GRID)
        expected_unit = (
            HD95_UNIT_NATIVE if domain.value == NATIVE_HD95_DOMAIN else HD95_UNIT_GRID
        )
        if unit != expected_unit:
            raise Stage3SingleSourceReportError(
                f"{context} {domain.value} reports HD95 in {unit!r}, expected "
                f"{expected_unit!r}"
            )
        per_image[domain] = csv_path
        stored[domain] = block
        hd95_unit[domain] = unit

    identity = RunIdentity(
        arm=str(_require(metadata, "arm", context)),
        source_domain=source_domain,
        run_seed=int(_require(metadata, "run_seed", context)),
        manifest_sha256=_require_sha256(metadata, "manifest_sha256", context),
        parent_manifest_sha256=_require_sha256(
            metadata, "parent_manifest_sha256", context
        ),
        config_sha256=_require_sha256(metadata, "config_sha256", context),
        git_revision=str(metadata.get("git_revision", "unavailable")),
        completed_at_utc=str(_require(metadata, "completed_at_utc", context)),
        train_budget=int(_require(budget, "train", context)),
        val_budget=int(_require(budget, "val", context)),
        test_budget=(
            None if budget.get("test") is None else int(budget["test"])
        ),
    )
    return SingleSourceRun(
        identity=identity,
        directory=directory,
        metrics_path=metrics_path,
        target_domains=tuple(sorted(target_domains, key=lambda item: item.value)),
        per_image_csv=per_image,
        stored=stored,
        hd95_unit=hd95_unit,
    )


def discover_runs(roots: Iterable[str | Path]) -> tuple[SingleSourceRun, ...]:
    """Find every run of this arm beneath the given roots, sorted deterministically."""

    seen: dict[Path, SingleSourceRun] = {}
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise Stage3SingleSourceReportError(f"Run root is not a directory: {root_path}")
        for metrics_path in sorted(root_path.rglob(TEST_METRICS_NAME)):
            run = build_run(metrics_path)
            if run is not None:
                seen[metrics_path] = run
    return tuple(
        sorted(
            seen.values(),
            key=lambda run: (
                run.identity.source_domain.value,
                run.identity.run_seed,
                run.identity.completed_at_utc,
            ),
        )
    )


def select_scientific_runs(
    runs: Sequence[SingleSourceRun],
    expected_seeds: Sequence[int] = DEFAULT_EXPECTED_SEEDS,
) -> tuple[SingleSourceRun, ...]:
    """Keep one run per source/seed and prove the grid is complete.

    A preempted job that was relaunched leaves two directories for the same
    source and seed. The later completion is the real one; taking both would
    weight that cell twice.
    """

    by_cell: dict[tuple[str, int], SingleSourceRun] = {}
    for run in runs:
        key = (run.identity.source_domain.value, run.identity.run_seed)
        previous = by_cell.get(key)
        if (
            previous is None
            or run.identity.completed_at_utc > previous.identity.completed_at_utc
        ):
            by_cell[key] = run

    selected = tuple(sorted(by_cell.values(), key=lambda run: (
        run.identity.source_domain.value, run.identity.run_seed
    )))
    if not selected:
        raise Stage3SingleSourceReportError("No scientific runs of this arm were found")

    arms = {run.identity.arm for run in selected}
    if len(arms) != 1:
        raise Stage3SingleSourceReportError(
            f"Runs mix experimental arms: {sorted(arms)}"
        )
    for field in ("manifest_sha256", "parent_manifest_sha256"):
        digests = {getattr(run.identity, field) for run in selected}
        if len(digests) != 1:
            raise Stage3SingleSourceReportError(
                f"Runs disagree on {field}; they did not share one membership: "
                f"{sorted(digests)}"
            )
    budgets = {
        (run.identity.train_budget, run.identity.val_budget, run.identity.test_budget)
        for run in selected
    }
    if len(budgets) != 1:
        raise Stage3SingleSourceReportError(
            f"Runs disagree on the fixed budget: {sorted(budgets)}"
        )

    sources = sorted({run.identity.source_domain.value for run in selected})
    missing: list[str] = []
    for source in sources:
        present = {
            run.identity.run_seed
            for run in selected
            if run.identity.source_domain.value == source
        }
        for seed in expected_seeds:
            if seed not in present:
                missing.append(f"{source}/seed_{seed}")
    if missing:
        raise Stage3SingleSourceReportError(
            "Incomplete seed grid; these runs are missing: " + ", ".join(missing)
        )
    return selected


# --------------------------------------------------------------------------
# Stage B: verify each run against the manifest and its own files
# --------------------------------------------------------------------------


def verify_run_membership(
    run: SingleSourceRun,
    manifest: SingleSourceManifest,
) -> dict[Domain, int]:
    """Prove each target CSV scored exactly the manifest's budgeted test images."""

    fold = next(
        (
            fold
            for fold in manifest.folds
            if fold.source_domain == run.identity.source_domain
        ),
        None,
    )
    if fold is None:
        raise Stage3SingleSourceReportError(
            f"{run.label}: the manifest has no fold for this source domain"
        )
    if tuple(fold.target_domains) != run.target_domains:
        raise Stage3SingleSourceReportError(
            f"{run.label}: target domains {[d.value for d in run.target_domains]} do "
            f"not match the manifest's {[d.value for d in fold.target_domains]}"
        )

    counts: dict[Domain, int] = {}
    for domain in run.target_domains:
        expected = {sample.sample_id for sample in fold.test_samples(domain)}
        with run.per_image_csv[domain].open(newline="", encoding="utf-8") as stream:
            scored = {row["image_id"] for row in csv.DictReader(stream)}
        if scored != expected:
            extra = sorted(scored - expected)[:5]
            absent = sorted(expected - scored)[:5]
            raise Stage3SingleSourceReportError(
                f"{run.label} {domain.value}: scored images do not match the locked "
                f"test partition (unexpected={extra}, missing={absent})"
            )
        counts[domain] = len(expected)
    return counts


def _agrees(recomputed: object, stored: object) -> bool:
    if recomputed is None or stored is None:
        return recomputed is None and stored is None
    if isinstance(recomputed, (int, float)) and isinstance(stored, (int, float)):
        return math.isclose(
            float(recomputed),
            float(stored),
            rel_tol=SUMMARY_RELATIVE_TOLERANCE,
            abs_tol=SUMMARY_ABSOLUTE_TOLERANCE,
        )
    return recomputed == stored


def verify_run_summary(
    run: SingleSourceRun,
) -> dict[Domain, dict[str, dict[str, Any]]]:
    """Recompute every target summary from its CSV and require the stored one to match."""

    recomputed: dict[Domain, dict[str, dict[str, Any]]] = {}
    for domain in run.target_domains:
        summary = summarise_per_image_csv(run.per_image_csv[domain])
        if set(summary) != set(CHANNEL_NAMES):
            raise Stage3SingleSourceReportError(
                f"{run.label} {domain.value}: expected disc and cup rows, got "
                f"{sorted(summary)}"
            )
        stored_block = run.stored[domain]
        for structure in CHANNEL_NAMES:
            stored_structure = stored_block.get(structure)
            if not isinstance(stored_structure, dict):
                raise Stage3SingleSourceReportError(
                    f"{run.label} {domain.value}: no stored {structure} summary"
                )
            for field, value in summary[structure].items():
                if field not in stored_structure:
                    continue
                if not _agrees(value, stored_structure[field]):
                    raise Stage3SingleSourceReportError(
                        f"{run.label} {domain.value} {structure}.{field}: the CSV "
                        f"gives {value!r} but the report stored "
                        f"{stored_structure[field]!r}"
                    )
        recomputed[domain] = summary
    return recomputed


# --------------------------------------------------------------------------
# Stage C: aggregate over seeds
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedInterval:
    """Mean over per-seed run means, with the seed-level spread around it."""

    metric: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    std: float
    half_width: float
    low: float
    high: float
    confidence: float


def seed_confidence_interval(
    metric: str,
    seeds: Sequence[int],
    values: Sequence[float],
    confidence: float = CONFIDENCE_LEVEL,
) -> SeedInterval:
    """Mean +/- t(n-1, 0.975) * s/sqrt(n) over the per-seed run means.

    The spread is training-run wobble under a fixed, locked dataset -- not the
    image-to-image spread, which is far larger and is what each run's own
    ``dice_std`` reports.
    """

    if len(seeds) != len(values):
        raise Stage3SingleSourceReportError(
            f"{metric}: got {len(values)} values for {len(seeds)} seeds"
        )
    if len(values) < 2:
        raise Stage3SingleSourceReportError(
            f"{metric}: a confidence interval over seeds needs at least 2 seeds"
        )
    sample = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(sample)):
        raise Stage3SingleSourceReportError(
            f"{metric}: seed values must all be finite, got {list(values)}"
        )
    count = sample.size
    mean = float(np.mean(sample))
    # ddof=1: these seeds are a sample of the training procedure's outcomes,
    # not the whole population of them.
    std = float(np.std(sample, ddof=1))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, count - 1))
    half_width = critical * std / math.sqrt(count)
    return SeedInterval(
        metric=metric,
        seeds=tuple(seeds),
        values=tuple(float(value) for value in sample),
        mean=mean,
        std=std,
        half_width=half_width,
        low=mean - half_width,
        high=mean + half_width,
        confidence=confidence,
    )


@dataclass(frozen=True)
class CellReport:
    """One (source domain -> target domain, structure) cell across all seeds."""

    source_domain: Domain
    target_domain: Domain
    structure: str
    test_image_count: int
    hd95_unit: str
    intervals: Mapping[str, SeedInterval]
    hd95_excluded_total: int
    hd95_complete: bool


def build_cell_reports(
    runs: Sequence[SingleSourceRun],
    recomputed: Mapping[str, Mapping[Domain, dict[str, dict[str, Any]]]],
    counts: Mapping[str, Mapping[Domain, int]],
) -> tuple[CellReport, ...]:
    """Reduce every source/target/structure cell over its seeds."""

    cells: list[CellReport] = []
    sources = sorted(
        {run.identity.source_domain for run in runs}, key=lambda item: item.value
    )
    for source in sources:
        source_runs = sorted(
            (run for run in runs if run.identity.source_domain == source),
            key=lambda run: run.identity.run_seed,
        )
        seeds = [run.identity.run_seed for run in source_runs]
        for target in source_runs[0].target_domains:
            image_counts = {counts[run.label][target] for run in source_runs}
            if len(image_counts) != 1:
                raise Stage3SingleSourceReportError(
                    f"{source.value} -> {target.value}: seeds scored different "
                    f"numbers of images {sorted(image_counts)}"
                )
            units = {run.hd95_unit[target] for run in source_runs}
            if len(units) != 1:
                raise Stage3SingleSourceReportError(
                    f"{source.value} -> {target.value}: seeds mix HD95 units "
                    f"{sorted(units)}"
                )
            for structure in CHANNEL_NAMES:
                intervals: dict[str, SeedInterval] = {}
                excluded = 0
                complete = True
                for metric in SEED_METRICS:
                    values: list[float] = []
                    for run in source_runs:
                        summary = recomputed[run.label][target][structure]
                        value = summary[f"{metric}_mean"]
                        if value is None:
                            complete = False
                            break
                        values.append(float(value))
                    if len(values) == len(source_runs):
                        intervals[metric] = seed_confidence_interval(
                            metric, seeds, values
                        )
                for run in source_runs:
                    excluded += int(
                        recomputed[run.label][target][structure][
                            "hd95_excluded_count"
                        ]
                    )
                cells.append(
                    CellReport(
                        source_domain=source,
                        target_domain=target,
                        structure=structure,
                        test_image_count=next(iter(image_counts)),
                        hd95_unit=next(iter(units)),
                        intervals=intervals,
                        hd95_excluded_total=excluded,
                        hd95_complete=complete and excluded == 0,
                    )
                )
    return tuple(cells)


# --------------------------------------------------------------------------
# Stage D: render
# --------------------------------------------------------------------------


def _cell(interval: SeedInterval | None, digits: int = 4) -> str:
    if interval is None:
        return "n/a"
    return f"{interval.mean:.{digits}f} ± {interval.std:.{digits}f}"


def render_matrix(
    cells: Sequence[CellReport],
    structure: str,
    metric: str = "dice",
) -> str:
    """The headline table: every source domain against every unseen target."""

    sources = sorted({cell.source_domain for cell in cells}, key=lambda d: d.value)
    targets = sorted({cell.target_domain for cell in cells}, key=lambda d: d.value)
    lookup = {
        (cell.source_domain, cell.target_domain): cell
        for cell in cells
        if cell.structure == structure
    }
    lines = [
        f"| Trained on ↓ / tested on → | "
        + " | ".join(f"`{target.value}`" for target in targets)
        + " | Mean over targets |",
        "|---|" + "---:|" * (len(targets) + 1),
    ]
    for source in sources:
        row = [f"`{source.value}`"]
        means: list[float] = []
        for target in targets:
            cell = lookup.get((source, target))
            if cell is None:
                # The diagonal: a domain is never a target of itself, because its
                # test partition is excluded from its own fold entirely.
                row.append("—")
                continue
            interval = cell.intervals.get(metric)
            row.append(_cell(interval))
            if interval is not None:
                means.append(interval.mean)
        row.append(f"**{np.mean(means):.4f}**" if means else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_detail_table(cells: Sequence[CellReport]) -> str:
    lines = [
        "| Source | Target | Structure | Images | Dice, mean ± seed SD | 95% CI | "
        "IoU | HD95 | HD95 unit |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        dice = cell.intervals.get("dice")
        iou = cell.intervals.get("iou")
        hd95 = cell.intervals.get("hd95")
        unit = "Native px" if cell.hd95_unit == HD95_UNIT_NATIVE else "Grid px"
        ci = (
            f"[{dice.low:.4f}, {dice.high:.4f}]" if dice is not None else "n/a"
        )
        lines.append(
            f"| `{cell.source_domain.value}` | `{cell.target_domain.value}` | "
            f"{cell.structure} | {cell.test_image_count} | "
            f"**{_cell(dice)}** | {ci} | {_cell(iou)} | "
            f"{_cell(hd95, digits=2)} | {unit} |"
        )
    return "\n".join(lines)


def render_per_seed_table(
    cells: Sequence[CellReport],
    structure: str,
) -> str:
    relevant = [cell for cell in cells if cell.structure == structure]
    if not relevant:
        return "No cells to report."
    seeds = next(iter(relevant)).intervals.get("dice")
    if seeds is None:
        return "No per-seed Dice available."
    header = " | ".join(f"Seed {seed}" for seed in seeds.seeds)
    lines = [
        f"| Trained on | Tested on | {header} |",
        "|---|---|" + "---:|" * len(seeds.seeds),
    ]
    for cell in relevant:
        interval = cell.intervals.get("dice")
        if interval is None:
            continue
        values = " | ".join(f"{value:.4f}" for value in interval.values)
        lines.append(
            f"| `{cell.source_domain.value}` | `{cell.target_domain.value}` | "
            f"{values} |"
        )
    return "\n".join(lines)


def render_markdown_report(
    cells: Sequence[CellReport],
    runs: Sequence[SingleSourceRun],
    manifest: SingleSourceManifest,
    manifest_path: Path,
) -> str:
    """A report skeleton in the run_reports house style.

    Every figure is computed here. Every section that requires judgement is left
    as an explicit TODO rather than invented, because a generated sentence about
    what a result means is exactly the kind of claim that must not reach a thesis
    unexamined.
    """

    identity = runs[0].identity
    seeds = sorted({run.identity.run_seed for run in runs})
    sources = sorted({run.identity.source_domain.value for run in runs})
    lines: list[str] = []
    add = lines.append

    add("# Stage 3 train-on-one, test-on-three report: plain U-Net")
    add("")
    add(
        "**Evidence boundary.** Every figure below is computed by "
        "`aggregate_stage3_1_3.py` directly from the per-image metric CSVs of "
        f"{len(runs)} completed runs, validated against the locked budgeted "
        "manifest, and recomputed from those CSVs rather than read from any "
        "run's stored summary. The pooled `test_pooled` block in each run is "
        "deliberately ignored: it averages three acquisition domains and is not "
        "a per-domain result. Sections marked TODO require judgement the tool "
        "does not make."
    )
    add("")

    add("## 1. Protocol")
    add("")
    add(
        "Each run trains on a single acquisition domain under a fixed labelled "
        f"budget of **{identity.train_budget} training** and "
        f"**{identity.val_budget} validation** images drawn from that domain's "
        "own locked partitions, then scores every other domain separately on "
        f"**{identity.test_budget} held-out test images** each. The source "
        "domain's own test partition is excluded, so no image ever changes role."
    )
    add("")
    add(
        "The budget is the Drishti-GS floor: it has both the fewest training "
        "and the fewest test images, so capping every domain to it makes the "
        "cells comparable. Without the cap, pooled training volume would swing "
        "with which domain was used and confound the comparison."
    )
    add("")
    add(f"| Setting | Value |")
    add("|---|---|")
    add(f"| Arm | `{identity.arm}` |")
    add(f"| Source domains | {', '.join(f'`{name}`' for name in sources)} |")
    add(f"| Seeds | {', '.join(str(seed) for seed in seeds)} |")
    add(f"| Runs | {len(runs)} |")
    add(f"| Train / val / test budget | {identity.train_budget} / {identity.val_budget} / {identity.test_budget} |")
    add(f"| Checkpoint selection | lowest validation loss on the source domain |")
    add(f"| Manifest | `{manifest_path.name}` (`{identity.manifest_sha256[:12]}…`) |")
    add(f"| Parent LODO manifest | `{identity.parent_manifest_sha256[:12]}…` |")
    add(f"| Git revision | `{identity.git_revision}` |")
    add("")

    add("## 2. Cross-domain Dice matrix")
    add("")
    add(
        "Rows are the single training domain; columns are the unseen target "
        "domain. Each cell is the mean over the per-seed run means, plus or "
        "minus the seed-level standard deviation. The diagonal is empty because "
        "a domain is never a target of its own fold."
    )
    add("")
    add("### 2.1 Optic disc")
    add("")
    add(render_matrix(cells, "disc"))
    add("")
    add("### 2.2 Optic cup")
    add("")
    add(render_matrix(cells, "cup"))
    add("")

    add("## 3. Full per-cell results")
    add("")
    add(
        "Dice and IoU are unitless. HD95 is in letterboxed-grid pixels for every "
        "target except RIM-ONE-DL, which is in native source pixels; the two are "
        "never averaged together."
    )
    add("")
    add(render_detail_table(cells))
    add("")

    add("## 4. Per-seed disc Dice")
    add("")
    add(render_per_seed_table(cells, "disc"))
    add("")

    incomplete = [cell for cell in cells if not cell.hd95_complete]
    if incomplete:
        add("## 5. HD95 completeness")
        add("")
        add(
            "These cells had at least one image with an undefined HD95, which "
            "happens when a prediction or a target is empty. Their HD95 means "
            "are taken over the finite subset only and are not comparable with "
            "cells where every image was finite."
        )
        add("")
        add("| Source | Target | Structure | Excluded images (all seeds) |")
        add("|---|---|---|---:|")
        for cell in incomplete:
            add(
                f"| `{cell.source_domain.value}` | `{cell.target_domain.value}` | "
                f"{cell.structure} | {cell.hd95_excluded_total} |"
            )
        add("")

    add("## 6. Findings")
    add("")
    add(TODO)
    add("")
    add("## 7. Comparison against the literature")
    add("")
    add(TODO)
    add("")
    add("## 8. Limitations")
    add("")
    add(
        f"- Each cell rests on {identity.test_budget} test images and "
        f"{len(seeds)} seeds; the confidence intervals are over seeds, not images."
    )
    add(
        f"- Training uses only {identity.train_budget} images, far below what "
        "either dataset's own baseline used, so absolute Dice is not comparable "
        "with the Stage 2 in-domain numbers."
    )
    add(
        "- RIM-ONE-DL HD95 is in native source pixels while every other target "
        "is in letterboxed-grid pixels."
    )
    add(TODO)
    add("")
    return "\n".join(lines)


def write_csv(cells: Sequence[CellReport], path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        row: dict[str, Any] = {
            "source_domain": cell.source_domain.value,
            "target_domain": cell.target_domain.value,
            "structure": cell.structure,
            "test_images": cell.test_image_count,
            "hd95_unit": cell.hd95_unit,
            "hd95_excluded_total": cell.hd95_excluded_total,
        }
        for metric in SEED_METRICS:
            interval = cell.intervals.get(metric)
            if interval is None:
                row.update(
                    {
                        f"{metric}_mean": "",
                        f"{metric}_seed_std": "",
                        f"{metric}_ci_low": "",
                        f"{metric}_ci_high": "",
                    }
                )
                continue
            row.update(
                {
                    f"{metric}_mean": f"{interval.mean:.6g}",
                    f"{metric}_seed_std": f"{interval.std:.6g}",
                    f"{metric}_ci_low": f"{interval.low:.6g}",
                    f"{metric}_ci_high": f"{interval.high:.6g}",
                }
            )
            for seed, value in zip(interval.seeds, interval.values):
                row[f"{metric}_seed_{seed}"] = f"{value:.6g}"
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Stage 3 train-on-one runs into a per-target-domain report"
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="Directory to search for run outputs (repeatable)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Locked budgeted manifest the runs must agree with",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPECTED_SEEDS),
        help="Seeds every source domain must have completed",
    )
    parser.add_argument("--report-out", type=Path, help="Write the markdown report here")
    parser.add_argument("--csv-out", type=Path, help="Write the per-cell table here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.run_root or list(DEFAULT_RUN_ROOTS)
    try:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_single_source_manifest(manifest_path)
        runs = select_scientific_runs(
            discover_runs(roots), tuple(args.expected_seeds)
        )

        manifest_digest = _sha256(manifest_path)
        if runs[0].identity.manifest_sha256 != manifest_digest:
            raise Stage3SingleSourceReportError(
                "The runs were produced against a different manifest than "
                f"{manifest_path}: runs say "
                f"{runs[0].identity.manifest_sha256[:12]}…, this file is "
                f"{manifest_digest[:12]}…"
            )

        counts: dict[str, Mapping[Domain, int]] = {}
        recomputed: dict[str, Mapping[Domain, dict[str, dict[str, Any]]]] = {}
        for run in runs:
            counts[run.label] = verify_run_membership(run, manifest)
            recomputed[run.label] = verify_run_summary(run)
        cells = build_cell_reports(runs, recomputed, counts)
    except (Stage3SingleSourceReportError, OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    print(f"aggregated {len(runs)} runs into {len(cells)} source/target/structure cells")
    print()
    print("cross-domain disc Dice (rows = trained on, columns = tested on):")
    print(render_matrix(cells, "disc"))
    print()

    if args.report_out is not None:
        report = render_markdown_report(cells, runs, manifest, manifest_path)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {args.report_out}")
    if args.csv_out is not None:
        print(f"wrote {write_csv(cells, args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
