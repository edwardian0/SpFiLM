#!/usr/bin/env python3
"""Aggregate the fixed-budget LODO arm and test it against the train-on-one arm.

The project brief specifies the statistics this produces (Section 5): five seeds
per version, mean and 95% confidence interval, a **paired significance test on
per-image Dice over the same test images**, and Dice, IoU and a boundary error
reported for disc and cup separately, per held-out domain rather than only
averaged.

The brief frames that paired test as Global FiLM against SpFiLM. Those arms do
not exist yet. What does exist is a pair that satisfies the same requirement
exactly -- identical backbone, identical folds, identical hyperparameters,
identical test images -- differing in one controlled variable, training volume:

    pooled_120   train on three domains' budgeted 40s = 120 images
    single_40    train on one domain's budgeted 40 images

so the same machinery answers "does pooling three domains beat one well-matched
domain?" on the same footing the conditioning comparison will later use. This is
**not** the brief's headline comparison and does not substitute for it.

Pairing is only valid on identical image sets, so that is asserted here rather
than trusted; the shared budgeted manifest is what makes it true.
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
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from aggregate_stage3_1_3 import (  # noqa: E402
    SeedInterval,
    Stage3SingleSourceReportError,
    discover_runs as discover_single_source_runs,
    seed_confidence_interval,
    select_scientific_runs as select_single_source_runs,
)
from run_stage3_lodo_3_1_fixed import fixed_lodo_folds  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.metrics import CHANNEL_NAMES, read_per_image_csv  # noqa: E402
from spfilm.single_source import load_single_source_manifest  # noqa: E402


TEST_METRICS_NAME = "test_metrics.json"
FIXED_PER_IMAGE_CSV = "test_per_image_metrics.csv"
FIXED_PROTOCOL = "leave_one_domain_out_fixed_budget"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "splits" / "single_source" / "single_source_manifest.json"
)
DEFAULT_RUN_ROOTS = (PROJECT_ROOT / "artifacts",)
DEFAULT_EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CONFIDENCE_LEVEL = 0.95
POOLED_ARM = "pooled_120"
PAIRED_METHODS = ("wilcoxon", "ttest")
ALPHA = 0.05
TODO = "<!-- TODO: written by hand; the tool does not infer this. -->"


class FixedLodoReportError(ValueError):
    """Raised when the discovered runs cannot support an honest report."""


# --------------------------------------------------------------------------
# Stage A: load the fixed-budget runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedRun:
    """One completed fixed-budget LODO run."""

    arm: str
    held_out_domain: Domain
    run_seed: int
    source_domains: tuple[str, ...]
    manifest_sha256: str
    completed_at_utc: str
    train_budget: int
    val_budget: int
    test_budget: int | None
    directory: Path
    metrics_path: Path
    per_image_csv: Path
    stored: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"{self.held_out_domain.value}/seed_{self.run_seed}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedLodoReportError(f"Cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FixedLodoReportError(f"{path} must hold a JSON object")
    return payload


def build_fixed_run(metrics_path: Path) -> FixedRun | None:
    payload = _read_json(metrics_path)
    metadata = payload.get("fixed_lodo")
    if not isinstance(metadata, dict):
        return None
    if metadata.get("protocol") != FIXED_PROTOCOL:
        return None
    if metadata.get("smoke_rehearsal") is True:
        return None
    context = str(metrics_path)
    if metadata.get("scientific_result") is not True:
        raise FixedLodoReportError(
            f"{context} is not marked scientific_result; refusing to report it"
        )
    test = payload.get("test")
    if not isinstance(test, dict):
        raise FixedLodoReportError(f"{context} has no test block")
    csv_path = metrics_path.parent / FIXED_PER_IMAGE_CSV
    if not csv_path.is_file():
        raise FixedLodoReportError(f"{context} is missing {FIXED_PER_IMAGE_CSV}")
    budget = metadata.get("budget") or {}
    return FixedRun(
        arm=str(metadata["arm"]),
        held_out_domain=Domain(metadata["held_out_domain"]),
        run_seed=int(metadata["run_seed"]),
        source_domains=tuple(metadata.get("source_domains", ())),
        manifest_sha256=str(metadata["manifest_sha256"]),
        completed_at_utc=str(metadata["completed_at_utc"]),
        train_budget=int(budget.get("train", 0)),
        val_budget=int(budget.get("val", 0)),
        test_budget=(
            None if budget.get("test") is None else int(budget["test"])
        ),
        directory=metrics_path.parent,
        metrics_path=metrics_path,
        per_image_csv=csv_path,
        stored=test,
    )


def discover_fixed_runs(roots: Iterable[str | Path]) -> tuple[FixedRun, ...]:
    found: dict[Path, FixedRun] = {}
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FixedLodoReportError(f"Run root is not a directory: {root_path}")
        for metrics_path in sorted(root_path.rglob(TEST_METRICS_NAME)):
            run = build_fixed_run(metrics_path)
            if run is not None:
                found[metrics_path] = run
    return tuple(
        sorted(
            found.values(),
            key=lambda run: (
                run.held_out_domain.value,
                run.run_seed,
                run.completed_at_utc,
            ),
        )
    )


def select_fixed_runs(
    runs: Sequence[FixedRun],
    expected_seeds: Sequence[int] = DEFAULT_EXPECTED_SEEDS,
    arm: str | None = None,
) -> tuple[FixedRun, ...]:
    """Keep one run per domain/seed and prove the grid is complete.

    ``arm`` restricts the selection to one experiment when a run root holds
    several (the plain fixed-budget arm and the Global FiLM arm share
    ``artifacts/runs``); without it the runs must already be a single arm.
    """

    if arm is not None:
        runs = [run for run in runs if run.arm == arm]
        if not runs:
            raise FixedLodoReportError(f"No scientific runs found for arm {arm!r}")
    # Check the arms before collapsing to one run per cell: two arms share every
    # (domain, seed) cell, so a later dedup would silently keep whichever
    # finished last and report a mixture as one arm.
    arms = {run.arm for run in runs}
    if len(arms) > 1:
        raise FixedLodoReportError(
            f"Runs mix experimental arms: {sorted(arms)}; pass --arm to pick one"
        )
    by_cell: dict[tuple[str, int], FixedRun] = {}
    for run in runs:
        key = (run.held_out_domain.value, run.run_seed)
        previous = by_cell.get(key)
        if previous is None or run.completed_at_utc > previous.completed_at_utc:
            by_cell[key] = run
    selected = tuple(
        sorted(by_cell.values(), key=lambda run: (run.held_out_domain.value, run.run_seed))
    )
    if not selected:
        raise FixedLodoReportError("No scientific fixed-budget LODO runs were found")

    digests = {run.manifest_sha256 for run in selected}
    if len(digests) != 1:
        raise FixedLodoReportError(
            f"Runs disagree on manifest_sha256: {sorted(digests)}"
        )
    budgets = {(r.train_budget, r.val_budget, r.test_budget) for r in selected}
    if len(budgets) != 1:
        raise FixedLodoReportError(f"Runs disagree on the fixed budget: {sorted(budgets)}")

    missing = [
        f"{domain}/seed_{seed}"
        for domain in sorted({run.held_out_domain.value for run in selected})
        for seed in expected_seeds
        if seed not in {
            run.run_seed for run in selected if run.held_out_domain.value == domain
        }
    ]
    if missing:
        raise FixedLodoReportError(
            "Incomplete seed grid; these runs are missing: " + ", ".join(missing)
        )
    return selected


# --------------------------------------------------------------------------
# Stage B: per-image scores, seeds averaged, ready for pairing
# --------------------------------------------------------------------------


def _per_image_rows(csv_path: Path) -> list[dict[str, object]]:
    return read_per_image_csv(csv_path)


def _accumulate(
    store: dict[tuple[str, Domain, str, str], dict[str, list[float]]],
    seeds: dict[tuple[str, Domain], set[int]],
    arm: str,
    domain: Domain,
    seed: int,
    csv_path: Path,
) -> None:
    seeds.setdefault((arm, domain), set()).add(seed)
    for row in _per_image_rows(csv_path):
        structure = str(row["structure"])
        if structure not in CHANNEL_NAMES:
            raise FixedLodoReportError(
                f"{csv_path}: unexpected structure {structure!r}"
            )
        key = (arm, domain, structure, str(row["image_id"]))
        bucket = store.setdefault(key, {"dice": [], "iou": []})
        bucket["dice"].append(float(row["dice"]))
        bucket["iou"].append(float(row["iou"]))


@dataclass(frozen=True)
class Substrate:
    """One value per image per arm, with the seeds already averaged away.

    Averaging the seeds first is what makes the pairs real: pooling all
    5 x 50 seed-image scores would count each image five times and treat five
    correlated numbers as independent evidence.

    HD95 is deliberately absent. Its per-run figures are not taken over a common
    image set, because degenerate cases are excluded and the excluded set moves
    from seed to seed. Dice and IoU are defined for every image in every run,
    which is why the brief specifies the paired test on Dice.
    """

    seed_counts: Mapping[tuple[str, Domain], int]
    values: Mapping[tuple[str, Domain, str, str], Mapping[str, float]]

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted({key[0] for key in self.values}))

    def image_ids(self, arm: str, domain: Domain, structure: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                key[3]
                for key in self.values
                if key[0] == arm and key[1] == domain and key[2] == structure
            )
        )

    def series(
        self, arm: str, domain: Domain, structure: str, image_ids: Sequence[str]
    ) -> np.ndarray:
        return np.asarray(
            [self.values[(arm, domain, structure, i)]["dice"] for i in image_ids],
            dtype=float,
        )


def build_substrate(
    fixed_runs: Sequence[FixedRun],
    single_runs: Sequence[Any],
) -> Substrate:
    """Key the pooled arm and each single-source arm by held-out/target domain."""

    store: dict[tuple[str, Domain, str, str], dict[str, list[float]]] = {}
    seeds: dict[tuple[str, Domain], set[int]] = {}

    for run in fixed_runs:
        _accumulate(
            store, seeds, POOLED_ARM, run.held_out_domain, run.run_seed, run.per_image_csv
        )
    for run in single_runs:
        arm = f"single_{run.identity.source_domain.value}"
        for domain in run.target_domains:
            _accumulate(
                store,
                seeds,
                arm,
                domain,
                run.identity.run_seed,
                run.per_image_csv[domain],
            )

    values: dict[tuple[str, Domain, str, str], Mapping[str, float]] = {}
    for key, bucket in store.items():
        expected = len(seeds[(key[0], key[1])])
        for metric, samples in bucket.items():
            if len(samples) != expected:
                raise FixedLodoReportError(
                    f"{key[0]}/{key[1].value} {key[2]} image {key[3]!r} has "
                    f"{len(samples)} {metric} values but the arm ran {expected} seeds"
                )
        values[key] = {
            metric: float(np.mean(samples)) for metric, samples in bucket.items()
        }
    return Substrate(
        seed_counts={cell: len(v) for cell, v in seeds.items()}, values=values
    )


# --------------------------------------------------------------------------
# Stage C: per-domain summaries for the fixed arm
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainCell:
    held_out_domain: Domain
    structure: str
    test_image_count: int
    hd95_unit: str
    intervals: Mapping[str, SeedInterval]
    hd95_excluded_total: int


def build_domain_cells(runs: Sequence[FixedRun]) -> tuple[DomainCell, ...]:
    cells: list[DomainCell] = []
    for domain in sorted({r.held_out_domain for r in runs}, key=lambda d: d.value):
        domain_runs = sorted(
            (r for r in runs if r.held_out_domain == domain),
            key=lambda r: r.run_seed,
        )
        seeds = [r.run_seed for r in domain_runs]
        for structure in CHANNEL_NAMES:
            summaries = []
            for run in domain_runs:
                rows = [
                    r for r in _per_image_rows(run.per_image_csv)
                    if r["structure"] == structure
                ]
                if not rows:
                    raise FixedLodoReportError(
                        f"{run.label}: no {structure} rows in {run.per_image_csv.name}"
                    )
                dice = np.asarray([float(r["dice"]) for r in rows])
                iou = np.asarray([float(r["iou"]) for r in rows])
                hd = np.asarray([float(r["hd95"]) for r in rows])
                finite = hd[np.isfinite(hd)]
                summaries.append(
                    {
                        "dice": float(dice.mean()),
                        "iou": float(iou.mean()),
                        "hd95": float(finite.mean()) if finite.size else None,
                        "n": len(rows),
                        "excluded": int(hd.size - finite.size),
                    }
                )
            counts = {s["n"] for s in summaries}
            if len(counts) != 1:
                raise FixedLodoReportError(
                    f"{domain.value} {structure}: seeds scored different image counts "
                    f"{sorted(counts)}"
                )
            intervals: dict[str, SeedInterval] = {}
            for metric in ("dice", "iou", "hd95"):
                series = [s[metric] for s in summaries]
                if any(v is None for v in series):
                    continue
                intervals[metric] = seed_confidence_interval(metric, seeds, series)
            cells.append(
                DomainCell(
                    held_out_domain=domain,
                    structure=structure,
                    test_image_count=next(iter(counts)),
                    hd95_unit=str(
                        domain_runs[0].stored.get(
                            "hd95_unit", "letterboxed-grid pixels"
                        )
                    ),
                    intervals=intervals,
                    hd95_excluded_total=sum(s["excluded"] for s in summaries),
                )
            )
    return tuple(cells)


# --------------------------------------------------------------------------
# Stage D: the paired significance test the brief requires
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedResult:
    held_out_domain: Domain
    structure: str
    arm_a: str
    arm_b: str
    image_count: int
    mean_a: float
    mean_b: float
    mean_difference: float
    median_difference: float
    statistic: float
    p_value: float
    p_adjusted: float
    method: str
    significant: bool


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm-Bonferroni step-down adjustment, preserving input order.

    Four held-out domains times three single sources times two structures is 24
    tests. At alpha=0.05 that is a ~71% chance of at least one false positive if
    every null is true, so an uncorrected cell means very little on its own.
    """

    count = len(p_values)
    if count == 0:
        return ()
    for value in p_values:
        if not 0.0 <= float(value) <= 1.0:
            raise FixedLodoReportError(f"p-value out of range: {value!r}")
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * float(p_values[index])
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    return tuple(adjusted)


def paired_tests(
    substrate: Substrate,
    method: str = "wilcoxon",
    reference_arm: str = POOLED_ARM,
) -> tuple[PairedResult, ...]:
    """Test ``reference_arm`` minus every other arm, per domain and structure.

    By default the reference is the pooled 120-image arm and the others are the
    single-source arms. The Step 4 comparison reuses this with Global FiLM as
    the reference and the plain fixed-budget arm as the other, so a positive
    difference always reads "the reference arm helped".

    Wilcoxon signed-rank is the default because per-image Dice is bounded in
    [0, 1] and typically left-skewed with a clump at the ceiling, so the paired
    differences are often not normal. Choose ``ttest`` only after inspecting the
    differences.
    """

    if method not in PAIRED_METHODS:
        raise FixedLodoReportError(
            f"Unknown paired method {method!r}; expected one of {list(PAIRED_METHODS)}"
        )
    if reference_arm not in substrate.arms:
        raise FixedLodoReportError(
            f"{reference_arm} is absent; there is nothing to compare against"
        )

    raw: list[dict[str, Any]] = []
    domains = sorted(
        {key[1] for key in substrate.values if key[0] == reference_arm},
        key=lambda d: d.value,
    )
    for domain in domains:
        single_arms = sorted(
            {
                key[0]
                for key in substrate.values
                if key[1] == domain and key[0] != reference_arm
            }
        )
        for structure in CHANNEL_NAMES:
            pooled_ids = substrate.image_ids(reference_arm, domain, structure)
            if not pooled_ids:
                # No scores for this structure. Emitting a zero-image result with
                # p = 1 would pad the Holm family and make every real test's
                # correction more conservative than the evidence warrants.
                continue
            for arm in single_arms:
                other_ids = substrate.image_ids(arm, domain, structure)
                if pooled_ids != other_ids:
                    raise FixedLodoReportError(
                        f"{domain.value} {structure}: {reference_arm} scored "
                        f"{len(pooled_ids)} images and {arm} scored "
                        f"{len(other_ids)}; a paired test needs identical images"
                    )
                pooled_seeds = substrate.seed_counts[(reference_arm, domain)]
                other_seeds = substrate.seed_counts[(arm, domain)]
                if pooled_seeds != other_seeds:
                    raise FixedLodoReportError(
                        f"{domain.value}: {reference_arm} averaged {pooled_seeds} seeds "
                        f"and {arm} averaged {other_seeds}. A five-seed mean and a "
                        "three-seed mean are not the same estimator"
                    )
                a = substrate.series(arm, domain, structure, pooled_ids)
                b = substrate.series(reference_arm, domain, structure, pooled_ids)
                difference = b - a
                if np.allclose(difference, 0.0):
                    statistic, p_value = 0.0, 1.0
                elif method == "wilcoxon":
                    result = stats.wilcoxon(b, a, zero_method="wilcox")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                else:
                    result = stats.ttest_rel(b, a)
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                raw.append(
                    {
                        "domain": domain,
                        "structure": structure,
                        "arm_a": arm,
                        "arm_b": reference_arm,
                        "n": len(pooled_ids),
                        "mean_a": float(a.mean()),
                        "mean_b": float(b.mean()),
                        "mean_difference": float(difference.mean()),
                        "median_difference": float(np.median(difference)),
                        "statistic": statistic,
                        "p_value": p_value,
                    }
                )

    adjusted = holm_adjust([row["p_value"] for row in raw])
    return tuple(
        PairedResult(
            held_out_domain=row["domain"],
            structure=row["structure"],
            arm_a=row["arm_a"],
            arm_b=row["arm_b"],
            image_count=row["n"],
            mean_a=row["mean_a"],
            mean_b=row["mean_b"],
            mean_difference=row["mean_difference"],
            median_difference=row["median_difference"],
            statistic=row["statistic"],
            p_value=row["p_value"],
            p_adjusted=p_adj,
            method=method,
            significant=p_adj < ALPHA,
        )
        for row, p_adj in zip(raw, adjusted)
    )


# --------------------------------------------------------------------------
# Stage E: render
# --------------------------------------------------------------------------


def _interval(interval: SeedInterval | None, digits: int = 4) -> str:
    if interval is None:
        return "n/a"
    return f"{interval.mean:.{digits}f} ± {interval.std:.{digits}f}"


def render_domain_table(cells: Sequence[DomainCell]) -> str:
    lines = [
        "| Held-out domain | Structure | Images | Dice, mean ± seed SD | 95% CI | "
        "IoU | HD95 | HD95 unit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        dice = cell.intervals.get("dice")
        ci = f"[{dice.low:.4f}, {dice.high:.4f}]" if dice else "n/a"
        unit = "Native px" if "native" in cell.hd95_unit else "Grid px"
        lines.append(
            f"| `{cell.held_out_domain.value}` | {cell.structure} | "
            f"{cell.test_image_count} | **{_interval(dice)}** | {ci} | "
            f"{_interval(cell.intervals.get('iou'))} | "
            f"{_interval(cell.intervals.get('hd95'), 2)} | {unit} |"
        )
    return "\n".join(lines)


def render_paired_table(results: Sequence[PairedResult]) -> str:
    lines = [
        "| Held-out domain | Structure | Single source (40) | 120 Dice | 40 Dice | "
        "Δ (120−40) | p | p (Holm) | Significant |",
        "|---|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in sorted(
        results, key=lambda x: (x.held_out_domain.value, x.structure, x.arm_a)
    ):
        source = r.arm_a.removeprefix("single_")
        mark = "**yes**" if r.significant else "no"
        lines.append(
            f"| `{r.held_out_domain.value}` | {r.structure} | `{source}` | "
            f"{r.mean_b:.4f} | {r.mean_a:.4f} | {r.mean_difference:+.4f} | "
            f"{r.p_value:.4g} | {r.p_adjusted:.4g} | {mark} |"
        )
    return "\n".join(lines)


def render_markdown_report(
    cells: Sequence[DomainCell],
    results: Sequence[PairedResult],
    fixed_runs: Sequence[FixedRun],
    single_runs: Sequence[Any],
    manifest_path: Path,
) -> str:
    run = fixed_runs[0]
    seeds = sorted({r.run_seed for r in fixed_runs})
    lines: list[str] = []
    add = lines.append

    add("# Stage 3 fixed-budget leave-one-domain-out: plain U-Net")
    add("")
    add(
        "**Evidence boundary.** Every figure below is computed by "
        "`aggregate_stage3_fixed.py` directly from the per-image metric CSVs of "
        f"{len(fixed_runs)} fixed-budget runs and {len(single_runs)} "
        "train-on-one runs, validated against the shared budgeted manifest, and "
        "recomputed from those CSVs rather than read from any stored summary. "
        "Sections marked TODO require judgement the tool does not make."
    )
    add("")

    add("## 1. Protocol")
    add("")
    add(
        "Leave-one-domain-out under a fixed labelled budget. For each held-out "
        f"domain the model trains on the other three domains' budgeted "
        f"partitions — **{run.train_budget} images from each, {run.train_budget * 3} "
        f"in total** — validates on their pooled **{run.val_budget * 3}**, and is "
        f"scored on the held-out domain's **{run.test_budget}** locked test "
        "images with no adaptation."
    )
    add("")
    add(
        "The budget is the Drishti-GS floor. Capping every domain to it removes "
        "the confound in the original full-data arm, whose pooled training set "
        "swung between 552 and 852 images depending on which domain was dropped, "
        "and whose test sets ranged from 51 to 97 images."
    )
    add("")
    add("| Setting | Value |")
    add("|---|---|")
    add(f"| Arm | `{run.arm}` |")
    add(f"| Seeds | {', '.join(str(s) for s in seeds)} |")
    add(f"| Runs | {len(fixed_runs)} |")
    add(
        f"| Train / val / test | {run.train_budget * 3} / {run.val_budget * 3} / "
        f"{run.test_budget} |"
    )
    add("| Checkpoint selection | lowest pooled source validation loss |")
    add(f"| Manifest | `{manifest_path.name}` (`{run.manifest_sha256[:12]}…`) |")
    add("")

    add("## 2. Held-out results per domain")
    add("")
    add(
        "Means over the per-seed run means, with the seed-level standard "
        "deviation and a 95% confidence interval. Dice and IoU are unitless. "
        "HD95 is in letterboxed-grid pixels except for RIM-ONE-DL, which is in "
        "native source pixels; the two are never averaged together. Disc and cup "
        "are separate throughout."
    )
    add("")
    add(render_domain_table(cells))
    add("")

    add("## 3. Paired test: 120 training images against 40")
    add("")
    add(
        "The brief requires a paired significance test on per-image Dice over "
        "the same test images. Both arms draw from the same budgeted manifest, "
        "so for each held-out domain they score the identical "
        f"{run.test_budget} images; that is asserted before each test rather "
        "than assumed. Each image's five seed scores are averaged first, so the "
        "pairs are one value per image rather than five correlated ones."
    )
    add("")
    add(
        "Each row compares the pooled 120-image model against one single-source "
        "40-image model on the same held-out domain. A positive Δ means pooling "
        "three domains helped. p-values are Wilcoxon signed-rank, adjusted "
        f"across all {len(results)} tests by Holm-Bonferroni; "
        f"significance is at α = {ALPHA} on the adjusted value."
    )
    add("")
    add(render_paired_table(results))
    add("")
    wins = sum(1 for r in results if r.significant and r.mean_difference > 0)
    losses = sum(1 for r in results if r.significant and r.mean_difference < 0)
    add(
        f"**Counted outcome.** Of {len(results)} paired comparisons, "
        f"{wins} favour the pooled 120-image model at Holm-adjusted "
        f"α = {ALPHA}, {losses} favour the single 40-image model, and "
        f"{len(results) - wins - losses} are not separable."
    )
    add("")

    add("## 4. Findings")
    add("")
    add(TODO)
    add("")
    add("## 5. Limitations")
    add("")
    add(
        f"- This is not the brief's headline comparison. The brief specifies the "
        "paired test between Global FiLM and SpFiLM; neither conditioning arm "
        "has been run. This uses the same statistical protocol on the one "
        "controlled pair that exists, training volume."
    )
    add(
        f"- Each cell rests on {run.test_budget} test images and {len(seeds)} "
        "seeds; the confidence intervals are over seeds, not images."
    )
    add(
        "- The paired test is on Dice only. HD95 is excluded from pairing "
        "because its degenerate-case exclusions move from seed to seed, so it is "
        "not defined over a common image set."
    )
    add(
        "- Training volume and domain diversity are confounded with each other: "
        "the 120-image arm sees three domains, the 40-image arm sees one. This "
        "design cannot separate 'more data' from 'more domains'."
    )
    add(TODO)
    add("")
    return "\n".join(lines)


def write_csv(
    cells: Sequence[DomainCell], results: Sequence[PairedResult], path: Path
) -> Path:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        row: dict[str, Any] = {
            "kind": "domain_summary",
            "held_out_domain": cell.held_out_domain.value,
            "structure": cell.structure,
            "test_images": cell.test_image_count,
            "hd95_unit": cell.hd95_unit,
            "hd95_excluded_total": cell.hd95_excluded_total,
        }
        for metric in ("dice", "iou", "hd95"):
            interval = cell.intervals.get(metric)
            row[f"{metric}_mean"] = "" if interval is None else f"{interval.mean:.6g}"
            row[f"{metric}_seed_std"] = "" if interval is None else f"{interval.std:.6g}"
            row[f"{metric}_ci_low"] = "" if interval is None else f"{interval.low:.6g}"
            row[f"{metric}_ci_high"] = "" if interval is None else f"{interval.high:.6g}"
        rows.append(row)
    for r in results:
        rows.append(
            {
                "kind": "paired_test",
                "held_out_domain": r.held_out_domain.value,
                "structure": r.structure,
                "arm_a": r.arm_a,
                "arm_b": r.arm_b,
                "test_images": r.image_count,
                "dice_mean_a": f"{r.mean_a:.6g}",
                "dice_mean_b": f"{r.mean_b:.6g}",
                "mean_difference": f"{r.mean_difference:.6g}",
                "median_difference": f"{r.median_difference:.6g}",
                "method": r.method,
                "statistic": f"{r.statistic:.6g}",
                "p_value": f"{r.p_value:.6g}",
                "p_adjusted": f"{r.p_adjusted:.6g}",
                "significant": r.significant,
            }
        )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
            "Aggregate the fixed-budget LODO arm and run the brief's paired "
            "significance test against the train-on-one arm"
        )
    )
    parser.add_argument("--run-root", type=Path, action="append")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--expected-seeds", type=int, nargs="+", default=list(DEFAULT_EXPECTED_SEEDS)
    )
    parser.add_argument(
        "--method",
        choices=PAIRED_METHODS,
        default="wilcoxon",
        help="Paired test; Wilcoxon signed-rank unless the differences look normal",
    )
    parser.add_argument(
        "--skip-paired",
        action="store_true",
        help="Summarise the fixed arm only, without the train-on-one comparison",
    )
    parser.add_argument(
        "--arm",
        help=(
            "Experiment name of the fixed-budget arm to summarise when the run "
            "roots hold more than one (e.g. the plain and Global FiLM arms)"
        ),
    )
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.run_root or list(DEFAULT_RUN_ROOTS)
    try:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_single_source_manifest(manifest_path)
        fixed = select_fixed_runs(
            discover_fixed_runs(roots), tuple(args.expected_seeds), arm=args.arm
        )
        digest = _sha256(manifest_path)
        if fixed[0].manifest_sha256 != digest:
            raise FixedLodoReportError(
                "The runs were produced against a different manifest than "
                f"{manifest_path}: runs say {fixed[0].manifest_sha256[:12]}…, "
                f"this file is {digest[:12]}…"
            )
        # Membership: every run must have scored exactly the fold's locked test set.
        folds = {f.held_out_domain: f for f in fixed_lodo_folds(manifest)}
        for run in fixed:
            expected = {s.sample_id for s in folds[run.held_out_domain].test}
            scored = {str(r["image_id"]) for r in _per_image_rows(run.per_image_csv)}
            if scored != expected:
                raise FixedLodoReportError(
                    f"{run.label}: scored images do not match the locked test "
                    f"partition (unexpected={sorted(scored - expected)[:5]}, "
                    f"missing={sorted(expected - scored)[:5]})"
                )

        cells = build_domain_cells(fixed)
        single: tuple[Any, ...] = ()
        results: tuple[PairedResult, ...] = ()
        if not args.skip_paired:
            try:
                single = select_single_source_runs(
                    discover_single_source_runs(roots), tuple(args.expected_seeds)
                )
            except Stage3SingleSourceReportError as error:
                raise FixedLodoReportError(
                    "The paired test needs the completed train-on-one arm as its "
                    f"comparator, and it could not be loaded: {error}. Re-run with "
                    "--skip-paired to summarise the fixed-budget arm alone."
                ) from error
            substrate = build_substrate(fixed, single)
            results = paired_tests(substrate, method=args.method)
    except (FixedLodoReportError, Stage3SingleSourceReportError, OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    print(f"aggregated {len(fixed)} fixed-budget runs into {len(cells)} cells")
    print()
    print(render_domain_table(cells))
    if results:
        print()
        print(f"paired test ({args.method}, Holm-adjusted over {len(results)} tests):")
        print(render_paired_table(results))

    if args.report_out is not None:
        report = render_markdown_report(cells, results, fixed, single, manifest_path)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.report_out}")
    if args.csv_out is not None:
        print(f"wrote {write_csv(cells, results, args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
