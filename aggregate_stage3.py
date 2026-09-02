#!/usr/bin/env python3
"""Aggregate completed Stage 3 LODO runs into per-domain, seed-averaged reports.

Two different questions are answered here and they must not be conflated.

The 95% confidence interval is taken over the *seeds*. For one
(arm, held-out domain, structure, metric) there are five numbers, one per run,
each already a mean over that fold's test images. The interval

    mean +/- t(n-1, 0.975) * s / sqrt(n)

therefore quantifies how much the answer moves when the same locked data is
retrained with a different weight initialisation, augmentation draw, and shuffle
order. It is a claim about the reproducibility of the training procedure, not
about the population of fundus images. The much larger image-to-image spread
lives in ``dice_std`` inside each run's ``test_metrics.json`` and is deliberately
never used to build an interval here.

The paired test compares two conditioning arms on individual test images. The
five seeds are first averaged per image per arm, giving one value per image, and
the differences are then tested across images. Pairing is what removes the shared
image-difficulty variance that both arms see; it is only valid because the locked
manifest guarantees both arms are scored on identical image sets, which this
module asserts rather than assumes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from spfilm.lodo import (  # noqa: E402
    LODO_PROTOCOL_NAME,
    Domain,
    LodoManifest,
    LodoManifestError,
    load_lodo_manifest,
)
from spfilm.metrics import (  # noqa: E402
    CHANNEL_NAMES,
    PER_IMAGE_FIELDNAMES,
    summarise_per_image_csv,
)


TEST_METRICS_NAME = "test_metrics.json"
PER_IMAGE_CSV_NAME = "test_per_image_metrics.csv"
RESOLVED_CONFIG_NAME = "resolved_stage3_config.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "splits" / "lodo" / "lodo_manifest.json"
DEFAULT_RUN_ROOTS = (PROJECT_ROOT / "artifacts",)
DEFAULT_EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CONFIDENCE_LEVEL = 0.95
SEED_METRICS = ("dice", "iou", "hd95")
METRIC_LABELS = {"dice": "Dice", "iou": "IoU", "hd95": "HD95"}
PAIRED_METHODS = ("wilcoxon", "ttest")
SPLIT_NAMES = ("train", "val", "test")
SHA256_HEX_LENGTH = 64

# RIM-ONE-DL runs carry extra provenance columns that
# src/spfilm/engine.py::_append_rim_one_dl_per_image_context appends to the
# per-image CSV after the metrics are written. They are context, never scores,
# so they are tolerated here and never read as numbers; the test suite asserts
# this tuple still equals the engine's own so a new column cannot drift in
# unnoticed.
OPTIONAL_PER_IMAGE_FIELDS = (
    "release_prefix",
    "hospital_split",
    "diagnosis_class",
    "native_width",
    "native_height",
    "letterbox_scale",
    "hd95_unit",
)
RIM_PER_IMAGE_FIELDNAMES = (
    PER_IMAGE_FIELDNAMES[0],
    *OPTIONAL_PER_IMAGE_FIELDS,
    *PER_IMAGE_FIELDNAMES[1:],
)

# HD95 is reported in whichever frame the run measured it in. Three domains
# leave it in letterboxed-grid pixels; RIM-ONE-DL divides each value by that
# image's native-to-grid scale and reports native source pixels. Those scales
# span roughly 274-793px of native crop, so the two units differ by an
# image-dependent factor and must never share an unlabelled column.
HD95_UNIT_GRID = "letterboxed-grid pixels"
HD95_UNIT_NATIVE = "native pixels"

# summarise_per_image_rows is recomputed from the CSV and compared against the
# stored block; the round trip is through repr-formatted floats, so agreement is
# expected to be exact and the tolerance only absorbs platform noise.
SUMMARY_RELATIVE_TOLERANCE = 1e-9
SUMMARY_ABSOLUTE_TOLERANCE = 1e-12

_FLOAT_SUMMARY_FIELDS = (
    "dice_mean",
    "dice_std",
    "dice_median",
    "iou_mean",
    "iou_std",
    "accuracy_mean",
    "accuracy_std",
    "hd95_mean",
    "hd95_std",
    "hd95_median",
)
_INT_SUMMARY_FIELDS = (
    "hd95_sample_count",
    "hd95_excluded_count",
    "hd95_excluded_empty_prediction",
    "hd95_excluded_empty_target",
    "hd95_excluded_both_empty",
    "tp_total",
    "fp_total",
    "fn_total",
    "tn_total",
    "sample_count",
)


class Stage3ReportError(ValueError):
    """Raised when discovered Stage 3 output violates its reporting contract."""


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class RunIdentity:
    """Identity of one completed Stage 3 run within the reporting grid."""

    arm: str
    held_out_domain: Domain
    run_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm:
            raise Stage3ReportError("arm must be a non-empty string")
        if not isinstance(self.held_out_domain, Domain):
            raise Stage3ReportError("held_out_domain must be a Domain")
        if not isinstance(self.run_seed, int) or isinstance(self.run_seed, bool):
            raise Stage3ReportError("run_seed must be an integer")

    @property
    def label(self) -> str:
        return f"{self.arm}/{self.held_out_domain.value}/seed_{self.run_seed}"


@dataclass(frozen=True)
class StructureSummary:
    """The per-structure figures one run contributes to a seed aggregate."""

    dice_mean: float
    iou_mean: float
    hd95_mean: float | None
    hd95_sample_count: int
    hd95_excluded_count: int
    sample_count: int

    def metric_value(self, metric: str) -> float | None:
        if metric == "dice":
            return self.dice_mean
        if metric == "iou":
            return self.iou_mean
        if metric == "hd95":
            return self.hd95_mean
        raise Stage3ReportError(f"Unknown per-run metric: {metric!r}")


@dataclass(frozen=True)
class PerImageScore:
    """One row of a run's per-image CSV, the source of truth for every number."""

    image_id: str
    structure: str
    dice: float
    iou: float
    hd95: float
    acc: float
    tp: int
    fp: int
    fn: int
    tn: int

    def metric_value(self, metric: str) -> float:
        if metric == "dice":
            return self.dice
        if metric == "iou":
            return self.iou
        if metric == "acc":
            return self.acc
        # HD95 is deliberately unavailable for pairing: its per-run means are not
        # taken over a common image set, because degenerate cases are excluded
        # and the excluded set shifts from seed to seed.
        raise Stage3ReportError(
            f"{metric!r} is not poolable per image; use dice, iou, or acc"
        )


@dataclass(frozen=True)
class Stage3Run:
    """A discovered run plus the provenance needed to trust its numbers."""

    identity: RunIdentity
    run_dir: Path
    metrics_path: Path
    per_image_csv: Path
    scientific_result: bool
    smoke_rehearsal: bool
    manifest_sha256: str
    config_sha256: str
    git_revision: str
    locked_test_count: int
    locked_split_counts: Mapping[str, int]
    stored_summary: Mapping[str, Mapping[str, Any]]
    hd95_unit: str
    metric_frame: str

    @property
    def git_tree_was_dirty(self) -> bool:
        """True when the runner recorded an uncommitted working tree."""

        return self.git_revision.endswith("-dirty")


@dataclass(frozen=True)
class SeedInterval:
    """A mean over seeds with its Student-t confidence interval."""

    metric: str
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    std: float
    half_width: float
    low: float
    high: float
    confidence: float

    @property
    def count(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "seeds": list(self.seeds),
            "seed_values": list(self.values),
            "mean": self.mean,
            "std_across_seeds": self.std,
            "ci_half_width": self.half_width,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
            "seed_count": self.count,
        }


@dataclass(frozen=True)
class DomainStructureReport:
    """The reported cell: one arm, one held-out domain, one structure."""

    arm: str
    held_out_domain: Domain
    structure: str
    seeds: tuple[int, ...]
    image_count: int
    intervals: Mapping[str, SeedInterval | None]
    point_means: Mapping[str, float | None]
    hd95_sample_counts: tuple[int, ...]
    hd95_excluded_counts: tuple[int, ...]
    hd95_common_finite_count: int
    hd95_unit: str
    metric_frame: str
    hd95_common_interval: SeedInterval | None = None

    @property
    def has_interval(self) -> bool:
        """False when the cell holds too few seeds to support any interval."""

        return len(self.seeds) >= 2

    @property
    def hd95_subset_is_common(self) -> bool:
        """True when every seed scored HD95 on exactly the same images."""

        highest = max(self.hd95_sample_counts, default=0)
        return (
            highest > 0
            and self.hd95_common_finite_count == highest
            and len(set(self.hd95_sample_counts)) <= 1
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "held_out_domain": self.held_out_domain.value,
            "structure": self.structure,
            "seeds": list(self.seeds),
            "test_image_count": self.image_count,
            "metrics": {
                metric: (None if interval is None else interval.as_dict())
                for metric, interval in self.intervals.items()
            },
            "point_means": dict(self.point_means),
            "hd95_sample_count_per_seed": list(self.hd95_sample_counts),
            "hd95_excluded_count_per_seed": list(self.hd95_excluded_counts),
            "hd95_common_finite_count": self.hd95_common_finite_count,
            "hd95_subset_is_common": self.hd95_subset_is_common,
            "hd95_unit": self.hd95_unit,
            "metric_frame": self.metric_frame,
            # The HD95 figure taken over only the images finite in every seed:
            # the one HD95 number whose five means share a denominator.
            "hd95_common_subset": (
                None
                if self.hd95_common_interval is None
                else self.hd95_common_interval.as_dict()
            ),
        }


@dataclass(frozen=True)
class PairedTestResult:
    """One paired between-arm comparison for a single domain and structure."""

    held_out_domain: Domain
    structure: str
    metric: str
    method: str
    arm_a: str
    arm_b: str
    n_pairs: int
    n_informative_pairs: int
    seed_count_a: int
    seed_count_b: int
    median_difference: float
    mean_difference: float
    statistic: float
    p_value: float
    p_value_holm: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "held_out_domain": self.held_out_domain.value,
            "structure": self.structure,
            "metric": self.metric,
            "method": self.method,
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "n_pairs": self.n_pairs,
            "n_informative_pairs": self.n_informative_pairs,
            "seed_count_a": self.seed_count_a,
            "seed_count_b": self.seed_count_b,
            "median_difference": self.median_difference,
            "mean_difference": self.mean_difference,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_holm": self.p_value_holm,
        }


# --------------------------------------------------------------------------
# Stage A: run discovery
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage3ReportError(
                    f"Cannot read {path}: duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise Stage3ReportError(f"Cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise Stage3ReportError(f"{path} must contain a JSON object")
    return payload


def _parse_domain(value: object, context: str) -> Domain:
    if not isinstance(value, str):
        raise Stage3ReportError(f"{context} must be a string, got {value!r}")
    try:
        return Domain(value)
    except ValueError as error:
        raise Stage3ReportError(
            f"{context} is not a known acquisition domain: {value!r}"
        ) from error


def _require_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise Stage3ReportError(f"{context} must record a non-empty {key!r}")
    return value


def _require_bool(payload: Mapping[str, Any], key: str, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise Stage3ReportError(f"{context} must record a boolean {key!r}")
    return value


def _require_sha256(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_string(payload, key, context)
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Stage3ReportError(
            f"{context} must record {key!r} as 64 lowercase hexadecimal characters"
        )
    return value


def _require_split_counts(
    payload: Mapping[str, Any], key: str, context: str
) -> dict[str, int]:
    value = payload.get(key)
    if not isinstance(value, Mapping) or set(value) != set(SPLIT_NAMES):
        raise Stage3ReportError(
            f"{context} must record {key} with exactly {list(SPLIT_NAMES)}"
        )
    counts: dict[str, int] = {}
    for name in SPLIT_NAMES:
        count = value[name]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            raise Stage3ReportError(
                f"{context} {key}.{name} must be a positive integer"
            )
        counts[name] = count
    return counts


def resolve_run_arm(
    run_dir: Path,
    lodo: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
) -> str:
    """Resolve the arm from recorded evidence and reject contradictory sources.

    New runs stamp ``lodo.arm`` directly. Older runs are recoverable from the
    resolved source config or from the suffixed per-run experiment name. When
    more than one source exists it is corroborating evidence, not a precedence
    list: silently trusting the first would let a copied config mislabel an arm
    and invalidate every downstream paired comparison.
    """

    evidence: list[tuple[str, str]] = []
    stamped = lodo.get("arm")
    if stamped is not None:
        if not isinstance(stamped, str) or not stamped:
            raise Stage3ReportError(
                f"{run_dir} lodo.arm must be a non-empty string when present"
            )
        evidence.append(("lodo.arm", stamped))

    resolved_config = run_dir / RESOLVED_CONFIG_NAME
    if resolved_config.is_file():
        resolved_payload = _read_json(resolved_config)
        source_config = resolved_payload.get("source_config")
        if not isinstance(source_config, Mapping):
            raise Stage3ReportError(
                f"{resolved_config} must contain a source_config object"
            )
        name = source_config.get("experiment_name")
        if not isinstance(name, str) or not name:
            raise Stage3ReportError(
                f"{resolved_config} source_config must record a non-empty "
                "'experiment_name'"
            )
        evidence.append((f"{RESOLVED_CONFIG_NAME} source_config", name))

        execution = resolved_payload.get("execution")
        if not isinstance(execution, Mapping):
            raise Stage3ReportError(
                f"{resolved_config} must contain an execution object"
            )
        for key in (
            "protocol",
            "held_out_domain",
            "run_seed",
            "manifest_sha256",
            "config_sha256",
        ):
            if execution.get(key) != lodo.get(key):
                raise Stage3ReportError(
                    f"{resolved_config} execution.{key}={execution.get(key)!r} "
                    f"disagrees with {TEST_METRICS_NAME} lodo.{key}="
                    f"{lodo.get(key)!r}"
                )

    experiment_name = metrics_payload.get("experiment_name")
    domain = lodo.get("held_out_domain")
    seed = lodo.get("run_seed")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise Stage3ReportError(
            f"{run_dir} must record a non-empty experiment_name"
        )
    trimmed_name: str | None = None
    if isinstance(domain, str):
        for suffix in (f"_{domain}_seed_{seed}_smoke", f"_{domain}_seed_{seed}"):
            if experiment_name.endswith(suffix):
                candidate = experiment_name[: -len(suffix)]
                if candidate:
                    trimmed_name = candidate
                    break
    if trimmed_name is None:
        raise Stage3ReportError(
            f"{run_dir} experiment_name {experiment_name!r} does not carry the "
            "expected held-out-domain and seed suffix"
        )
    evidence.append((f"{TEST_METRICS_NAME} experiment_name", trimmed_name))

    if not evidence:
        raise Stage3ReportError(
            f"Cannot determine the conditioning arm for {run_dir}: no 'arm' in "
            f"the lodo block, no readable {RESOLVED_CONFIG_NAME}, and "
            "'experiment_name' does not carry the expected domain/seed suffix"
        )
    names = {name for _source, name in evidence}
    if len(names) != 1:
        details = ", ".join(f"{source}={name!r}" for source, name in evidence)
        raise Stage3ReportError(
            f"Conflicting conditioning-arm evidence for {run_dir}: {details}"
        )
    return names.pop()


def _build_run(metrics_path: Path) -> Stage3Run | None:
    """Turn one test_metrics.json into a run, or None if it is not a Stage 3 run."""

    payload = _read_json(metrics_path)
    lodo = payload.get("lodo")
    if lodo is None:
        return None
    if not isinstance(lodo, Mapping):
        raise Stage3ReportError(f"{metrics_path} has a non-object 'lodo' block")

    context = f"{metrics_path} lodo block"
    protocol = _require_string(lodo, "protocol", context)
    if protocol != LODO_PROTOCOL_NAME:
        raise Stage3ReportError(
            f"{context} records protocol {protocol!r}, expected "
            f"{LODO_PROTOCOL_NAME!r}"
        )

    held_out_domain = _parse_domain(
        lodo.get("held_out_domain"), f"{context} held_out_domain"
    )
    run_seed = lodo.get("run_seed")
    if not isinstance(run_seed, int) or isinstance(run_seed, bool):
        raise Stage3ReportError(f"{context} must record an integer 'run_seed'")

    locked = _require_split_counts(lodo, "locked_split_counts", context)
    executed = _require_split_counts(lodo, "executed_split_counts", context)
    scientific_result = _require_bool(lodo, "scientific_result", context)
    smoke_rehearsal = _require_bool(lodo, "smoke_rehearsal", context)
    if scientific_result == smoke_rehearsal:
        raise Stage3ReportError(
            f"{context} has contradictory result flags: scientific_result="
            f"{scientific_result!r}, smoke_rehearsal={smoke_rehearsal!r}"
        )
    if lodo.get("started_from_locked_membership") is not True:
        raise Stage3ReportError(
            f"{context} must record started_from_locked_membership=true"
        )
    if scientific_result and executed != locked:
        raise Stage3ReportError(
            f"{context} is marked scientific but executed_split_counts "
            f"{executed} do not equal locked_split_counts {locked}"
        )

    source_domains = lodo.get("source_domains")
    expected_sources = sorted(
        domain.value for domain in Domain if domain is not held_out_domain
    )
    if source_domains != expected_sources:
        raise Stage3ReportError(
            f"{context} source_domains must be {expected_sources} for held-out "
            f"{held_out_domain.value}, got {source_domains!r}"
        )

    test_block = payload.get("test")
    if not isinstance(test_block, Mapping):
        raise Stage3ReportError(f"{metrics_path} has no 'test' block")
    evaluated_count = test_block.get("evaluated_sample_count")
    if (
        not isinstance(evaluated_count, int)
        or isinstance(evaluated_count, bool)
        or evaluated_count <= 0
    ):
        raise Stage3ReportError(
            f"{metrics_path} test block must record evaluated_sample_count as a "
            "positive integer"
        )
    if evaluated_count != executed["test"]:
        raise Stage3ReportError(
            f"{metrics_path} test.evaluated_sample_count={evaluated_count} but "
            f"executed_split_counts.test={executed['test']}"
        )
    _require_string(test_block, "per_image_csv", f"{metrics_path} test block")
    stored_summary: dict[str, Mapping[str, Any]] = {}
    for structure in CHANNEL_NAMES:
        summary = test_block.get(structure)
        if not isinstance(summary, Mapping):
            raise Stage3ReportError(
                f"{metrics_path} test block has no {structure!r} summary"
            )
        stored_summary[structure] = summary

    # HD95 is only meaningful next to the grid it was measured on. The engine
    # writes metric_frame unconditionally and hd95_unit only when it converted
    # to native pixels, so an absent hd95_unit means letterboxed-grid pixels.
    metric_frame = test_block.get("metric_frame")
    if not isinstance(metric_frame, str) or not metric_frame:
        raise Stage3ReportError(
            f"{metrics_path} test block must record a non-empty 'metric_frame'; "
            "without it the HD95 figures have no stated unit"
        )
    stored_unit = test_block.get("hd95_unit")
    if stored_unit is not None and not isinstance(stored_unit, str):
        raise Stage3ReportError(
            f"{metrics_path} test block has a non-string 'hd95_unit': "
            f"{stored_unit!r}"
        )
    hd95_unit = stored_unit or HD95_UNIT_GRID
    expected_unit = (
        HD95_UNIT_NATIVE
        if held_out_domain is Domain.RIM_ONE_DL
        else HD95_UNIT_GRID
    )
    if hd95_unit != expected_unit:
        raise Stage3ReportError(
            f"{metrics_path} records HD95 in {hd95_unit!r} for "
            f"{held_out_domain.value}; current Stage 3 output requires "
            f"{expected_unit!r}"
        )

    run_dir = metrics_path.parent
    per_image_csv = run_dir / PER_IMAGE_CSV_NAME
    # test.per_image_csv stores an absolute path from the machine that trained;
    # it is stale for output copied off CREATE, so resolve beside the JSON.
    if not per_image_csv.is_file():
        raise Stage3ReportError(
            f"{run_dir} has {TEST_METRICS_NAME} but no {PER_IMAGE_CSV_NAME}"
        )

    identity = RunIdentity(
        arm=resolve_run_arm(run_dir, lodo, payload),
        held_out_domain=held_out_domain,
        run_seed=run_seed,
    )
    return Stage3Run(
        identity=identity,
        run_dir=run_dir,
        metrics_path=metrics_path,
        per_image_csv=per_image_csv,
        scientific_result=scientific_result,
        smoke_rehearsal=smoke_rehearsal,
        manifest_sha256=_require_sha256(lodo, "manifest_sha256", context),
        config_sha256=_require_sha256(lodo, "config_sha256", context),
        # Not a sha256: the runner writes a 40-character git commit, optionally
        # suffixed "-dirty" when the working tree had uncommitted changes.
        git_revision=_require_string(lodo, "git_revision", context),
        locked_test_count=locked["test"],
        locked_split_counts=locked,
        stored_summary=stored_summary,
        hd95_unit=hd95_unit,
        metric_frame=metric_frame,
    )


def discover_stage3_runs(roots: Iterable[str | Path]) -> tuple[Stage3Run, ...]:
    """Find every Stage 3 run under ``roots`` by file, not by directory shape.

    Two layouts exist -- ``artifacts/stage3_lodo/<domain>/seed_<n>`` locally and
    ``artifacts/runs/lodo_s3_<domain>_seed_<n>_<jobid>`` on CREATE -- so runs are
    located by their ``test_metrics.json`` and identified by the ``lodo`` block
    inside it. Output from other stages has no ``lodo`` block and is ignored.
    """

    root_paths = [Path(root).expanduser().resolve() for root in roots]
    if not root_paths:
        raise Stage3ReportError("At least one run root must be given")
    for root in root_paths:
        if not root.is_dir():
            raise Stage3ReportError(f"Run root is not a directory: {root}")

    seen_files: set[Path] = set()
    metrics_paths: list[Path] = []
    for root in root_paths:
        for candidate in sorted(root.rglob(TEST_METRICS_NAME)):
            resolved = candidate.resolve()
            if resolved not in seen_files:
                seen_files.add(resolved)
                metrics_paths.append(resolved)

    runs: list[Stage3Run] = []
    # Rehearsals and the eventual scientific run deliberately share the same
    # arm/domain/seed, and users may repeat a smoke before a long launch. They
    # may all coexist under one CREATE artifact root so --skip-smoke can discard
    # them later. What must remain impossible is two scientific results claiming
    # the same reportable grid cell.
    scientific_by_identity: dict[RunIdentity, Stage3Run] = {}
    for metrics_path in metrics_paths:
        run = _build_run(metrics_path)
        if run is None:
            continue
        if run.scientific_result:
            previous = scientific_by_identity.get(run.identity)
            if previous is not None:
                raise Stage3ReportError(
                    f"Duplicate Stage 3 run for {run.identity.label}: "
                    f"{previous.run_dir} and {run.run_dir}"
                )
            scientific_by_identity[run.identity] = run
        runs.append(run)

    if not runs:
        raise Stage3ReportError(
            "No Stage 3 runs found under "
            + ", ".join(str(root) for root in root_paths)
            + f"; a Stage 3 run is a directory containing {TEST_METRICS_NAME} "
            "with a 'lodo' block"
        )
    return tuple(sorted(runs, key=lambda run: run.identity))


def select_scientific_runs(
    runs: Sequence[Stage3Run], *, skip_smoke: bool = False
) -> tuple[Stage3Run, ...]:
    """Keep only runs that claim to be scientific results.

    A smoke rehearsal trains on a handful of images to prove the pipeline is
    wired up; its Dice is meaningless. Dropping such a run silently is the
    failure mode this refuses, so by default the presence of one is an error and
    ``skip_smoke`` must be asked for explicitly.
    """

    rehearsals = [run for run in runs if not run.scientific_result]
    if rehearsals and not skip_smoke:
        listed = "\n".join(
            f"  {run.identity.label}: {run.run_dir}" for run in rehearsals
        )
        raise Stage3ReportError(
            f"{len(rehearsals)} discovered run(s) record scientific_result=false "
            "and cannot enter a report:\n"
            f"{listed}\n"
            "Point --runs at the real runs, or pass --skip-smoke to exclude "
            "these deliberately."
        )
    kept = tuple(run for run in runs if run.scientific_result)
    if not kept:
        raise Stage3ReportError(
            "Every discovered run is a smoke rehearsal; there is nothing to report"
        )
    return kept


# --------------------------------------------------------------------------
# Stage B: per-image rows and locked-membership validation
# --------------------------------------------------------------------------


def fold_test_image_ids(manifest: LodoManifest, domain: Domain) -> tuple[str, ...]:
    """The locked test partition for one held-out domain, in manifest order."""

    for fold in manifest.folds:
        if fold.held_out_domain == domain:
            return tuple(sample.sample_id for sample in fold.test)
    raise Stage3ReportError(
        f"The manifest has no fold for held-out domain {domain.value!r}"
    )


def _parse_float(value: object, context: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise Stage3ReportError(f"{context} is not a number: {value!r}") from error


def _parse_int(value: object, context: str) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise Stage3ReportError(
            f"{context} is not an integer: {value!r}"
        ) from error


def _validate_score(score: PerImageScore, context: str) -> int:
    """Validate redundant row fields so self-consistent corruption cannot pass."""

    for name in ("dice", "iou", "acc"):
        value = float(getattr(score, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise Stage3ReportError(
                f"{context} {name} must be finite and lie in [0, 1], got {value!r}"
            )
    if math.isinf(score.hd95) or (
        math.isfinite(score.hd95) and score.hd95 < 0.0
    ):
        raise Stage3ReportError(
            f"{context} hd95 must be non-negative or nan, got {score.hd95!r}"
        )
    for name in ("tp", "fp", "fn", "tn"):
        value = int(getattr(score, name))
        if value < 0:
            raise Stage3ReportError(
                f"{context} {name} must be non-negative, got {value}"
            )

    total = score.tp + score.fp + score.fn + score.tn
    if total <= 0:
        raise Stage3ReportError(f"{context} confusion counts sum to {total}")
    smooth = 1e-8
    expected = {
        "dice": (2 * score.tp + smooth)
        / (2 * score.tp + score.fp + score.fn + smooth),
        "iou": (score.tp + smooth)
        / (score.tp + score.fp + score.fn + smooth),
        "acc": (score.tp + score.tn) / total,
    }
    for name, recomputed in expected.items():
        recorded = float(getattr(score, name))
        if not math.isclose(
            recorded,
            recomputed,
            rel_tol=SUMMARY_RELATIVE_TOLERANCE,
            abs_tol=SUMMARY_ABSOLUTE_TOLERANCE,
        ):
            raise Stage3ReportError(
                f"{context} {name}={recorded!r} disagrees with confusion-count "
                f"recomputation {recomputed!r}"
            )
    return total


def _rim_context(row: Mapping[str, object], context: str) -> tuple[str, ...]:
    """Validate the native-pixel context appended to every RIM-ONE-DL row."""

    for name in ("release_prefix", "hospital_split", "diagnosis_class"):
        value = row.get(name)
        if not isinstance(value, str) or not value:
            raise Stage3ReportError(f"{context} {name} must be a non-empty string")
    for name in ("native_width", "native_height"):
        value = _parse_int(row.get(name), f"{context} {name}")
        if value <= 0:
            raise Stage3ReportError(f"{context} {name} must be positive")
    scale = _parse_float(row.get("letterbox_scale"), f"{context} letterbox_scale")
    if not math.isfinite(scale) or scale <= 0.0:
        raise Stage3ReportError(
            f"{context} letterbox_scale must be finite and positive"
        )
    if row.get("hd95_unit") != "native_px":
        raise Stage3ReportError(
            f"{context} hd95_unit must be 'native_px', got "
            f"{row.get('hd95_unit')!r}"
        )
    return tuple(str(row[name]) for name in OPTIONAL_PER_IMAGE_FIELDS)


def load_run_scores(
    run: Stage3Run, manifest: LodoManifest
) -> tuple[PerImageScore, ...]:
    """Read one run's per-image CSV and prove it covers the locked test fold.

    A truncated file, a run whose output landed in the wrong directory, or a CSV
    written for a different domain all surface here rather than as a quietly
    wrong mean further down.
    """

    try:
        with run.per_image_csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            # Current RIM-ONE-DL output appends native-pixel provenance in one
            # fixed order. Other domains retain the base schema. Tying the
            # accepted header to the held-out domain prevents a copied or
            # partially annotated CSV from silently assigning HD95 the wrong
            # coordinate frame.
            expected_header = (
                RIM_PER_IMAGE_FIELDNAMES
                if run.identity.held_out_domain is Domain.RIM_ONE_DL
                else PER_IMAGE_FIELDNAMES
            )
            if fieldnames != expected_header:
                raise Stage3ReportError(
                    f"{run.per_image_csv} header is {list(fieldnames)}, expected "
                    f"{list(expected_header)} for "
                    f"{run.identity.held_out_domain.value}"
                )
            rows = list(reader)
    except OSError as error:
        raise Stage3ReportError(
            f"Cannot read {run.per_image_csv}: {error}"
        ) from error

    scores: list[PerImageScore] = []
    seen: set[tuple[str, str]] = set()
    pixel_totals: set[int] = set()
    rim_context_by_image: dict[str, tuple[str, ...]] = {}
    for line_number, row in enumerate(rows, start=2):
        context = f"{run.per_image_csv} line {line_number}"
        if None in row:
            raise Stage3ReportError(
                f"{context} has values beyond the declared CSV header"
            )
        if any(row.get(name) is None for name in PER_IMAGE_FIELDNAMES):
            raise Stage3ReportError(f"{context} is missing one or more columns")
        image_id = str(row["image_id"])
        structure = str(row["structure"])
        if structure not in CHANNEL_NAMES:
            raise Stage3ReportError(
                f"{context} has unknown structure {structure!r}; expected one of "
                f"{list(CHANNEL_NAMES)}"
            )
        key = (image_id, structure)
        if key in seen:
            raise Stage3ReportError(
                f"{context} repeats {structure} for image {image_id!r}"
            )
        seen.add(key)
        score = PerImageScore(
            image_id=image_id,
            structure=structure,
            dice=_parse_float(row["dice"], f"{context} dice"),
            iou=_parse_float(row["iou"], f"{context} iou"),
            hd95=_parse_float(row["hd95"], f"{context} hd95"),
            acc=_parse_float(row["acc"], f"{context} acc"),
            tp=_parse_int(row["tp"], f"{context} tp"),
            fp=_parse_int(row["fp"], f"{context} fp"),
            fn=_parse_int(row["fn"], f"{context} fn"),
            tn=_parse_int(row["tn"], f"{context} tn"),
        )
        pixel_totals.add(_validate_score(score, context))
        if run.identity.held_out_domain is Domain.RIM_ONE_DL:
            row_context = _rim_context(row, context)
            previous_context = rim_context_by_image.setdefault(
                image_id, row_context
            )
            if previous_context != row_context:
                raise Stage3ReportError(
                    f"{context} RIM-ONE-DL context disagrees between disc and cup "
                    f"rows for image {image_id!r}"
                )
        scores.append(score)

    if len(pixel_totals) != 1:
        raise Stage3ReportError(
            f"{run.identity.label} rows span different evaluation-grid pixel "
            f"counts: {sorted(pixel_totals)}"
        )

    # The CSV carries no domain column, and it does not need one: a run's test
    # set is single-domain by protocol. The domain comes from lodo.held_out_domain.
    fold = next(
        (
            item
            for item in manifest.folds
            if item.held_out_domain == run.identity.held_out_domain
        ),
        None,
    )
    if fold is None:
        raise Stage3ReportError(
            f"The manifest has no fold for {run.identity.held_out_domain.value}"
        )
    expected_counts = {
        name: len(getattr(fold, name)) for name in SPLIT_NAMES
    }
    if dict(run.locked_split_counts) != expected_counts:
        raise Stage3ReportError(
            f"{run.identity.label} records locked split counts "
            f"{dict(run.locked_split_counts)} but the manifest fold has "
            f"{expected_counts}"
        )

    expected_ids = {sample.sample_id for sample in fold.test}
    if len(expected_ids) != run.locked_test_count:
        raise Stage3ReportError(
            f"{run.identity.label} records locked test count "
            f"{run.locked_test_count} but the manifest fold holds "
            f"{len(expected_ids)} images"
        )

    test_block = _read_json(run.metrics_path).get("test")
    recorded_ids = test_block.get("sample_ids") if isinstance(test_block, Mapping) else None
    if (
        not isinstance(recorded_ids, list)
        or not all(isinstance(image_id, str) for image_id in recorded_ids)
        or len(recorded_ids) != len(set(recorded_ids))
        or set(recorded_ids) != expected_ids
    ):
        raise Stage3ReportError(
            f"{run.identity.label} test.sample_ids do not exactly match the "
            f"locked {run.identity.held_out_domain.value} test partition"
        )

    for structure in CHANNEL_NAMES:
        present = {
            score.image_id for score in scores if score.structure == structure
        }
        if present != expected_ids:
            missing = sorted(expected_ids - present)
            unknown = sorted(present - expected_ids)
            raise Stage3ReportError(
                f"{run.identity.label} {structure} rows do not match the locked "
                f"{run.identity.held_out_domain.value} test partition "
                f"({len(present)} of {len(expected_ids)} images): "
                f"missing={missing[:5]}, unexpected={unknown[:5]}"
            )
    return tuple(scores)


# --------------------------------------------------------------------------
# Stage C: recomputed per-run summaries must equal the stored ones
# --------------------------------------------------------------------------


def _summaries_agree(recomputed: object, stored: object) -> bool:
    if recomputed is None or stored is None:
        return recomputed is None and stored is None
    try:
        left = float(recomputed)  # type: ignore[arg-type]
        right = float(stored)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return math.isclose(
        left,
        right,
        rel_tol=SUMMARY_RELATIVE_TOLERANCE,
        abs_tol=SUMMARY_ABSOLUTE_TOLERANCE,
    )


def verify_run_summary(run: Stage3Run) -> dict[str, StructureSummary]:
    """Recompute the per-structure summary from the CSV and match it to the JSON.

    The CSV is the source of truth. If the stored block ever disagrees with it,
    the report would otherwise quote a number no file supports.
    """

    recomputed = summarise_per_image_csv(run.per_image_csv)
    summaries: dict[str, StructureSummary] = {}
    for structure in CHANNEL_NAMES:
        if structure not in recomputed:
            raise Stage3ReportError(
                f"{run.identity.label}: {run.per_image_csv} has no {structure} rows"
            )
        fresh = recomputed[structure]
        stored = run.stored_summary[structure]
        for field_name in _FLOAT_SUMMARY_FIELDS:
            if not _summaries_agree(fresh.get(field_name), stored.get(field_name)):
                raise Stage3ReportError(
                    f"{run.identity.label} {structure}.{field_name} recomputed as "
                    f"{fresh.get(field_name)!r} but {TEST_METRICS_NAME} stores "
                    f"{stored.get(field_name)!r}"
                )
        for field_name in _INT_SUMMARY_FIELDS:
            if fresh.get(field_name) != stored.get(field_name):
                raise Stage3ReportError(
                    f"{run.identity.label} {structure}.{field_name} recomputed as "
                    f"{fresh.get(field_name)!r} but {TEST_METRICS_NAME} stores "
                    f"{stored.get(field_name)!r}"
                )
        hd95_mean = fresh["hd95_mean"]
        summaries[structure] = StructureSummary(
            dice_mean=float(fresh["dice_mean"]),
            iou_mean=float(fresh["iou_mean"]),
            hd95_mean=None if hd95_mean is None else float(hd95_mean),
            hd95_sample_count=int(fresh["hd95_sample_count"]),
            hd95_excluded_count=int(fresh["hd95_excluded_count"]),
            sample_count=int(fresh["sample_count"]),
        )
    return summaries


# --------------------------------------------------------------------------
# Stage D: seed aggregation and the confidence interval
# --------------------------------------------------------------------------


def seed_confidence_interval(
    metric: str,
    seeds: Sequence[int],
    values: Sequence[float],
    confidence: float = CONFIDENCE_LEVEL,
) -> SeedInterval:
    """Mean +/- t(n-1, 0.975) * s/sqrt(n) over the per-seed run means.

    ``values`` must be one number per seed, each already a mean over that run's
    test images. The spread here is training-run wobble under a fixed, locked
    dataset -- not the image-to-image spread, which is far larger and is what
    ``dice_std`` in each run's JSON reports.
    """

    if len(seeds) != len(values):
        raise Stage3ReportError(
            f"{metric}: got {len(values)} values for {len(seeds)} seeds"
        )
    if len(values) < 2:
        raise Stage3ReportError(
            f"{metric}: a confidence interval over seeds needs at least 2 seeds, "
            f"got {len(values)}"
        )
    if not 0.0 < confidence < 1.0:
        raise Stage3ReportError(f"confidence must lie in (0, 1), got {confidence!r}")

    sample = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(sample)):
        raise Stage3ReportError(
            f"{metric}: seed values must all be finite, got {list(values)}"
        )
    count = sample.size
    mean = float(np.mean(sample))
    # ddof=1: these five runs are a sample of the training procedure's outcomes,
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


# --------------------------------------------------------------------------
# Stage E: the per-domain table
# --------------------------------------------------------------------------


def build_domain_reports(
    runs: Sequence[Stage3Run],
    summaries: Mapping[RunIdentity, Mapping[str, StructureSummary]],
    scores: Mapping[RunIdentity, Sequence[PerImageScore]],
    *,
    expected_seeds: Sequence[int] | None = DEFAULT_EXPECTED_SEEDS,
    expected_domains: Sequence[Domain] | None = tuple(Domain),
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[DomainStructureReport, ...]:
    """Collapse seeds into one reported cell per arm, domain, and structure.

    Disc and cup stay separate throughout; no combined figure is produced, and
    no cross-domain average is produced either, because the four held-out
    domains are four different questions rather than four samples of one.

    A grid missing a whole held-out domain still renders as a tidy table, so
    completeness is checked rather than assumed: a three-domain report that
    looks like a four-domain report is the quiet failure worth refusing.
    """

    grouped: dict[tuple[str, Domain], list[Stage3Run]] = {}
    for run in runs:
        key = (run.identity.arm, run.identity.held_out_domain)
        grouped.setdefault(key, []).append(run)

    if expected_domains is not None:
        wanted_domains = set(expected_domains)
        for arm in sorted({run.identity.arm for run in runs}):
            present = {
                domain for (cell_arm, domain) in grouped if cell_arm == arm
            }
            if present != wanted_domains:
                missing = sorted(
                    domain.value for domain in wanted_domains - present
                )
                extra = sorted(
                    domain.value for domain in present - wanted_domains
                )
                raise Stage3ReportError(
                    f"Arm {arm!r} does not cover every held-out domain: "
                    f"missing={missing}, unexpected={extra}; pass "
                    "--expect-domains with no values to report an incomplete "
                    "grid deliberately"
                )

    # A ragged grid is refused by default, because five seeds in one domain and
    # three in another silently changes what the interval means from cell to
    # cell. It is allowed only under the same deliberate opt-out that relaxes
    # the seed expectation, so a partially finished grid can still be inspected.
    seed_sets: dict[str, tuple[Domain, set[int]]] = {}
    for (arm, domain), cell_runs in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        seeds = {run.identity.run_seed for run in cell_runs}
        first_domain, previous = seed_sets.setdefault(arm, (domain, seeds))
        if previous != seeds and expected_seeds is not None:
            raise Stage3ReportError(
                f"Arm {arm!r} does not use the same seeds for every held-out "
                f"domain: {first_domain.value} has {sorted(previous)} but "
                f"{domain.value} has {sorted(seeds)}; pass --expect-seeds with "
                "no values to report a partially finished grid deliberately"
            )

    if expected_seeds is not None:
        wanted = set(expected_seeds)
        for (arm, domain), cell_runs in sorted(grouped.items()):
            seeds = {run.identity.run_seed for run in cell_runs}
            if seeds != wanted:
                raise Stage3ReportError(
                    f"{arm}/{domain.value} has seeds {sorted(seeds)} but "
                    f"{sorted(wanted)} were expected; pass --expect-seeds with no "
                    "values to report an incomplete grid deliberately"
                )

    reports: list[DomainStructureReport] = []
    for (arm, domain), cell_runs in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        ordered = sorted(cell_runs, key=lambda run: run.identity.run_seed)
        seeds = tuple(run.identity.run_seed for run in ordered)
        # HD95 is only additive within one measurement frame. Three domains
        # report letterboxed-grid pixels and RIM-ONE-DL reports native source
        # pixels, so the unit is carried per cell; within a cell the seeds must
        # agree, or the grid mixes resolutions and its means are not the same
        # quantity. The frame is a property of the run, not of the structure.
        frames = {run.metric_frame for run in ordered}
        if len(frames) != 1:
            raise Stage3ReportError(
                f"{arm}/{domain.value} spans more than one metric frame across "
                "its seeds, so its HD95 values are not in a common unit and its "
                f"Dice is not on a common grid: {sorted(frames)}"
            )
        units = {run.hd95_unit for run in ordered}
        if len(units) != 1:
            raise Stage3ReportError(
                f"{arm}/{domain.value} spans more than one HD95 unit across its "
                f"seeds: {sorted(units)}"
            )

        for structure in CHANNEL_NAMES:
            cells = [summaries[run.identity][structure] for run in ordered]
            image_counts = {cell.sample_count for cell in cells}
            if len(image_counts) != 1:
                raise Stage3ReportError(
                    f"{arm}/{domain.value} {structure} runs score different image "
                    f"counts across seeds: {sorted(image_counts)}"
                )
            intervals: dict[str, SeedInterval | None] = {}
            point_means: dict[str, float | None] = {}
            for metric in SEED_METRICS:
                values = [cell.metric_value(metric) for cell in cells]
                if any(value is None for value in values):
                    # Only reachable for HD95, when a whole run had no finite
                    # distance at all. Reporting nothing beats reporting a mean
                    # over a subset of the seeds.
                    intervals[metric] = None
                    point_means[metric] = None
                    continue
                numbers = [float(value) for value in values]
                point_means[metric] = float(np.mean(numbers))
                if len(seeds) < 2:
                    # One seed is a measurement, not a sample; it carries a mean
                    # but cannot carry an interval.
                    intervals[metric] = None
                    continue
                intervals[metric] = seed_confidence_interval(
                    metric,
                    seeds,
                    numbers,
                    confidence=confidence,
                )
            # Equal exclusion counts do not imply the same excluded images:
            # the degenerate cases move from seed to seed, so the intersection
            # is what says whether the five HD95 means share a denominator.
            finite_sets = [
                {
                    score.image_id
                    for score in scores[run.identity]
                    if score.structure == structure and math.isfinite(score.hd95)
                }
                for run in ordered
            ]
            common_ids = (
                set.intersection(*finite_sets) if finite_sets else set()
            )
            common_finite = len(common_ids)
            # The all-finite HD95 mean averages five means taken over five
            # different image sets. Restricting every seed to the images finite
            # in all of them gives the one HD95 figure whose five means share a
            # denominator, so it is the number that can be defended.
            common_interval: SeedInterval | None = None
            if common_ids and len(seeds) >= 2:
                common_values = []
                for run in ordered:
                    per_image = {
                        score.image_id: score.hd95
                        for score in scores[run.identity]
                        if score.structure == structure
                    }
                    common_values.append(
                        float(
                            np.mean(
                                [per_image[image_id] for image_id in sorted(common_ids)]
                            )
                        )
                    )
                common_interval = seed_confidence_interval(
                    "hd95_common_subset",
                    seeds,
                    common_values,
                    confidence=confidence,
                )
            reports.append(
                DomainStructureReport(
                    arm=arm,
                    held_out_domain=domain,
                    structure=structure,
                    seeds=seeds,
                    image_count=image_counts.pop(),
                    intervals=intervals,
                    point_means=point_means,
                    hd95_sample_counts=tuple(
                        cell.hd95_sample_count for cell in cells
                    ),
                    hd95_excluded_counts=tuple(
                        cell.hd95_excluded_count for cell in cells
                    ),
                    hd95_common_finite_count=common_finite,
                    hd95_unit=ordered[0].hd95_unit,
                    metric_frame=ordered[0].metric_frame,
                    hd95_common_interval=common_interval,
                )
            )
    return tuple(reports)


def render_domain_table(reports: Sequence[DomainStructureReport]) -> str:
    """Render one block per held-out domain; disc and cup are never combined.

    Rows are kept inside a normal terminal width and the HD95 caveat is carried
    on its own wrapped line beneath them, so the numbers stay in aligned columns
    instead of being pushed off the edge by a long note.
    """

    lines: list[str] = []
    grouped: dict[tuple[str, Domain], list[DomainStructureReport]] = {}
    for report in reports:
        grouped.setdefault((report.arm, report.held_out_domain), []).append(report)

    for (arm, domain), cells in grouped.items():
        seeds = cells[0].seeds
        ordered = sorted(cells, key=lambda cell: CHANNEL_NAMES.index(cell.structure))
        lines.append("")
        lines.append(f"held-out domain: {domain.value}    arm: {arm}")
        lines.append(
            f"  {cells[0].image_count} locked test images; "
            f"seeds {', '.join(str(seed) for seed in seeds)} (n={len(seeds)})"
        )
        if cells[0].has_interval:
            lines.append("  intervals are 95% CI over the per-seed means")
        else:
            lines.append(
                f"  one seed only: means are shown, no interval exists "
                f"(n={len(seeds)})"
            )
        lines.append(f"  HD95 unit: {cells[0].hd95_unit}")
        lines.append(
            f"  {'structure':<10}{'metric':<8}{'mean':>10}"
            f"{'95% CI':>24}{'sd(seeds)':>12}"
        )
        for report in ordered:
            for metric in SEED_METRICS:
                lines.append(_metric_row(report, metric, METRIC_LABELS[metric]))
            common = report.hd95_common_interval
            if common is not None and not report.hd95_subset_is_common:
                # The all-finite HD95 above is five means over five different
                # image sets; this one is the same images in every seed.
                bounds = f"[{common.low:.2f}, {common.high:.2f}]"
                lines.append(
                    f"  {report.structure:<10}{'HD95*':<8}"
                    f"{common.mean:>10.2f}{bounds:>24}{common.std:>12.3g}"
                )
        if any(
            cell.hd95_common_interval is not None
            and not cell.hd95_subset_is_common
            for cell in ordered
        ):
            lines.append(
                "  HD95* is the same metric restricted to the images finite in "
                "every seed"
            )
        for report in ordered:
            lines.extend(
                textwrap.wrap(
                    f"HD95 ({report.structure}): {_hd95_note(report)}",
                    width=78,
                    initial_indent="  ",
                    subsequent_indent="      ",
                )
            )
    return "\n".join(lines)


def _metric_row(
    report: DomainStructureReport, metric: str, label: str
) -> str:
    """One aligned table row, with a mean even where no interval can exist."""

    interval = report.intervals.get(metric)
    point = report.point_means.get(metric)
    digits = 2 if metric == "hd95" else 4
    if interval is None:
        # Either HD95 had no finite distance in some seed, or the cell holds a
        # single seed; the note lines say which, and neither may print a zero.
        mean = "n/a" if point is None else f"{point:.{digits}f}"
        return (
            f"  {report.structure:<10}{label:<8}{mean:>10}{'-':>24}{'-':>12}"
        )
    bounds = f"[{interval.low:.{digits}f}, {interval.high:.{digits}f}]"
    return (
        f"  {report.structure:<10}{label:<8}"
        f"{interval.mean:>10.{digits}f}{bounds:>24}"
        # 3 significant figures, so a real but tiny seed spread never renders
        # as a flat 0.00.
        f"{interval.std:>12.3g}"
    )


def _hd95_note(report: DomainStructureReport) -> str:
    """Say whether the seeds' HD95 means share a denominator, not just a count.

    Equal per-seed exclusion counts are not evidence of a stable subset. When the
    intersection is smaller than the per-seed counts, the seed mean is an average
    of five means taken over five different image sets, and the note says so.
    """

    finite = report.hd95_sample_counts
    total = report.image_count
    low, high = min(finite), max(finite)
    if high == 0:
        return (
            f"undefined for all {total} images in every seed, so no HD95 mean "
            "or interval is reported"
        )
    if low == 0:
        return (
            f"undefined for every image in at least one seed (finite 0-{high}/"
            f"{total} per seed), so no interval is reported"
        )
    span = f"{low}" if low == high else f"{low}-{high}"
    base = f"finite {span}/{total} per seed"
    if report.hd95_subset_is_common:
        return f"{base}, the same images every seed"
    if report.hd95_common_finite_count == 0:
        return (
            f"{base}; no image is finite in every seed, so the means above use "
            f"{len(finite)} different image sets and no common-subset HD95 can "
            "be reported"
        )
    return (
        f"{base}; only {report.hd95_common_finite_count}/{total} are finite in "
        f"all {len(finite)} seeds, so the means above are taken over "
        f"{len(finite)} different image sets and are not directly comparable "
        "across seeds; the HD95* row restricts every seed to the common images"
    )


# --------------------------------------------------------------------------
# Stage F: the paired between-arm test substrate
# --------------------------------------------------------------------------


PairKey = tuple[str, Domain, str, str]


@dataclass(frozen=True)
class PairedSubstrate:
    """One value per image per arm, with the seeds already averaged away.

    Keys are ``(arm, held_out_domain, structure, image_id)``. Averaging the five
    seeds first is what makes the pairs real: the alternative of pooling all
    5 x ~80 seed-image scores would count each image five times and treat five
    correlated numbers as independent evidence.
    """

    metrics: tuple[str, ...]
    seed_counts: Mapping[tuple[str, Domain], int]
    values: Mapping[PairKey, Mapping[str, float]]

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted({key[0] for key in self.values}))

    def cells(self, arm: str) -> tuple[tuple[Domain, str], ...]:
        return tuple(
            sorted(
                {(key[1], key[2]) for key in self.values if key[0] == arm},
                key=lambda item: (item[0].value, CHANNEL_NAMES.index(item[1])),
            )
        )

    def image_ids(self, arm: str, domain: Domain, structure: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                key[3]
                for key in self.values
                if key[0] == arm and key[1] == domain and key[2] == structure
            )
        )


def build_paired_substrate(
    runs: Sequence[Stage3Run],
    scores: Mapping[RunIdentity, Sequence[PerImageScore]],
    *,
    metrics: Sequence[str] = ("dice", "iou"),
) -> PairedSubstrate:
    """Average each image's score over that arm's seeds, ready for pairing.

    HD95 is deliberately absent: its per-run figures are not taken over a common
    image set, because degenerate cases are excluded and the excluded set moves
    from seed to seed. Dice and IoU are defined for every image in every run,
    which is why the paired test is specified on Dice.
    """

    accumulated: dict[PairKey, dict[str, list[float]]] = {}
    seed_counts: dict[tuple[str, Domain], set[int]] = {}
    for run in runs:
        identity = run.identity
        cell = (identity.arm, identity.held_out_domain)
        seed_counts.setdefault(cell, set()).add(identity.run_seed)
        for score in scores[identity]:
            key = (
                identity.arm,
                identity.held_out_domain,
                score.structure,
                score.image_id,
            )
            bucket = accumulated.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                bucket[metric].append(score.metric_value(metric))

    values: dict[PairKey, Mapping[str, float]] = {}
    for key, bucket in accumulated.items():
        expected = len(seed_counts[(key[0], key[1])])
        for metric in metrics:
            if len(bucket[metric]) != expected:
                raise Stage3ReportError(
                    f"{key[0]}/{key[1].value} {key[2]} image {key[3]!r} has "
                    f"{len(bucket[metric])} {metric} values but the arm ran "
                    f"{expected} seeds"
                )
        values[key] = {
            metric: float(np.mean(bucket[metric])) for metric in metrics
        }

    return PairedSubstrate(
        metrics=tuple(metrics),
        seed_counts={cell: len(seeds) for cell, seeds in seed_counts.items()},
        values=values,
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm-Bonferroni step-down adjustment, preserving input order.

    Four held-out domains times two structures is eight tests. At alpha=0.05 that
    is roughly a 34% chance of at least one false positive if every null is true,
    so a single significant cell out of eight is weak evidence on its own.
    """

    count = len(p_values)
    if count == 0:
        return ()
    for value in p_values:
        if not 0.0 <= float(value) <= 1.0:
            raise Stage3ReportError(f"p-value out of range: {value!r}")
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * float(p_values[index])
        running = max(running, min(candidate, 1.0))
        adjusted[index] = running
    return tuple(adjusted)


def paired_arm_test(
    substrate: PairedSubstrate,
    arm_a: str,
    arm_b: str,
    *,
    metric: str = "dice",
    method: str = "wilcoxon",
    allow_unequal_seeds: bool = False,
) -> tuple[PairedTestResult, ...]:
    """Test ``arm_b - arm_a`` per image, separately for each domain and structure.

    This comparison needs two arms. GlobalFiLM is the arm SpFiLM has to beat:
    same backbone, same folds, same hyperparameters, with the conditioning layer
    the only difference, so a tie means the spatial idea is not what helped.
    Until a second arm has been run over the same locked folds there is nothing
    to pair against, and this refuses rather than returning a number.

    Pairing is only valid on identical image sets, so that is asserted here
    rather than trusted; the locked manifest is what makes it true.

    Wilcoxon signed-rank is the default because per-image Dice is bounded in
    [0, 1] and typically left-skewed with a clump at the ceiling, so the paired
    differences are often not normal. Choose ``method='ttest'`` only after
    inspecting the differences.

    Both arms must have averaged the same number of seeds. A five-seed mean and
    a three-seed mean are not the same estimator: the shorter arm carries more
    residual training noise per image, which inflates the spread of the
    differences and biases the comparison toward finding nothing. That is
    invisible in the output, so it is refused here rather than caveated.
    """

    if method not in PAIRED_METHODS:
        raise Stage3ReportError(
            f"Unknown paired method {method!r}; expected one of {list(PAIRED_METHODS)}"
        )
    if metric not in substrate.metrics:
        raise Stage3ReportError(
            f"The substrate carries {list(substrate.metrics)}, not {metric!r}"
        )
    available = substrate.arms
    missing = [arm for arm in (arm_a, arm_b) if arm not in available]
    if missing:
        raise Stage3ReportError(
            "A paired between-arm test needs two arms scored on the same locked "
            f"folds; {missing} not among the discovered arms {list(available)}. "
            "Only one conditioning arm has been run so far, so there is no "
            "second arm to pair against and no comparison to report."
        )
    if arm_a == arm_b:
        raise Stage3ReportError(
            f"arm_a and arm_b are both {arm_a!r}; a paired test needs two arms"
        )

    cells_a = substrate.cells(arm_a)
    cells_b = substrate.cells(arm_b)
    if cells_a != cells_b:
        raise Stage3ReportError(
            f"{arm_a} covers {[(d.value, s) for d, s in cells_a]} but {arm_b} "
            f"covers {[(d.value, s) for d, s in cells_b]}"
        )
    if not allow_unequal_seeds:
        mismatches = [
            (
                domain.value,
                substrate.seed_counts[(arm_a, domain)],
                substrate.seed_counts[(arm_b, domain)],
            )
            for domain in sorted({domain for domain, _structure in cells_a})
            if substrate.seed_counts[(arm_a, domain)]
            != substrate.seed_counts[(arm_b, domain)]
        ]
        if mismatches:
            raise Stage3ReportError(
                f"The paired arms averaged different seed counts in these "
                f"domains (domain, {arm_a}, {arm_b}): {mismatches}. Pairing "
                "different estimators biases the comparison toward the null; "
                "finish the missing runs, or pass --allow-unequal-seeds "
                "deliberately."
            )

    results: list[PairedTestResult] = []
    for domain, structure in cells_a:
        images_a = substrate.image_ids(arm_a, domain, structure)
        images_b = substrate.image_ids(arm_b, domain, structure)
        if images_a != images_b:
            only_a = sorted(set(images_a) - set(images_b))
            only_b = sorted(set(images_b) - set(images_a))
            raise Stage3ReportError(
                f"{domain.value} {structure}: the two arms were scored on "
                f"different images, so they cannot be paired "
                f"(only in {arm_a}: {only_a[:5]}, only in {arm_b}: {only_b[:5]})"
            )
        differences = np.array(
            [
                substrate.values[(arm_b, domain, structure, image_id)][metric]
                - substrate.values[(arm_a, domain, structure, image_id)][metric]
                for image_id in images_a
            ],
            dtype=float,
        )
        statistic, p_value, informative = _paired_statistic(differences, method)
        results.append(
            PairedTestResult(
                held_out_domain=domain,
                structure=structure,
                metric=metric,
                method=method,
                arm_a=arm_a,
                arm_b=arm_b,
                n_pairs=int(differences.size),
                n_informative_pairs=informative,
                seed_count_a=substrate.seed_counts[(arm_a, domain)],
                seed_count_b=substrate.seed_counts[(arm_b, domain)],
                median_difference=float(np.median(differences)),
                mean_difference=float(np.mean(differences)),
                statistic=statistic,
                p_value=p_value,
            )
        )

    adjusted = holm_adjust([result.p_value for result in results])
    return tuple(
        replace(result, p_value_holm=holm)
        for result, holm in zip(results, adjusted)
    )


def _paired_statistic(
    differences: np.ndarray, method: str
) -> tuple[float, float, int]:
    """Return the statistic, its p-value, and the pairs the test actually used.

    Signed-rank discards exact ties, so its effective sample size can be far
    below the number of images. Reporting only the image count would overstate
    the evidence behind the p-value, so the used count is returned alongside.
    """

    if differences.size < 2:
        raise Stage3ReportError(
            f"A paired test needs at least 2 pairs, got {differences.size}"
        )
    if method == "ttest":
        if np.all(differences == 0.0):
            # scipy correctly returns NaN for a zero-variance all-zero sample,
            # but the scientific conclusion here is exact equality in the
            # observed pairs and a two-sided p-value of one.
            return 0.0, 1.0, int(differences.size)
        outcome = stats.ttest_1samp(differences, popmean=0.0)
        return float(outcome.statistic), float(outcome.pvalue), int(differences.size)
    informative = int(np.count_nonzero(differences))
    if informative == 0:
        # Every image scored identically under both arms; signed-rank is
        # undefined with nothing to rank, and the honest answer is no effect.
        return 0.0, 1.0, 0
    if informative < 2:
        raise Stage3ReportError(
            "Signed-rank needs at least 2 non-tied pairs; only "
            f"{informative} of {differences.size} differences are non-zero. "
            "Inspect the differences and choose method='ttest' deliberately if "
            "that is the comparison you want."
        )
    outcome = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return float(outcome.statistic), float(outcome.pvalue), informative


def render_paired_table(
    results: Sequence[PairedTestResult], alpha: float = 0.05
) -> str:
    """Render p-values beside the median difference they belong next to.

    Statistical significance is not practical significance: with ~80 pairs a
    0.003 Dice gap can be significant and clinically meaningless, so the effect
    size is never printed without the p-value or the p-value without the effect.
    A p at or above alpha is a genuine finding, not a failure, and is labelled
    plainly rather than hedged.
    """

    if not results:
        return "no paired comparisons"
    first = results[0]
    lines = textwrap.wrap(
        f"paired per-image {first.metric} test "
        f"({first.arm_b} minus {first.arm_a}), {first.method}, seeds averaged "
        f"per image, Holm-adjusted across all {len(results)} tests",
        width=78,
        subsequent_indent="  ",
    )
    seed_counts = sorted(
        {result.seed_count_a for result in results}
        | {result.seed_count_b for result in results}
    )
    lines.append(
        "  each per-image value is a mean over "
        + (
            f"{seed_counts[0]} seeds"
            if len(seed_counts) == 1
            else f"{seed_counts} seeds -- the arms are not matched"
        )
    )
    lines.append(
        "  n = image pairs; used = pairs ranked (signed-rank drops exact ties)"
    )
    lines.append(
        f"  sig = Holm-adjusted p < {alpha:g}; 'no' is a genuine null result"
    )
    lines.append(
        f"  {'domain':<17}{'str':<5}{'n':>4}{'used':>5}"
        f"{'median d':>10}{'mean d':>10}{'p':>10}{'p(Holm)':>10} sig"
    )
    for result in results:
        holm = result.p_value_holm
        verdict = "yes" if holm is not None and holm < alpha else "no"
        lines.append(
            f"  {result.held_out_domain.value:<17}{result.structure:<5}"
            f"{result.n_pairs:>4}{result.n_informative_pairs:>5}"
            f"{result.median_difference:>10.5f}{result.mean_difference:>10.5f}"
            f"{result.p_value:>10.3g}"
            f"{(float('nan') if holm is None else holm):>10.3g} {verdict}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3Report:
    """Everything the aggregation established, ready to render or serialise."""

    runs: tuple[Stage3Run, ...]
    domain_reports: tuple[DomainStructureReport, ...]
    substrate: PairedSubstrate
    paired_results: tuple[PairedTestResult, ...]
    paired_note: str
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": LODO_PROTOCOL_NAME,
            "confidence_interval": (
                "95% Student-t interval over per-seed run means; it measures "
                "training-run reproducibility on locked data, not image-to-image "
                "variability"
            ),
            "arms": sorted({run.identity.arm for run in self.runs}),
            "run_count": len(self.runs),
            "warnings": list(self.warnings),
            "runs": [
                {
                    "arm": run.identity.arm,
                    "held_out_domain": run.identity.held_out_domain.value,
                    "run_seed": run.identity.run_seed,
                    "run_dir": str(run.run_dir),
                    "manifest_sha256": run.manifest_sha256,
                    "config_sha256": run.config_sha256,
                    "git_revision": run.git_revision,
                }
                for run in self.runs
            ],
            "per_domain": [report.as_dict() for report in self.domain_reports],
            "paired_test": {
                "note": self.paired_note,
                "results": [result.as_dict() for result in self.paired_results],
            },
        }


def _sha256(path: Path) -> str:
    """Digest a file the same way run_stage3_lodo.py digests the manifest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_identity(
    runs: Sequence[Stage3Run], manifest_path: str | Path
) -> None:
    """Prove the runs were scored against the manifest being validated against.

    Membership validation only proves each run covers the test partition of the
    manifest loaded here. If the runs were actually produced against a different
    manifest whose test folds happen to match, their train and validation
    partitions could differ and the report would describe an experiment that
    never ran. The runner records the digest it used, so this is checkable.
    """

    actual = _sha256(manifest_path)
    mismatched = [run for run in runs if run.manifest_sha256 != actual]
    if mismatched:
        listed = "\n".join(
            f"  {run.identity.label}: {run.manifest_sha256}"
            for run in mismatched
        )
        raise Stage3ReportError(
            f"{Path(manifest_path)} hashes to {actual} but the following run(s) "
            "record having been scored against a different manifest:\n"
            f"{listed}\n"
            "Report against the manifest the runs actually used, or re-run them "
            "against this one; the fold composition is not to be edited."
        )


def verify_grid_provenance(runs: Sequence[Stage3Run]) -> None:
    """Require one config and code revision within each seed-CI arm.

    A seed interval is interpretable as training-run variability only when seed
    is the intended changing factor. Mixing configs or revisions changes the
    procedure itself; a warning beside the resulting interval is too late.
    """

    for arm in sorted({run.identity.arm for run in runs}):
        arm_runs = [run for run in runs if run.identity.arm == arm]
        configs = sorted({run.config_sha256 for run in arm_runs})
        if len(configs) != 1:
            raise Stage3ReportError(
                f"Arm {arm!r} spans more than one config SHA-256: {configs}; "
                "a seed interval requires the same training procedure"
            )
        revisions = sorted({run.git_revision for run in arm_runs})
        if len(revisions) != 1:
            raise Stage3ReportError(
                f"Arm {arm!r} spans more than one git revision: {revisions}; "
                "a seed interval requires the same code"
            )


def _provenance_warnings(runs: Sequence[Stage3Run]) -> list[str]:
    warnings: list[str] = []
    dirty = [run for run in runs if run.git_tree_was_dirty]
    if dirty:
        warnings.append(
            f"{len(dirty)} run(s) were produced from a dirty working tree, so "
            "the recorded commit does not fully identify the code that ran: "
            + ", ".join(run.identity.label for run in dirty)
        )
    unavailable = [
        run for run in runs if run.git_revision == "unavailable"
    ]
    if unavailable:
        warnings.append(
            f"{len(unavailable)} run(s) do not record a usable git revision: "
            + ", ".join(run.identity.label for run in unavailable)
        )
    return warnings


def aggregate(
    roots: Iterable[str | Path],
    manifest_path: str | Path,
    *,
    skip_smoke: bool = False,
    expected_seeds: Sequence[int] | None = DEFAULT_EXPECTED_SEEDS,
    expected_domains: Sequence[Domain] | None = tuple(Domain),
    confidence: float = CONFIDENCE_LEVEL,
    paired_arms: tuple[str, str] | None = None,
    paired_metric: str = "dice",
    paired_method: str = "wilcoxon",
    allow_unequal_seeds: bool = False,
) -> Stage3Report:
    """Discover, validate, and aggregate every completed Stage 3 run."""

    try:
        manifest = load_lodo_manifest(manifest_path)
    except LodoManifestError as error:
        raise Stage3ReportError(f"Cannot use the locked manifest: {error}") from error

    runs = select_scientific_runs(
        discover_stage3_runs(roots), skip_smoke=skip_smoke
    )
    verify_manifest_identity(runs, manifest_path)
    verify_grid_provenance(runs)

    scores: dict[RunIdentity, tuple[PerImageScore, ...]] = {}
    summaries: dict[RunIdentity, Mapping[str, StructureSummary]] = {}
    for run in runs:
        scores[run.identity] = load_run_scores(run, manifest)
        summaries[run.identity] = verify_run_summary(run)

    domain_reports = build_domain_reports(
        runs,
        summaries,
        scores,
        expected_seeds=expected_seeds,
        expected_domains=expected_domains,
        confidence=confidence,
    )
    substrate = build_paired_substrate(runs, scores)

    paired_results: tuple[PairedTestResult, ...] = ()
    arms = substrate.arms
    if paired_arms is not None:
        paired_results = paired_arm_test(
            substrate,
            paired_arms[0],
            paired_arms[1],
            metric=paired_metric,
            method=paired_method,
            allow_unequal_seeds=allow_unequal_seeds,
        )
        paired_note = (
            f"{paired_arms[1]} - {paired_arms[0]} on per-image {paired_metric}, "
            f"{paired_method}, Holm-adjusted across {len(paired_results)} tests"
        )
    elif len(arms) < 2:
        paired_note = (
            "not run: the paired comparison needs two conditioning arms scored "
            f"on the same locked folds, and only {list(arms)} has been run. The "
            "substrate is built and ready; pass --paired-arms A B once the "
            "second arm exists."
        )
    else:
        paired_note = (
            f"not run: {list(arms)} were discovered; name the two to compare "
            "with --paired-arms A B"
        )

    return Stage3Report(
        runs=tuple(runs),
        domain_reports=domain_reports,
        substrate=substrate,
        paired_results=paired_results,
        paired_note=paired_note,
        warnings=tuple(_provenance_warnings(runs)),
    )


def write_report_csv(
    reports: Sequence[DomainStructureReport], csv_path: str | Path
) -> Path:
    """Write the per-domain table as one row per arm/domain/structure/metric."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "arm",
        "held_out_domain",
        "structure",
        "metric",
        "test_image_count",
        "seed_count",
        "mean",
        "std_across_seeds",
        "ci_low",
        "ci_high",
        "ci_half_width",
        "hd95_unit",
        "hd95_excluded_per_seed",
        "hd95_common_finite_count",
        "hd95_subset_is_common",
        "hd95_common_subset_mean",
        "hd95_common_subset_ci_low",
        "hd95_common_subset_ci_high",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            for metric in SEED_METRICS:
                interval = report.intervals.get(metric)
                writer.writerow(
                    {
                        "arm": report.arm,
                        "held_out_domain": report.held_out_domain.value,
                        "structure": report.structure,
                        "metric": metric,
                        "test_image_count": report.image_count,
                        "seed_count": len(report.seeds),
                        "mean": (
                            report.point_means.get(metric)
                            if interval is None
                            else interval.mean
                        ),
                        "std_across_seeds": (
                            "" if interval is None else interval.std
                        ),
                        "ci_low": "" if interval is None else interval.low,
                        "ci_high": "" if interval is None else interval.high,
                        "ci_half_width": (
                            "" if interval is None else interval.half_width
                        ),
                        "hd95_unit": (
                            report.hd95_unit if metric == "hd95" else ""
                        ),
                        "hd95_excluded_per_seed": (
                            " ".join(
                                str(value)
                                for value in report.hd95_excluded_counts
                            )
                            if metric == "hd95"
                            else ""
                        ),
                        "hd95_common_finite_count": (
                            report.hd95_common_finite_count
                            if metric == "hd95"
                            else ""
                        ),
                        "hd95_subset_is_common": (
                            report.hd95_subset_is_common
                            if metric == "hd95"
                            else ""
                        ),
                        "hd95_common_subset_mean": _common_field(
                            report, metric, "mean"
                        ),
                        "hd95_common_subset_ci_low": _common_field(
                            report, metric, "low"
                        ),
                        "hd95_common_subset_ci_high": _common_field(
                            report, metric, "high"
                        ),
                    }
                )
    return csv_path


def _common_field(
    report: DomainStructureReport, metric: str, field_name: str
) -> object:
    """One field of the common-subset HD95 interval, blank where it has none."""

    if metric != "hd95" or report.hd95_common_interval is None:
        return ""
    return getattr(report.hd95_common_interval, field_name)


# --------------------------------------------------------------------------
# The written report
# --------------------------------------------------------------------------


TODO = "<!-- TODO: written by hand; the tool does not infer this. -->"


def _source_config(runs: Sequence[Stage3Run]) -> Mapping[str, Any]:
    """The training settings the runs actually resolved, or an empty mapping."""

    for run in runs:
        path = run.run_dir / RESOLVED_CONFIG_NAME
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except Stage3ReportError:
            continue
        source = payload.get("source_config")
        if isinstance(source, Mapping):
            return source
    return {}


def _setting(config: Mapping[str, Any], key: str, default: str = "not recorded") -> str:
    value = config.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return f"**{value}**"


def _training_summary(run: Stage3Run) -> tuple[object, ...]:
    """Recorded checkpoint/training fields for the report's run-level table."""

    payload = _read_json(run.metrics_path)
    early = payload.get("early_stopping")
    would_stop = (
        early.get("would_have_stopped_at_epoch")
        if isinstance(early, Mapping)
        else None
    )
    return (
        payload.get("best_epoch"),
        payload.get("epochs_run"),
        payload.get("epochs_configured"),
        would_stop,
        payload.get("training_seconds"),
    )


def _report_value(value: object, *, seconds: bool = False) -> str:
    if value is None:
        return "—"
    if seconds:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "—"
    return str(value)


def _interval_cell(
    interval: SeedInterval | None, point: float | None, digits: int
) -> str:
    if interval is None:
        if point is None:
            return "n/a"
        return f"**{point:.{digits}f}** (one seed, no interval)"
    return (
        f"**{interval.mean:.{digits}f}** "
        f"[{interval.low:.{digits}f}, {interval.high:.{digits}f}]"
    )


def render_markdown_report(report: Stage3Report) -> str:
    """A Stage 3 report skeleton in the run_reports house style.

    Every figure is filled in from the aggregation, so no number is ever
    transcribed by hand. Every section that requires judgement -- the findings,
    the comparison against the literature -- is left as an explicit TODO rather
    than being invented, because a generated sentence about what a result means
    is exactly the kind of claim that must not appear unexamined in a thesis.
    """

    runs = report.runs
    arms = sorted({run.identity.arm for run in runs})
    seeds = sorted({run.identity.run_seed for run in runs})
    lines: list[str] = []
    add = lines.append

    title_suffix = (
        "plain U-Net"
        if arms == ["stage3_lodo_plain_unet"]
        else "configured model arms"
    )
    add(f"# Step 3 leave-one-domain-out report: {title_suffix}")
    add("")
    add(
        "**Evidence boundary.** Every figure below is computed by "
        "`aggregate_stage3.py` directly from the per-image metric CSVs of "
        f"{len(runs)} completed runs, validated against the locked fold "
        "manifest, and recomputed from those CSVs rather than read from any "
        "run's stored summary. No value is carried over from the Step 2 "
        "in-domain reports. Sections marked TODO require judgement the tool "
        "does not make.[^runs]"
    )
    add("")
    add("## 1. Objective")
    add("")
    if arms == ["stage3_lodo_plain_unet"]:
        add(
            "This run measures how the plain U-Net baseline degrades under "
            "acquisition shift, using leave-one-domain-out evaluation: for each "
            "held-out domain the model trains on the remaining domains only and "
            "is scored on the held-out domain's locked test partition. It "
            "establishes the unconditioned reference; by itself it provides no "
            "evidence about conditioning."
        )
    else:
        add(
            "This report covers the configured Stage 3 arms "
            f"({', '.join(arms)}) under locked leave-one-domain-out evaluation. "
            "For each held-out domain, each arm trains on the remaining domains "
            "and is scored on the same locked test partition."
        )
    add("")
    add(TODO + " state what this run was for in the thesis narrative.")
    add("")
    add("## 2. Dataset and split")
    add("")
    add(
        "Fold membership is fixed by "
        "`splits/lodo/lodo_manifest.json` and was not regenerated for this "
        f"report; each run records the digest `{runs[0].manifest_sha256}` and "
        "the aggregation refuses to report against any other manifest.[^manifest]"
    )
    add("")
    add("| Arm | Held-out domain | Train | Validation | Test | Seeds |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    domains = sorted(
        {run.identity.held_out_domain for run in runs},
        key=lambda domain: domain.value,
    )
    for arm in arms:
        for domain in domains:
            cell = [
                run
                for run in runs
                if run.identity.arm == arm
                and run.identity.held_out_domain == domain
            ]
            if not cell:
                continue
            counts = cell[0].locked_split_counts
            cell_seeds = sorted({run.identity.run_seed for run in cell})
            add(
                f"| {arm} | {domain.value} "
                f"| {counts.get('train', '—')} | {counts.get('val', '—')} "
                f"| **{counts.get('test', '—')}** "
                f"| {len(cell_seeds)} ({', '.join(str(s) for s in cell_seeds)}) |"
            )
    add("")
    add(
        "Source domains for each fold are the other three; the held-out "
        "domain contributes no training or validation image."
    )
    add("")
    add("## 3. Model and training setup")
    add("")
    for arm in arms:
        arm_runs = [run for run in runs if run.identity.arm == arm]
        config = _source_config(arm_runs)
        add(f"### Arm: `{arm}`")
        add("")
        add("| Parameter | Run setting |")
        add("| --- | --- |")
        add(f"| Configuration identity | `{arm}` |")
        add(f"| Base width | {_setting(config, 'base_channels')} |")
        add("| Outputs | Two masks: optic disc and optic cup |")
        add(f"| Epoch budget | {_setting(config, 'epochs')} |")
        add(f"| Batch size | {_setting(config, 'batch_size')} |")
        size = _setting(config, "image_size")
        add(f"| Input size | {size} x {size} |")
        add(f"| Initial learning rate | {_setting(config, 'learning_rate')} |")
        add(f"| Weight decay | {_setting(config, 'weight_decay')} |")
        add("| Loss | BCE plus soft Dice |")
        add(f"| Hard-mask threshold | {_setting(config, 'threshold')} |")
        flip = _setting(config, "horizontal_flip_probability")
        add(f"| Horizontal-flip probability | {flip} |")
        add(f"| Rotation | {_setting(config, 'rotation_degrees')} degrees |")
        add(f"| Brightness/contrast | {_setting(config, 'brightness_contrast')} |")
        add(f"| Early-stopping mode | {_setting(config, 'early_stopping_mode')} |")
        patience = _setting(config, "patience")
        add(
            f"| Patience / minimum epochs | {patience} / "
            f"{_setting(config, 'min_epochs')} |"
        )
        arm_seeds = sorted({run.identity.run_seed for run in arm_runs})
        add(f"| Seeds | **{', '.join(str(seed) for seed in arm_seeds)}** |")
        add("")
        add(
            TODO
            + " confirm the architecture/conditioning description for this "
            "configuration identity."
        )
        add("")
    add("")
    add("## 4. Per-epoch results")
    add("")
    add(
        "The validated summaries record the selected checkpoint and completed "
        "epoch count for every run. Full epoch trajectories remain in each "
        "run's `history.csv`; they are not pooled because each seed has its own "
        "optimisation path.[^runs]"
    )
    add("")
    add(
        "| Arm | Held-out domain | Seed | Best epoch | Epochs run / budget | "
        "Would-stop epoch | Training seconds |"
    )
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        best, epochs_run, epochs_budget, would_stop, seconds = _training_summary(run)
        add(
            f"| {run.identity.arm} | {run.identity.held_out_domain.value} "
            f"| {run.identity.run_seed} | {_report_value(best)} "
            f"| {_report_value(epochs_run)} / {_report_value(epochs_budget)} "
            f"| {_report_value(would_stop)} "
            f"| {_report_value(seconds, seconds=True)} |"
        )
    add("")
    add(TODO + " add selected per-epoch milestones only if the thesis needs them.")
    add("")
    add("## 5. Test results")
    add("")
    add(
        "Each multi-seed figure is the mean over the per-seed run means, with "
        "a 95% Student-t interval on n-1 degrees of freedom taken **over the "
        "seeds in that cell**. A one-seed cell is labelled as a point estimate "
        "and receives no interval. The interval measures how much the answer "
        "moves when the same locked data is retrained; it is not the "
        "image-to-image spread stored as `dice_std` inside each run.[^method]"
    )
    add("")
    for arm in arms:
        for domain in sorted(
            {r.held_out_domain for r in report.domain_reports if r.arm == arm},
            key=lambda d: d.value,
        ):
            cells = [
                r
                for r in report.domain_reports
                if r.arm == arm and r.held_out_domain == domain
            ]
            cells.sort(key=lambda c: CHANNEL_NAMES.index(c.structure))
            first = cells[0]
            heading = (
                domain.value
                if len(arms) == 1
                else f"{arm} — {domain.value}"
            )
            add(f"### {heading}")
            add("")
            add(
                f"{first.image_count} locked test images; "
                f"{len(first.seeds)} seeds "
                f"({', '.join(str(s) for s in first.seeds)})."
            )
            add("")
            add("| Structure | Dice (95% CI) | IoU (95% CI) | HD95 (95% CI) |")
            add("| --- | --- | --- | --- |")
            for cell in cells:
                columns = [
                    _interval_cell(
                        cell.intervals.get(metric),
                        cell.point_means.get(metric),
                        2 if metric == "hd95" else 4,
                    )
                    for metric in SEED_METRICS
                ]
                add(f"| {cell.structure} | " + " | ".join(columns) + " |")
            add("")
            add(f"HD95 is measured in **{first.hd95_unit}**.")
            add("")
            for cell in cells:
                add(f"- HD95 ({cell.structure}): {_hd95_note(cell)}")
                common = cell.hd95_common_interval
                if common is not None and not cell.hd95_subset_is_common:
                    add(
                        f"  Restricted to the {cell.hd95_common_finite_count} "
                        f"images finite in every seed, HD95 is "
                        f"**{common.mean:.2f}** "
                        f"[{common.low:.2f}, {common.high:.2f}]."
                    )
            add("")
    add("### Between-arm paired test")
    add("")
    if report.paired_results:
        first = report.paired_results[0]
        test_name = (
            "Wilcoxon signed-rank"
            if first.method == "wilcoxon"
            else "paired t-test"
        )
        add(
            f"Per-image {first.metric}, {first.arm_b} minus {first.arm_a}, "
            f"{test_name} on the seed-averaged per-image values, "
            f"Holm-adjusted across all {len(report.paired_results)} tests."
        )
        add("")
        add(
            "| Domain | Structure | Seeds A/B | n | Ranked | Median diff | p | "
            "p (Holm) |"
        )
        add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in report.paired_results:
            holm = (
                "—"
                if result.p_value_holm is None
                else format(result.p_value_holm, ".3g")
            )
            add(
                f"| {result.held_out_domain.value} | {result.structure} "
                f"| {result.seed_count_a}/{result.seed_count_b} "
                f"| {result.n_pairs} | {result.n_informative_pairs} "
                f"| **{result.median_difference:+.5f}** "
                f"| {result.p_value:.3g} "
                f"| {holm} |"
            )
        add("")
        add(
            "Statistical significance is not practical significance: the "
            "median difference is the effect size and is reported beside every "
            "p-value. A p at or above 0.05 is a non-significant result, not "
            "evidence that the arms are identical."
        )
    else:
        add(report.paired_note)
    add("")
    add("## 6. Findings")
    add("")
    add(TODO + " every finding below the surface of the table is yours to write.")
    add("")
    add("The aggregation established these facts, which the findings may draw on:")
    add("")
    for line in _report_facts(report):
        add(f"- {line}")
    add("")
    add("## 7. Sanity check against the literature")
    add("")
    add(
        TODO
        + " compare the per-domain figures against published LODO or "
        "cross-domain fundus segmentation results, noting that HD95 units and "
        "mask provenance differ between sources."
    )
    add("")
    add("## 8. Limitations and reproducibility")
    add("")
    for line in _report_limitations(report):
        add(f"- {line}")
    add("")
    add("| Reproducibility field | Value |")
    add("| --- | --- |")
    add(f"| Protocol | {LODO_PROTOCOL_NAME} |")
    add(f"| Runs aggregated | **{len(runs)}** |")
    add(f"| Arms | {', '.join(arms)} |")
    add(f"| Seeds | {', '.join(str(seed) for seed in seeds)} |")
    add(f"| Manifest SHA-256 | `{runs[0].manifest_sha256}` |")
    for arm in arms:
        revisions = sorted({r.git_revision for r in runs if r.identity.arm == arm})
        configs = sorted({r.config_sha256 for r in runs if r.identity.arm == arm})
        add(f"| Git revision ({arm}) | `{', '.join(revisions)}` |")
        add(f"| Config SHA-256 ({arm}) | `{', '.join(configs)}` |")
    frames_by_domain = {
        cell.held_out_domain: cell.metric_frame for cell in report.domain_reports
    }
    for domain, frame in sorted(
        frames_by_domain.items(), key=lambda item: item[0].value
    ):
        add(f"| Metric frame ({domain.value}) | {frame} |")
    add("")
    add("## 9. Appendix: run inventory")
    add("")
    add("| Arm | Held-out domain | Seed | Output directory |")
    add("| --- | --- | ---: | --- |")
    for run in runs:
        add(
            f"| {run.identity.arm} | {run.identity.held_out_domain.value} "
            f"| {run.identity.run_seed} | `{run.run_dir}` |"
        )
    add("")
    add(
        "[^manifest]: `splits/lodo/lodo_manifest.json`, exact bytes identified "
        "by the SHA-256 shown in the reproducibility table."
    )
    add("")
    add(
        "[^runs]: Each listed run's `test_metrics.json`, sibling "
        "`test_per_image_metrics.csv`, `resolved_stage3_config.json`, and "
        "`history.csv` where present."
    )
    add("")
    add(
        "[^method]: `STAGE3_REPORTING_PLAN.md`, seed-interval and paired-test "
        "contracts."
    )
    add("")
    return "\n".join(lines) + "\n"


def _report_facts(report: Stage3Report) -> list[str]:
    """Machine-checkable statements the findings section can safely rest on."""

    facts: list[str] = []
    for arm in sorted({run.identity.arm for run in report.runs}):
        for structure in CHANNEL_NAMES:
            scoped = [
                (cell, cell.intervals["dice"])
                for cell in report.domain_reports
                if cell.arm == arm
                and cell.structure == structure
                and cell.intervals.get("dice") is not None
            ]
            if not scoped:
                continue
            ordered = sorted(scoped, key=lambda item: item[1].mean)
            (worst, worst_dice), (best, best_dice) = ordered[0], ordered[-1]
            spread = best_dice.mean - worst_dice.mean
            facts.append(
                f"{arm}: {structure} Dice is lowest on "
                f"{worst.held_out_domain.value} ({worst_dice.mean:.4f}) and "
                f"highest on {best.held_out_domain.value} "
                f"({best_dice.mean:.4f}), a spread of {spread:.4f} Dice across "
                "held-out domains."
            )
            widest, widest_dice = max(scoped, key=lambda item: item[1].half_width)
            facts.append(
                f"{arm}: the widest seed interval for {structure} Dice is "
                f"{widest.held_out_domain.value} at "
                f"+/-{widest_dice.half_width:.4f}. This is a CI half-width, "
                "not the standard deviation across images."
            )
    shifting = [
        cell
        for cell in report.domain_reports
        if not cell.hd95_subset_is_common
        and max(cell.hd95_sample_counts, default=0) > 0
    ]
    if shifting:
        facts.append(
            f"{len(shifting)} of {len(report.domain_reports)} cells have an "
            "HD95 exclusion set that moves between seeds, so their headline "
            "HD95 means are not taken over a common image set."
        )
    by_domain = {
        cell.held_out_domain: cell.hd95_unit for cell in report.domain_reports
    }
    if len(set(by_domain.values())) > 1:
        listed = "; ".join(
            f"{domain.value} in {unit}"
            for domain, unit in sorted(
                by_domain.items(), key=lambda item: item[0].value
            )
        )
        facts.append(
            f"HD95 is not in one unit across the table: {listed}. It must not "
            "be compared across domains without converting."
        )
    return facts


def _report_limitations(report: Stage3Report) -> list[str]:
    """Limitations the aggregation can actually evidence, not boilerplate."""

    limitations: list[str] = []
    seed_counts = sorted({len(c.seeds) for c in report.domain_reports})
    if seed_counts != [len(DEFAULT_EXPECTED_SEEDS)]:
        limitations.append(
            f"The grid is not the full five seeds per cell: seed counts are "
            f"{seed_counts}. Intervals from different cells rest on different "
            "amounts of evidence and are not directly comparable."
        )
    if len({run.identity.arm for run in report.runs}) < 2:
        limitations.append(
            "Only one conditioning arm has been run, so no between-arm paired "
            "comparison is possible and nothing here speaks to whether spatial "
            "conditioning helps."
        )
    units = sorted({c.hd95_unit for c in report.domain_reports})
    if len(units) > 1:
        limitations.append(
            "HD95 is reported in more than one unit across held-out domains ("
            + ", ".join(units)
            + "), because RIM-ONE-DL is converted to native source pixels and "
            "the other domains are not. HD95 columns are within-domain only."
        )
    else:
        limitations.append(
            f"HD95 is reported in {units[0]} and is not in millimetres, so it "
            "is not comparable with published millimetre boundary errors."
        )
    shifting_hd95 = [
        cell
        for cell in report.domain_reports
        if not cell.hd95_subset_is_common
        and max(cell.hd95_sample_counts, default=0) > 0
    ]
    if shifting_hd95:
        limitations.append(
            "In some cells the images with a defined HD95 differ between "
            "seeds, so the headline HD95 mean averages means over different "
            "image sets; the common-subset figure is given alongside it."
        )
    for warning in report.warnings:
        limitations.append(f"Provenance: {warning}")
    limitations.append(
        "Each interval is over the training seeds in that cell on fixed data. "
        "It measures "
        "reproducibility of the training procedure, not generalisation to new "
        "fundus images, and it must not be read as a population interval."
    )
    limitations.append(TODO + " add dataset-specific and clinical limitations.")
    return limitations


def write_markdown_report(report: Stage3Report, path: str | Path) -> Path:
    """Write the report skeleton, creating its directory if it does not exist."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed Stage 3 LODO runs into per-domain tables with "
            "95% confidence intervals taken over seeds, and run the paired "
            "between-arm test on per-image Dice once two arms exist."
        )
    )
    parser.add_argument(
        "--runs",
        dest="roots",
        action="append",
        metavar="PATH",
        help=(
            "directory to search recursively for Stage 3 runs; repeatable "
            f"(default: {', '.join(str(root) for root in DEFAULT_RUN_ROOTS)})"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="locked LODO manifest that defines each fold's test partition",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help=(
            "exclude runs recording scientific_result=false instead of refusing "
            "to report at all"
        ),
    )
    parser.add_argument(
        "--expect-seeds",
        nargs="*",
        type=int,
        default=list(DEFAULT_EXPECTED_SEEDS),
        metavar="SEED",
        help=(
            "seeds every arm/domain cell must contain; pass with no values to "
            "report an incomplete grid deliberately"
        ),
    )
    parser.add_argument(
        "--expect-domains",
        nargs="*",
        default=[domain.value for domain in Domain],
        choices=[domain.value for domain in Domain],
        metavar="DOMAIN",
        help=(
            "held-out domains every arm must cover; pass with no values to "
            "report an incomplete grid deliberately"
        ),
    )
    parser.add_argument(
        "--paired-arms",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help=(
            "run the paired per-image test of CANDIDATE minus BASELINE, e.g. "
            "the GlobalFiLM and SpFiLM arm names"
        ),
    )
    parser.add_argument(
        "--paired-metric",
        default="dice",
        choices=("dice", "iou"),
        help="per-image metric to pair on (HD95 is excluded by design)",
    )
    parser.add_argument(
        "--paired-method",
        default="wilcoxon",
        choices=PAIRED_METHODS,
        help="signed-rank by default; the t-test assumes normal differences",
    )
    parser.add_argument(
        "--allow-unequal-seeds",
        action="store_true",
        help=(
            "pair two arms even when they averaged different numbers of seeds "
            "per image; off by default because it biases the test to the null"
        ),
    )
    parser.add_argument(
        "--report-out",
        metavar="PATH",
        help=(
            "write a Stage 3 report skeleton in the run_reports house style, "
            "with every figure filled in and the narrative sections marked TODO"
        ),
    )
    parser.add_argument(
        "--json-out",
        metavar="PATH",
        help="write the full report, including provenance, as JSON",
    )
    parser.add_argument(
        "--csv-out",
        metavar="PATH",
        help="write the per-domain table as CSV",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.roots if args.roots else [str(root) for root in DEFAULT_RUN_ROOTS]
    expected_seeds = args.expect_seeds if args.expect_seeds else None
    expected_domains = (
        tuple(Domain(value) for value in args.expect_domains)
        if args.expect_domains
        else None
    )
    paired_arms = tuple(args.paired_arms) if args.paired_arms else None

    try:
        report = aggregate(
            roots,
            args.manifest,
            skip_smoke=args.skip_smoke,
            expected_seeds=expected_seeds,
            expected_domains=expected_domains,
            paired_arms=paired_arms,  # type: ignore[arg-type]
            paired_metric=args.paired_metric,
            paired_method=args.paired_method,
            allow_unequal_seeds=args.allow_unequal_seeds,
        )
    except Stage3ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    arms = sorted({run.identity.arm for run in report.runs})
    print(
        "\n".join(
            textwrap.wrap(
                f"aggregated {len(report.runs)} Stage 3 run(s) across "
                f"{len(arms)} arm(s): {', '.join(arms)}",
                width=78,
                subsequent_indent="  ",
            )
        )
    )
    for warning in report.warnings:
        print(
            "\n".join(
                textwrap.wrap(
                    f"warning: {warning}",
                    width=78,
                    subsequent_indent="  ",
                )
            )
        )
    print(render_domain_table(report.domain_reports))
    print()
    if report.paired_results:
        print(render_paired_table(report.paired_results))
    else:
        print(
            "\n".join(
                textwrap.wrap(
                    f"paired between-arm test {report.paired_note}",
                    width=78,
                    subsequent_indent="  ",
                )
            )
        )

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {json_path}")
    if args.csv_out:
        print(f"wrote {write_report_csv(report.domain_reports, args.csv_out)}")
    if args.report_out:
        print(f"wrote {write_markdown_report(report, args.report_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
