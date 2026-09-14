#!/usr/bin/env python3
"""Put the Global FiLM train-on-one control next to the plain train-on-one arm.

Both arms train on one source domain's budgeted partition and score the other
active domains' budgeted test partitions, so for each (source, target) pair
they score identical images and differ in one thing: the FiLM arm adds
channel-wise FiLM to the encoder. With one source domain there is one code, so
the FiLM generators see a constant input and the modulation collapses to a
fixed per-channel affine per level. That makes this a *control*: the FiLM arm
must match the plain arm within seed noise. A significant difference here is a
bug to chase, not a finding.

The plain half is the existing Stage 3 single-source grid. Training on one
domain does not depend on the other domains, so a plain run's scores on the
two active targets are exactly "train on 1, test on the remaining 2" once its
RIM-ONE-DL column is set aside; plain is not re-run. This script reuses the
train-on-one aggregator's loading and the fixed-budget aggregator's paired
test, FiLM as the reference arm so a positive difference reads "FiLM helped",
Holm-adjusted over every (source, target, structure) cell at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from aggregate_stage3_1_3 import (  # noqa: E402
    DEFAULT_EXPECTED_SEEDS,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOTS,
    SingleSourceRun,
    Stage3SingleSourceReportError,
    _sha256,
    discover_runs,
    seed_confidence_interval,
    select_scientific_runs,
    verify_run_summary,
)
from aggregate_stage3_fixed import (  # noqa: E402
    ALPHA,
    PAIRED_METHODS,
    PairedResult,
    Substrate,
    _accumulate,
    holm_adjust,
    paired_tests,
)
from spfilm.lodo import Domain  # noqa: E402
from spfilm.metrics import CHANNEL_NAMES, read_per_image_csv  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    load_single_source_manifest,
)


DEFAULT_PLAIN_ARM = "stage3_single_source_plain_unet"
DEFAULT_FILM_ARM = "stage4_single_source_global_film_3dom"


class SingleSourceFilmReportError(Stage3SingleSourceReportError):
    """Raised when the two arms cannot be paired honestly."""


# --------------------------------------------------------------------------
# Loading: one arm at a time, each run checked against the manifest
# --------------------------------------------------------------------------


def _restrict(
    runs: Sequence[SingleSourceRun],
    seeds: Sequence[int],
    sources: Sequence[Domain] | None,
) -> tuple[SingleSourceRun, ...]:
    """Keep the requested seeds (and sources) so a partial FiLM grid pairs with plain."""

    wanted_seeds = set(seeds)
    wanted_sources = None if sources is None else set(sources)
    return tuple(
        run
        for run in runs
        if run.identity.run_seed in wanted_seeds
        and (wanted_sources is None or run.identity.source_domain in wanted_sources)
    )


def verify_target_membership(
    run: SingleSourceRun, manifest: SingleSourceManifest
) -> dict[Domain, int]:
    """Prove each scored target is a locked target and its CSV holds exactly that partition.

    Unlike the Stage 3 aggregator's check this allows a run to score a *subset*
    of the fold's targets: the manifest lists every other domain, and a protocol
    that drops one simply does not score it.
    """

    fold = next(
        (f for f in manifest.folds if f.source_domain == run.identity.source_domain),
        None,
    )
    if fold is None:
        raise SingleSourceFilmReportError(
            f"{run.label}: the manifest has no fold for this source domain"
        )
    unknown = sorted(set(run.target_domains) - set(fold.target_domains), key=lambda d: d.value)
    if unknown:
        raise SingleSourceFilmReportError(
            f"{run.label}: scored {[d.value for d in unknown]}, which the manifest "
            "does not list as a target of this source"
        )
    counts: dict[Domain, int] = {}
    for domain in run.target_domains:
        expected = {sample.sample_id for sample in fold.test_samples(domain)}
        scored = {str(row["image_id"]) for row in read_per_image_csv(run.per_image_csv[domain])}
        if scored != expected:
            raise SingleSourceFilmReportError(
                f"{run.label} {domain.value}: scored images do not match the locked "
                f"test partition (unexpected={sorted(scored - expected)[:5]}, "
                f"missing={sorted(expected - scored)[:5]})"
            )
        counts[domain] = len(expected)
    return counts


def load_arm(
    roots: Sequence[Path],
    arm: str,
    expected_seeds: Sequence[int],
    manifest: SingleSourceManifest,
    manifest_path: Path,
    sources: Sequence[Domain] | None = None,
) -> tuple[SingleSourceRun, ...]:
    runs = select_scientific_runs(
        _restrict(discover_runs(roots), expected_seeds, sources),
        tuple(expected_seeds),
        arm=arm,
    )
    digest = _sha256(manifest_path)
    if runs[0].identity.manifest_sha256 != digest:
        raise SingleSourceFilmReportError(
            f"{arm} runs were produced against a different manifest than "
            f"{manifest_path}: runs say {runs[0].identity.manifest_sha256[:12]}…, "
            f"this file is {digest[:12]}…"
        )
    for run in runs:
        verify_target_membership(run, manifest)
        verify_run_summary(run)
    return runs


# --------------------------------------------------------------------------
# Pairing: which (source, target) cells both arms scored on identical images
# --------------------------------------------------------------------------


def _metadata(run: SingleSourceRun) -> dict[str, Any]:
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    block = payload.get("single_source")
    if not isinstance(block, dict):
        raise SingleSourceFilmReportError(f"{run.label}: no single_source block")
    return block


def film_active_domains(film_runs: Sequence[SingleSourceRun]) -> tuple[Domain, ...]:
    """The FiLM protocol's active set, which every FiLM run must agree on."""

    active_sets: set[tuple[str, ...]] = set()
    for run in film_runs:
        metadata = _metadata(run)
        recorded = metadata.get("active_domains")
        if recorded is None:
            # Older metadata: the active set is whatever the run touched.
            recorded = sorted(
                {run.identity.source_domain.value, *(d.value for d in run.target_domains)}
            )
        active_sets.add(tuple(sorted(str(value) for value in recorded)))
    if len(active_sets) != 1:
        raise SingleSourceFilmReportError(
            f"FiLM runs disagree on the active domain set: {sorted(active_sets)}"
        )
    return tuple(Domain(value) for value in next(iter(active_sets)))


def paired_targets(
    plain_runs: Sequence[SingleSourceRun],
    film_runs: Sequence[SingleSourceRun],
) -> dict[Domain, tuple[Domain, ...]]:
    """Per source, the targets both arms scored, after dropping the inactive ones.

    The plain arm ran the four-domain protocol and scored three targets; the
    FiLM arm ran the three-domain one and scored two. Restricting the plain
    targets to the FiLM arm's active set must give exactly the FiLM targets, or
    the two arms are not scoring the same images and there is nothing to pair.
    """

    active = set(film_active_domains(film_runs))
    plain_targets = {
        run.identity.source_domain: frozenset(run.target_domains) for run in plain_runs
    }
    film_targets = {
        run.identity.source_domain: frozenset(run.target_domains) for run in film_runs
    }
    for run in plain_runs:
        if frozenset(run.target_domains) != plain_targets[run.identity.source_domain]:
            raise SingleSourceFilmReportError(
                f"{run.label}: plain seeds disagree on the target set"
            )
    for run in film_runs:
        if frozenset(run.target_domains) != film_targets[run.identity.source_domain]:
            raise SingleSourceFilmReportError(
                f"{run.label}: FiLM seeds disagree on the target set"
            )
    missing = sorted(set(film_targets) - set(plain_targets), key=lambda d: d.value)
    if missing:
        raise SingleSourceFilmReportError(
            "The plain arm has no runs for FiLM source(s) "
            f"{[d.value for d in missing]}; nothing to pair them with"
        )
    pairs: dict[Domain, tuple[Domain, ...]] = {}
    for source in sorted(film_targets, key=lambda d: d.value):
        plain_active = {d for d in plain_targets[source] if d in active}
        if plain_active != set(film_targets[source]):
            raise SingleSourceFilmReportError(
                f"{source.value}: after dropping inactive domains the plain arm scored "
                f"{sorted(d.value for d in plain_active)} but the FiLM arm scored "
                f"{sorted(d.value for d in film_targets[source])}; the target sets "
                "must match to pair"
            )
        pairs[source] = tuple(sorted(film_targets[source], key=lambda d: d.value))
    return pairs


def build_source_substrate(
    plain_runs: Sequence[SingleSourceRun],
    film_runs: Sequence[SingleSourceRun],
    targets: Sequence[Domain],
) -> Substrate:
    """Both arms for one source, keyed by target, seeds averaged per image."""

    store: dict[tuple[str, Domain, str, str], dict[str, list[float]]] = {}
    seeds: dict[tuple[str, Domain], set[int]] = {}
    for run in (*plain_runs, *film_runs):
        for domain in targets:
            _accumulate(
                store,
                seeds,
                run.identity.arm,
                domain,
                run.identity.run_seed,
                run.per_image_csv[domain],
            )
    values: dict[tuple[str, Domain, str, str], Mapping[str, float]] = {}
    for key, bucket in store.items():
        expected = len(seeds[(key[0], key[1])])
        for metric, samples in bucket.items():
            if len(samples) != expected:
                raise SingleSourceFilmReportError(
                    f"{key[0]}/{key[1].value} {key[2]} image {key[3]!r} has "
                    f"{len(samples)} {metric} values but the arm ran {expected} seeds"
                )
        values[key] = {m: float(np.mean(v)) for m, v in bucket.items()}
    return Substrate(seed_counts={c: len(v) for c, v in seeds.items()}, values=values)


@dataclass(frozen=True)
class SourcePairedResult:
    """One paired test, tagged with the source domain the pair was trained on."""

    source_domain: Domain
    result: PairedResult

    @property
    def target_domain(self) -> Domain:
        return self.result.held_out_domain


def paired_tests_per_source(
    plain_runs: Sequence[SingleSourceRun],
    film_runs: Sequence[SingleSourceRun],
    pairs: Mapping[Domain, Sequence[Domain]],
    film_arm: str,
    method: str = "wilcoxon",
) -> tuple[SourcePairedResult, ...]:
    """FiLM minus plain per (source, target, structure), one Holm family over all.

    ``paired_tests`` compares its reference arm with every other arm on the
    same domain, so it is called once per source with a substrate that holds
    only that source's two arms; a plain model trained on another source must
    never be paired with this FiLM model. Each call adjusts its own four
    p-values, which is then undone: the family is every cell in the report
    (3 sources x 2 targets x 2 structures = 12), adjusted together.
    """

    tagged: list[SourcePairedResult] = []
    for source in sorted(pairs, key=lambda d: d.value):
        substrate = build_source_substrate(
            [r for r in plain_runs if r.identity.source_domain == source],
            [r for r in film_runs if r.identity.source_domain == source],
            pairs[source],
        )
        for result in paired_tests(substrate, method=method, reference_arm=film_arm):
            tagged.append(SourcePairedResult(source_domain=source, result=result))
    adjusted = holm_adjust([item.result.p_value for item in tagged])
    return tuple(
        SourcePairedResult(
            source_domain=item.source_domain,
            result=replace(item.result, p_adjusted=p_adj, significant=p_adj < ALPHA),
        )
        for item, p_adj in zip(tagged, adjusted)
    )


# --------------------------------------------------------------------------
# Per-cell seed summaries for each arm
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmCell:
    """One arm's Dice for one (source, target, structure), reduced over seeds.

    A confidence interval needs at least two seeds; the first real check of
    this control is one seed against the plain grid's matching seed, so the
    spread is ``None`` rather than refused when there is only one.
    """

    arm: str
    source_domain: Domain
    target_domain: Domain
    structure: str
    test_image_count: int
    seeds: tuple[int, ...]
    seed_means: tuple[float, ...]
    mean: float
    std: float | None


def build_arm_cells(
    runs: Sequence[SingleSourceRun], pairs: Mapping[Domain, Sequence[Domain]]
) -> tuple[ArmCell, ...]:
    cells: list[ArmCell] = []
    for source in sorted(pairs, key=lambda d: d.value):
        source_runs = sorted(
            (r for r in runs if r.identity.source_domain == source),
            key=lambda r: r.identity.run_seed,
        )
        if not source_runs:
            continue
        seeds = [r.identity.run_seed for r in source_runs]
        for target in pairs[source]:
            for structure in CHANNEL_NAMES:
                means: list[float] = []
                counts: set[int] = set()
                for run in source_runs:
                    rows = [
                        row
                        for row in read_per_image_csv(run.per_image_csv[target])
                        if row["structure"] == structure
                    ]
                    if not rows:
                        raise SingleSourceFilmReportError(
                            f"{run.label}: no {structure} rows for {target.value}"
                        )
                    means.append(float(np.mean([float(r["dice"]) for r in rows])))
                    counts.add(len(rows))
                if len(counts) != 1:
                    raise SingleSourceFilmReportError(
                        f"{source.value} -> {target.value} {structure}: seeds scored "
                        f"different image counts {sorted(counts)}"
                    )
                cells.append(
                    ArmCell(
                        arm=source_runs[0].identity.arm,
                        source_domain=source,
                        target_domain=target,
                        structure=structure,
                        test_image_count=next(iter(counts)),
                        seeds=tuple(seeds),
                        seed_means=tuple(means),
                        mean=float(np.mean(means)),
                        std=(
                            seed_confidence_interval("dice", seeds, means).std
                            if len(means) >= 2
                            else None
                        ),
                    )
                )
    return tuple(cells)


# --------------------------------------------------------------------------
# The degeneracy check: one code, and the run knew it
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DegeneracyCell:
    source_domain: Domain
    seeds: tuple[int, ...]
    vocabulary: tuple[str, ...]
    selector_val_accuracy: float
    fixed_code_entries: int
    targets_assigned_to_source: Mapping[str, float]  # target -> fraction over seeds


def build_degeneracy_cells(film_runs: Sequence[SingleSourceRun]) -> tuple[DegeneracyCell, ...]:
    """Confirm each FiLM run is the one-code control it claims to be.

    The vocabulary must be exactly the source domain, the run must have stamped
    ``degenerate_conditioning``, the fixed-code sweep must have one entry, and
    every target image must have been assigned the one code. Anything else
    means the run was not produced by the train-on-one runner as intended.
    """

    cells: list[DegeneracyCell] = []
    for source in sorted({r.identity.source_domain for r in film_runs}, key=lambda d: d.value):
        runs = sorted((r for r in film_runs if r.identity.source_domain == source), key=lambda r: r.identity.run_seed)
        vocabularies: set[tuple[str, ...]] = set()
        accuracies: list[float] = []
        sweep_sizes: set[int] = set()
        assigned: dict[str, list[float]] = {}
        for run in runs:
            payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
            metadata = payload["single_source"]
            if metadata.get("degenerate_conditioning") is not True:
                raise SingleSourceFilmReportError(
                    f"{run.label} ({run.identity.arm}) is not stamped "
                    "degenerate_conditioning; was it produced by the train-on-one runner?"
                )
            conditioning = payload.get("conditioning")
            if not isinstance(conditioning, dict):
                raise SingleSourceFilmReportError(
                    f"{run.label} ({run.identity.arm}) has no conditioning block; is it a FiLM run?"
                )
            vocabulary = tuple(conditioning["vocabulary"])
            if vocabulary != (source.value,):
                raise SingleSourceFilmReportError(
                    f"{run.label}: a train-on-one model must have exactly one code, "
                    f"its source {source.value!r}; got {list(vocabulary)}"
                )
            vocabularies.add(vocabulary)
            accuracies.append(float(conditioning["selector_validation"]["accuracy"]))
            sweep_sizes.add(len(conditioning["fixed_code_sweep"]))
            for target in run.target_domains:
                block = payload["test_by_domain"][target.value]["conditioning"]
                counts = block["assignment_counts"]
                total = sum(int(v) for v in counts.values())
                assigned.setdefault(target.value, []).append(
                    int(counts.get(source.value, 0)) / total if total else float("nan")
                )
                if block.get("true_domains_in_vocabulary") is not False:
                    raise SingleSourceFilmReportError(
                        f"{run.label} {target.value}: an unseen target cannot be in the "
                        "one-domain vocabulary, yet true_domains_in_vocabulary is not False"
                    )
        if sweep_sizes != {1}:
            raise SingleSourceFilmReportError(
                f"{source.value}: the fixed-code sweep should have one entry, got {sorted(sweep_sizes)}"
            )
        cells.append(
            DegeneracyCell(
                source_domain=source,
                seeds=tuple(r.identity.run_seed for r in runs),
                vocabulary=next(iter(vocabularies)),
                selector_val_accuracy=float(np.mean(accuracies)),
                fixed_code_entries=1,
                targets_assigned_to_source={t: float(np.mean(v)) for t, v in assigned.items()},
            )
        )
    return tuple(cells)


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_side_by_side(
    plain_cells: Sequence[ArmCell],
    film_cells: Sequence[ArmCell],
    results: Sequence[SourcePairedResult],
    plain_arm: str,
    film_arm: str,
) -> str:
    plain = {(c.source_domain, c.target_domain, c.structure): c for c in plain_cells}
    film = {(c.source_domain, c.target_domain, c.structure): c for c in film_cells}
    tests = {(r.source_domain, r.target_domain, r.result.structure): r.result for r in results}
    lines = [
        "| Trained on | Tested on | Structure | Images | Plain Dice, mean ± seed SD | "
        "FiLM Dice, mean ± seed SD | Δ (FiLM − plain) | p | p (Holm) | Significant |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    def fmt(cell: ArmCell | None) -> str:
        if cell is None:
            return "—"
        if cell.std is None:
            return f"{cell.mean:.4f} (1 seed)"
        return f"{cell.mean:.4f} ± {cell.std:.4f}"

    for key in sorted(set(plain) | set(film), key=lambda k: (k[0].value, k[1].value, CHANNEL_NAMES.index(k[2]))):
        source, target, structure = key
        p, f, t = plain.get(key), film.get(key), tests.get(key)
        images = (p or f).test_image_count if (p or f) else 0
        if t is None:
            delta, pv, ph, sig = "—", "—", "—", "—"
        else:
            delta = f"{t.mean_difference:+.4f}"
            pv, ph = f"{t.p_value:.4g}", f"{t.p_adjusted:.4g}"
            sig = "**yes**" if t.significant else "no"
        lines.append(
            f"| `{source.value}` | `{target.value}` | {structure} | {images} | {fmt(p)} | "
            f"{fmt(f)} | {delta} | {pv} | {ph} | {sig} |"
        )
    lines.append("")
    lines.append(f"Plain arm: `{plain_arm}`. FiLM arm: `{film_arm}`.")
    return "\n".join(lines)


def render_degeneracy_table(cells: Sequence[DegeneracyCell]) -> str:
    lines = [
        "| Trained on | Seeds | Codes | Selector val. accuracy | Fixed-code sweep entries | "
        "Target images given the one code |",
        "|---|---:|---|---:|---:|---|",
    ]
    for c in cells:
        assigned = ", ".join(
            f"`{target}`: {fraction:.0%}" for target, fraction in sorted(c.targets_assigned_to_source.items())
        )
        lines.append(
            f"| `{c.source_domain.value}` | {len(c.seeds)} | {', '.join(f'`{v}`' for v in c.vocabulary)} | "
            f"{c.selector_val_accuracy:.3f} | {c.fixed_code_entries} | {assigned} |"
        )
    return "\n".join(lines)


def render_markdown_report(
    plain_cells: Sequence[ArmCell],
    film_cells: Sequence[ArmCell],
    results: Sequence[SourcePairedResult],
    degeneracy: Sequence[DegeneracyCell],
    plain_runs: Sequence[SingleSourceRun],
    film_runs: Sequence[SingleSourceRun],
    manifest_path: Path,
    method: str,
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Step 4 control: Global FiLM with one code against the plain U-Net, train on one, test on two")
    add("")
    add(
        "**Evidence boundary.** Every figure is computed by "
        "`aggregate_stage4_single_source_film.py` from the per-image metric CSVs of "
        f"{len(plain_runs)} plain and {len(film_runs)} Global FiLM runs, validated "
        f"against the shared budgeted manifest `{manifest_path.name}`. "
        "Interpretation is written by hand."
    )
    add("")
    active = sorted(d.value for d in film_active_domains(film_runs))
    budget = film_runs[0].identity
    add("## 1. What this can and cannot show")
    add("")
    add(
        "With one source domain there is one code. The FiLM generators receive a "
        "constant input, so `(1+γ)·F+β` is a fixed per-channel affine per encoder "
        "level, which `InstanceNorm2d(affine=True)` already provides; the "
        "nearest-domain selector has one candidate and the fixed-code sweep one "
        "entry. **The expected result is no difference.** A significant cell is a "
        "bug to chase, not a finding. This is the sanity control for the FiLM "
        "block; the conditioning test is the leave-one-domain-out comparison in "
        "`aggregate_stage4_film.py`."
    )
    add("")
    add("## 2. Target Dice side by side")
    add("")
    add(
        f"Train on one, test on the other two, over {len(active)} domains "
        f"(`{'`, `'.join(active)}`): {budget.train_budget} / {budget.val_budget} source "
        f"images, {budget.test_budget} images per target. The plain half is the "
        "Stage 3 single-source grid with its RIM-ONE-DL column set aside; training "
        "on one domain does not depend on the others, so it was not re-run. Same "
        "backbone, source images, budget, seeds, augmentation, optimiser and test "
        "images; the arms differ only in the FiLM block. Δ is FiLM minus plain on "
        f"per-image Dice with seeds averaged first; p-values are {method}, "
        f"Holm-adjusted over {len(results)} tests, significance at α = {ALPHA}."
    )
    add("")
    add(render_side_by_side(plain_cells, film_cells, results, plain_runs[0].identity.arm, film_runs[0].identity.arm))
    add("")
    add("## 3. Was it really one code?")
    add("")
    add(
        "Selector accuracy on source validation images is 100% by construction "
        "(one candidate). Every target image must have been given the source's code."
    )
    add("")
    add(render_degeneracy_table(degeneracy))
    add("")
    add("## 4. Findings")
    add("")
    add("<!-- TODO: written by hand; the tool does not infer this. -->")
    add("")
    return "\n".join(lines)


def write_csv(
    results: Sequence[SourcePairedResult],
    degeneracy: Sequence[DegeneracyCell],
    path: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for item in results:
        r = item.result
        rows.append(
            {
                "kind": "paired_test",
                "source_domain": item.source_domain.value,
                "target_domain": item.target_domain.value,
                "structure": r.structure,
                "arm_plain": r.arm_a,
                "arm_film": r.arm_b,
                "test_images": r.image_count,
                "dice_mean_plain": f"{r.mean_a:.6g}",
                "dice_mean_film": f"{r.mean_b:.6g}",
                "mean_difference": f"{r.mean_difference:.6g}",
                "median_difference": f"{r.median_difference:.6g}",
                "method": r.method,
                "p_value": f"{r.p_value:.6g}",
                "p_adjusted": f"{r.p_adjusted:.6g}",
                "significant": r.significant,
            }
        )
    for c in degeneracy:
        row: dict[str, Any] = {
            "kind": "degeneracy",
            "source_domain": c.source_domain.value,
            "seeds": " ".join(str(s) for s in c.seeds),
            "codes": " ".join(c.vocabulary),
            "selector_val_accuracy": f"{c.selector_val_accuracy:.6g}",
            "fixed_code_entries": c.fixed_code_entries,
        }
        for target, fraction in sorted(c.targets_assigned_to_source.items()):
            row[f"assigned_to_source_{target}"] = f"{fraction:.6g}"
        rows.append(row)
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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, action="append")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plain-arm", default=DEFAULT_PLAIN_ARM)
    parser.add_argument("--film-arm", default=DEFAULT_FILM_ARM)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPECTED_SEEDS),
        help="Seeds both arms must have; pass a subset to compare a partial FiLM grid",
    )
    parser.add_argument("--method", choices=PAIRED_METHODS, default="wilcoxon")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(r) for r in (args.run_root or list(DEFAULT_RUN_ROOTS))]
    try:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = load_single_source_manifest(manifest_path)
        film = load_arm(roots, args.film_arm, args.expected_seeds, manifest, manifest_path)
        # Only the FiLM arm's sources are needed from plain; an incomplete
        # plain grid elsewhere (RIM-ONE-DL as source) must not block the pairing.
        plain = load_arm(
            roots,
            args.plain_arm,
            args.expected_seeds,
            manifest,
            manifest_path,
            sources=sorted({r.identity.source_domain for r in film}, key=lambda d: d.value),
        )
        pairs = paired_targets(plain, film)
        plain_cells = build_arm_cells(plain, pairs)
        film_cells = build_arm_cells(film, pairs)
        results = paired_tests_per_source(plain, film, pairs, args.film_arm, method=args.method)
        degeneracy = build_degeneracy_cells(film)
    except (Stage3SingleSourceReportError, OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    print(
        f"plain: {len(plain)} runs | film: {len(film)} runs | paired tests: {len(results)} | "
        f"expected: no significant cell (one code is a fixed affine)"
    )
    print()
    print(render_side_by_side(plain_cells, film_cells, results, args.plain_arm, args.film_arm))
    print()
    print(render_degeneracy_table(degeneracy))
    if args.report_out is not None:
        report = render_markdown_report(
            plain_cells, film_cells, results, degeneracy, plain, film, manifest_path, args.method
        )
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.report_out}")
    if args.csv_out is not None:
        print(f"wrote {write_csv(results, degeneracy, args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
