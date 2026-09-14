#!/usr/bin/env python3
"""Aggregate the train-on-all arms: FiLM next to plain, per domain, plus the wrong-code penalty.

One model per seed is trained on all three active domains and scored on each
domain's own 50 test images (plain arm: no codes; FiLM arm: the domain's own
code). This puts the two arms side by side per domain, runs the brief's paired
test on per-image Dice with seeds averaged first (FiLM minus plain, so a positive
difference reads "FiLM helped"), and reduces the FiLM runs' fixed-code sweeps
into the wrong-code penalty: how much Dice the FiLM model loses when a test
image is given another domain's code. A penalty near zero means the network
ignores the code; a clear penalty means it uses it, which is the precondition
for any conditioning result to mean anything.

Reuses the fixed-budget aggregator's statistics; only run discovery and the
sweep reduction are new.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from aggregate_stage3_1_3 import SeedInterval, seed_confidence_interval  # noqa: E402
from aggregate_stage3_fixed import (  # noqa: E402
    ALPHA,
    DEFAULT_EXPECTED_SEEDS,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOTS,
    PAIRED_METHODS,
    FixedLodoReportError,
    PairedResult,
    Substrate,
    _accumulate,
    _sha256,
    paired_tests,
)
from spfilm.all_domains import ALL_DOMAINS_PROTOCOL_NAME, compose_all_domains_fold  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.metrics import CHANNEL_NAMES, read_per_image_csv  # noqa: E402
from spfilm.single_source import load_single_source_manifest  # noqa: E402


TEST_METRICS_NAME = "test_metrics.json"
DEFAULT_PLAIN_ARM = "stage4_all_domains_fixed_budget_plain_unet_3dom"
DEFAULT_FILM_ARM = "stage4_all_domains_fixed_budget_global_film_3dom"


# --------------------------------------------------------------------------
# Stage A: load
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AllDomainsRun:
    arm: str
    conditioning_arm: str
    run_seed: int
    active_domains: tuple[str, ...]
    manifest_sha256: str
    completed_at_utc: str
    train_budget: int
    val_budget: int
    test_budget: int | None
    directory: Path
    metrics_path: Path
    per_image_csv: Mapping[Domain, Path]
    test_by_domain: Mapping[str, Any]

    @property
    def label(self) -> str:
        return f"{self.arm}/seed_{self.run_seed}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedLodoReportError(f"Cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FixedLodoReportError(f"{path} must hold a JSON object")
    return payload


def build_run(metrics_path: Path) -> AllDomainsRun | None:
    payload = _read_json(metrics_path)
    metadata = payload.get("all_domains")
    if not isinstance(metadata, dict) or metadata.get("protocol") != ALL_DOMAINS_PROTOCOL_NAME:
        return None
    if metadata.get("smoke_rehearsal") is True:
        return None
    if metadata.get("scientific_result") is not True:
        raise FixedLodoReportError(f"{metrics_path} is not marked scientific_result")
    test_by_domain = payload.get("test_by_domain")
    if not isinstance(test_by_domain, dict):
        raise FixedLodoReportError(f"{metrics_path} has no test_by_domain block")
    active = tuple(str(d) for d in metadata["active_domains"])
    per_image: dict[Domain, Path] = {}
    for domain in active:
        csv_path = metrics_path.parent / f"test_{domain}_per_image_metrics.csv"
        if not csv_path.is_file():
            raise FixedLodoReportError(f"{metrics_path} is missing {csv_path.name}")
        per_image[Domain(domain)] = csv_path
    budget = metadata.get("budget") or {}
    return AllDomainsRun(
        arm=str(metadata["arm"]),
        conditioning_arm=str(metadata.get("conditioning_arm", "plain")),
        run_seed=int(metadata["run_seed"]),
        active_domains=active,
        manifest_sha256=str(metadata["manifest_sha256"]),
        completed_at_utc=str(metadata["completed_at_utc"]),
        train_budget=int(budget.get("train", 0)),
        val_budget=int(budget.get("val", 0)),
        test_budget=None if budget.get("test") is None else int(budget["test"]),
        directory=metrics_path.parent,
        metrics_path=metrics_path,
        per_image_csv=per_image,
        test_by_domain=test_by_domain,
    )


def discover_runs(roots: Iterable[str | Path]) -> tuple[AllDomainsRun, ...]:
    found: list[AllDomainsRun] = []
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FixedLodoReportError(f"Run root is not a directory: {root_path}")
        for metrics_path in sorted(root_path.rglob(TEST_METRICS_NAME)):
            run = build_run(metrics_path)
            if run is not None:
                found.append(run)
    return tuple(sorted(found, key=lambda r: (r.arm, r.run_seed, r.completed_at_utc)))


def select_runs(
    runs: Sequence[AllDomainsRun], arm: str, expected_seeds: Sequence[int]
) -> tuple[AllDomainsRun, ...]:
    """One run per seed for one arm; the seed grid must be complete."""

    candidates = [run for run in runs if run.arm == arm and run.run_seed in set(expected_seeds)]
    if not candidates:
        raise FixedLodoReportError(f"No scientific train-on-all runs found for arm {arm!r}")
    by_seed: dict[int, AllDomainsRun] = {}
    for run in candidates:
        previous = by_seed.get(run.run_seed)
        if previous is None or run.completed_at_utc > previous.completed_at_utc:
            by_seed[run.run_seed] = run
    selected = tuple(by_seed[seed] for seed in sorted(by_seed))
    missing = [seed for seed in expected_seeds if seed not in by_seed]
    if missing:
        raise FixedLodoReportError(f"{arm}: missing seeds {missing}")
    if len({r.manifest_sha256 for r in selected}) != 1:
        raise FixedLodoReportError(f"{arm}: runs disagree on manifest_sha256")
    if len({r.active_domains for r in selected}) != 1:
        raise FixedLodoReportError(f"{arm}: runs disagree on the active domain set")
    return selected


def verify_membership(runs: Sequence[AllDomainsRun], manifest_path: Path) -> None:
    digest = _sha256(manifest_path)
    manifest = load_single_source_manifest(manifest_path)
    for run in runs:
        if run.manifest_sha256 != digest:
            raise FixedLodoReportError(
                f"{run.label} was produced against a different manifest than {manifest_path}"
            )
        fold = compose_all_domains_fold(
            manifest.budgeted_partitions, [Domain(d) for d in run.active_domains]
        )
        for domain, keys in fold.tests:
            expected = {k.sample_id for k in keys}
            scored = {str(r["image_id"]) for r in read_per_image_csv(run.per_image_csv[domain])}
            if scored != expected:
                raise FixedLodoReportError(
                    f"{run.label} {domain.value}: scored images do not match the locked "
                    f"test partition (unexpected={sorted(scored - expected)[:5]}, "
                    f"missing={sorted(expected - scored)[:5]})"
                )


# --------------------------------------------------------------------------
# Stage B: cells, substrate, paired tests
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainCell:
    arm: str
    domain: Domain
    structure: str
    test_image_count: int
    intervals: Mapping[str, SeedInterval]


def build_cells(runs: Sequence[AllDomainsRun]) -> tuple[DomainCell, ...]:
    cells: list[DomainCell] = []
    arm = runs[0].arm
    seeds = [r.run_seed for r in runs]
    for domain in sorted({Domain(d) for r in runs for d in r.active_domains}, key=lambda d: d.value):
        for structure in CHANNEL_NAMES:
            summaries = []
            for run in runs:
                rows = [r for r in read_per_image_csv(run.per_image_csv[domain]) if r["structure"] == structure]
                if not rows:
                    raise FixedLodoReportError(f"{run.label}: no {structure} rows for {domain.value}")
                hd = np.asarray([float(r["hd95"]) for r in rows])
                finite = hd[np.isfinite(hd)]
                summaries.append({
                    "dice": float(np.mean([float(r["dice"]) for r in rows])),
                    "iou": float(np.mean([float(r["iou"]) for r in rows])),
                    "hd95": float(finite.mean()) if finite.size else None,
                    "n": len(rows),
                })
            counts = {s["n"] for s in summaries}
            if len(counts) != 1:
                raise FixedLodoReportError(f"{arm} {domain.value} {structure}: seeds scored different image counts")
            intervals: dict[str, SeedInterval] = {}
            for metric in ("dice", "iou", "hd95"):
                series = [s[metric] for s in summaries]
                if any(v is None for v in series):
                    continue
                intervals[metric] = seed_confidence_interval(metric, seeds, series)
            cells.append(DomainCell(arm, domain, structure, next(iter(counts)), intervals))
    return tuple(cells)


def build_substrate(plain: Sequence[AllDomainsRun], film: Sequence[AllDomainsRun]) -> Substrate:
    store: dict[tuple[str, Domain, str, str], dict[str, list[float]]] = {}
    seeds: dict[tuple[str, Domain], set[int]] = {}
    for run in (*plain, *film):
        for domain, csv_path in run.per_image_csv.items():
            _accumulate(store, seeds, run.arm, domain, run.run_seed, csv_path)
    values: dict[tuple[str, Domain, str, str], Mapping[str, float]] = {}
    for key, bucket in store.items():
        expected = len(seeds[(key[0], key[1])])
        for metric, samples in bucket.items():
            if len(samples) != expected:
                raise FixedLodoReportError(
                    f"{key[0]}/{key[1].value} {key[2]} image {key[3]!r} has {len(samples)} "
                    f"{metric} values but the arm ran {expected} seeds"
                )
        values[key] = {m: float(np.mean(v)) for m, v in bucket.items()}
    return Substrate(seed_counts={c: len(v) for c, v in seeds.items()}, values=values)


# --------------------------------------------------------------------------
# Stage C: the wrong-code penalty from the FiLM runs' sweeps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PenaltyCell:
    domain: Domain
    structure: str
    seeds: tuple[int, ...]
    own_code_dice: float  # mean over seeds
    best_other_code_dice: float
    worst_other_code_dice: float
    own_code_was_best: int  # seeds where the domain's own code scored highest
    sweep_by_code: Mapping[str, float]  # mean over seeds


def build_penalty_cells(film: Sequence[AllDomainsRun]) -> tuple[PenaltyCell, ...]:
    cells: list[PenaltyCell] = []
    domains = sorted({Domain(d) for r in film for d in r.active_domains}, key=lambda d: d.value)
    for domain in domains:
        for structure in CHANNEL_NAMES:
            own, best_other, worst_other, wins = [], [], [], 0
            by_code: dict[str, list[float]] = {}
            for run in film:
                block = run.test_by_domain[domain.value]
                conditioning = block.get("conditioning") or {}
                sweep = conditioning.get("fixed_code_sweep")
                if not sweep:
                    raise FixedLodoReportError(f"{run.label} {domain.value}: no fixed-code sweep recorded")
                scores = {code: float(entry[structure]["dice_mean"]) for code, entry in sweep.items()}
                for code, value in scores.items():
                    by_code.setdefault(code, []).append(value)
                own_dice = float(block[structure]["dice_mean"])
                others = [v for c, v in scores.items() if c != domain.value]
                own.append(own_dice)
                best_other.append(max(others))
                worst_other.append(min(others))
                wins += int(own_dice >= max(scores.values()) - 1e-9)
            cells.append(PenaltyCell(
                domain=domain, structure=structure,
                seeds=tuple(r.run_seed for r in film),
                own_code_dice=float(np.mean(own)),
                best_other_code_dice=float(np.mean(best_other)),
                worst_other_code_dice=float(np.mean(worst_other)),
                own_code_was_best=wins,
                sweep_by_code={c: float(np.mean(v)) for c, v in sorted(by_code.items())},
            ))
    return tuple(cells)


# --------------------------------------------------------------------------
# Stage D: render
# --------------------------------------------------------------------------


def _fmt(cell: DomainCell | None) -> str:
    if cell is None or "dice" not in cell.intervals:
        return "—"
    d = cell.intervals["dice"]
    return f"{d.mean:.4f} ± {d.std:.4f}"


def render_side_by_side(
    plain_cells: Sequence[DomainCell], film_cells: Sequence[DomainCell],
    results: Sequence[PairedResult], plain_arm: str, film_arm: str,
) -> str:
    plain = {(c.domain, c.structure): c for c in plain_cells}
    film = {(c.domain, c.structure): c for c in film_cells}
    tests = {(r.held_out_domain, r.structure): r for r in results}
    lines = [
        "| Test domain | Structure | Images | Plain Dice, mean ± seed SD | FiLM Dice, mean ± seed SD | "
        "Δ (FiLM − plain) | p | p (Holm) | Significant |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for domain in sorted({k[0] for k in plain} | {k[0] for k in film}, key=lambda d: d.value):
        for structure in CHANNEL_NAMES:
            p, f, t = plain.get((domain, structure)), film.get((domain, structure)), tests.get((domain, structure))
            images = (p or f).test_image_count if (p or f) else 0
            if t is None:
                delta, pv, ph, sig = "—", "—", "—", "—"
            else:
                delta, pv, ph = f"{t.mean_difference:+.4f}", f"{t.p_value:.4g}", f"{t.p_adjusted:.4g}"
                sig = "**yes**" if t.significant else "no"
            lines.append(
                f"| `{domain.value}` | {structure} | {images} | {_fmt(p)} | {_fmt(f)} | {delta} | {pv} | {ph} | {sig} |"
            )
    lines.append("")
    lines.append(f"Plain arm: `{plain_arm}`. FiLM arm: `{film_arm}`.")
    return "\n".join(lines)


def render_penalty_table(cells: Sequence[PenaltyCell]) -> str:
    lines = [
        "| Test domain | Structure | Own code Dice | Best other code | Worst other code | "
        "Penalty (own − worst) | Own code best in N/seeds | Dice under each code (mean over seeds) |",
        "|---|---|---:|---:|---:|---:|:---:|---|",
    ]
    for c in cells:
        codes = ", ".join(f"`{k}`: {v:.4f}" for k, v in c.sweep_by_code.items())
        lines.append(
            f"| `{c.domain.value}` | {c.structure} | {c.own_code_dice:.4f} | {c.best_other_code_dice:.4f} | "
            f"{c.worst_other_code_dice:.4f} | {c.own_code_dice - c.worst_other_code_dice:+.4f} | "
            f"{c.own_code_was_best}/{len(c.seeds)} | {codes} |"
        )
    return "\n".join(lines)


def render_markdown_report(
    plain_cells, film_cells, results, penalties, plain_runs, film_runs, manifest_path, method
) -> str:
    run = film_runs[0]
    active = list(run.active_domains)
    lines: list[str] = []
    add = lines.append
    add("# Step 4: train on all domains, test on each — Global FiLM against the plain U-Net")
    add("")
    add(
        "**Evidence boundary.** Every figure is computed by `aggregate_stage4_all_domains.py` "
        f"from the per-image metric CSVs of {len(plain_runs)} plain and {len(film_runs)} Global "
        f"FiLM runs, validated against the shared budgeted manifest `{manifest_path.name}`. "
        "Interpretation is written by hand."
    )
    add("")
    add("## 1. Protocol")
    add("")
    add(
        f"One model per seed trained on the pooled budgeted training partitions of "
        f"`{'`, `'.join(active)}` ({run.train_budget} each, {run.train_budget * len(active)} in total), "
        f"selected on their pooled validation ({run.val_budget * len(active)}), and scored on each "
        f"domain's own {run.test_budget} locked test images. Nothing is held out: the FiLM arm is "
        "trained with the true domain code and tested with it, the regime of the SpFiLM draft's "
        "'both' setting. It asks whether conditioning helps when the camera is known, and — through "
        "the fixed-code sweep — whether the network uses the code at all."
    )
    add("")
    add("## 2. Dice per domain, side by side")
    add("")
    add(
        "Same backbone, folds, budget, seeds, augmentation, optimiser and test images; the arms "
        f"differ only in the conditioning. Δ is FiLM minus plain on per-image Dice with seeds "
        f"averaged first; p-values are {method}, Holm-adjusted over {len(results)} tests, "
        f"significance at α = {ALPHA}."
    )
    add("")
    add(render_side_by_side(plain_cells, film_cells, results, plain_runs[0].arm, film_runs[0].arm))
    add("")
    add("## 3. Does the network use the code? The wrong-code penalty")
    add("")
    add(
        "Each FiLM model was also scored on every test domain under every *other* domain's "
        "code. 'Penalty' is Dice under the domain's own code minus Dice under the worst other "
        "code. A penalty near zero means the codes are interchangeable and the FiLM layers are "
        "inert; a clear penalty means the code carries information the network acts on."
    )
    add("")
    add(render_penalty_table(penalties))
    add("")
    add("## 4. Findings")
    add("")
    add("<!-- TODO: written by hand; the tool does not infer this. -->")
    add("")
    return "\n".join(lines)


def write_csv(results: Sequence[PairedResult], penalties: Sequence[PenaltyCell], path: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "kind": "paired_test", "domain": r.held_out_domain.value, "structure": r.structure,
            "arm_plain": r.arm_a, "arm_film": r.arm_b, "test_images": r.image_count,
            "dice_mean_plain": f"{r.mean_a:.6g}", "dice_mean_film": f"{r.mean_b:.6g}",
            "mean_difference": f"{r.mean_difference:.6g}", "median_difference": f"{r.median_difference:.6g}",
            "method": r.method, "p_value": f"{r.p_value:.6g}", "p_adjusted": f"{r.p_adjusted:.6g}",
            "significant": r.significant,
        })
    for c in penalties:
        row: dict[str, Any] = {
            "kind": "wrong_code_penalty", "domain": c.domain.value, "structure": c.structure,
            "seeds": " ".join(str(s) for s in c.seeds),
            "own_code_dice": f"{c.own_code_dice:.6g}", "best_other_code_dice": f"{c.best_other_code_dice:.6g}",
            "worst_other_code_dice": f"{c.worst_other_code_dice:.6g}",
            "penalty_own_minus_worst": f"{c.own_code_dice - c.worst_other_code_dice:.6g}",
            "own_code_was_best_seeds": c.own_code_was_best,
        }
        for code, value in c.sweep_by_code.items():
            row[f"dice_under_{code}"] = f"{value:.6g}"
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
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(DEFAULT_EXPECTED_SEEDS))
    parser.add_argument("--method", choices=PAIRED_METHODS, default="wilcoxon")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = [Path(r) for r in (args.run_root or list(DEFAULT_RUN_ROOTS))]
    try:
        manifest_path = args.manifest.expanduser().resolve()
        runs = discover_runs(roots)
        plain = select_runs(runs, args.plain_arm, args.expected_seeds)
        film = select_runs(runs, args.film_arm, args.expected_seeds)
        if plain[0].active_domains != film[0].active_domains:
            raise FixedLodoReportError(
                f"Arms trained on different active domains: {plain[0].active_domains} vs {film[0].active_domains}"
            )
        verify_membership((*plain, *film), manifest_path)
        plain_cells = build_cells(plain)
        film_cells = build_cells(film)
        substrate = build_substrate(plain, film)
        results = paired_tests(substrate, method=args.method, reference_arm=args.film_arm)
        penalties = build_penalty_cells(film)
    except (FixedLodoReportError, OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    print(f"plain: {len(plain)} runs | film: {len(film)} runs | paired tests: {len(results)}")
    print()
    print(render_side_by_side(plain_cells, film_cells, results, args.plain_arm, args.film_arm))
    print()
    print(render_penalty_table(penalties))
    if args.report_out is not None:
        report = render_markdown_report(
            plain_cells, film_cells, results, penalties, plain, film, manifest_path, args.method
        )
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.report_out}")
    if args.csv_out is not None:
        print(f"wrote {write_csv(results, penalties, args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
