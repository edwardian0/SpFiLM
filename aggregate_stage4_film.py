#!/usr/bin/env python3
"""Put the Global FiLM arm next to the plain fixed-budget arm (Step 4 exit gate).

Both arms run the same fixed-budget leave-one-domain-out protocol from the same
budgeted manifest, so for each held-out domain they score the identical test
images and differ in one thing: the plain arm has no conditioning, the FiLM arm
adds channel-wise FiLM to the encoder and picks each held-out image's code with
the nearest-source-domain rule. This script reuses the fixed-budget aggregator's
loading, per-domain summaries, and paired test, with FiLM as the reference arm
so a positive difference reads "FiLM helped".

It also reduces the conditioning diagnostics each FiLM run records: how well the
selector recovers the true domain on source validation images, which codes the
held-out images were assigned, and how the nearest-domain code compares with
the best of the fixed-code sweep. Those say whether the selector matters at all
before anyone reads meaning into the Dice difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
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

from aggregate_stage3_fixed import (  # noqa: E402
    ALPHA,
    DEFAULT_EXPECTED_SEEDS,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOTS,
    PAIRED_METHODS,
    DomainCell,
    FixedLodoReportError,
    FixedRun,
    PairedResult,
    Substrate,
    _accumulate,
    _per_image_rows,
    _sha256,
    build_domain_cells,
    discover_fixed_runs,
    paired_tests,
    select_fixed_runs,
)
from run_stage3_lodo_3_1_fixed import fixed_lodo_folds  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.metrics import CHANNEL_NAMES  # noqa: E402
from spfilm.single_source import load_single_source_manifest  # noqa: E402


DEFAULT_PLAIN_ARM = "stage4_lodo_fixed_budget_plain_unet_3dom"
DEFAULT_FILM_ARM = "stage4_lodo_fixed_budget_global_film_3dom"


# --------------------------------------------------------------------------
# Loading: one arm at a time, then the two together
# --------------------------------------------------------------------------


def _restrict_seeds(runs: Sequence[FixedRun], seeds: Sequence[int]) -> tuple[FixedRun, ...]:
    """Keep only the requested seeds so a partial FiLM grid pairs with plain."""

    return tuple(run for run in runs if run.run_seed in set(seeds))


def load_arm(
    roots: Sequence[Path],
    arm: str,
    expected_seeds: Sequence[int],
    manifest_path: Path,
) -> tuple[FixedRun, ...]:
    runs = select_fixed_runs(
        _restrict_seeds(discover_fixed_runs(roots), expected_seeds),
        tuple(expected_seeds),
        arm=arm,
    )
    digest = _sha256(manifest_path)
    if runs[0].manifest_sha256 != digest:
        raise FixedLodoReportError(
            f"{arm} runs were produced against a different manifest than "
            f"{manifest_path}: runs say {runs[0].manifest_sha256[:12]}…, this "
            f"file is {digest[:12]}…"
        )
    folds = {f.held_out_domain: f for f in fixed_lodo_folds(load_single_source_manifest(manifest_path))}
    for run in runs:
        expected = {s.sample_id for s in folds[run.held_out_domain].test}
        scored = {str(r["image_id"]) for r in _per_image_rows(run.per_image_csv)}
        if scored != expected:
            raise FixedLodoReportError(
                f"{arm} {run.label}: scored images do not match the locked test "
                f"partition (unexpected={sorted(scored - expected)[:5]}, "
                f"missing={sorted(expected - scored)[:5]})"
            )
    return runs


def require_same_folds(
    plain_runs: Sequence[FixedRun], film_runs: Sequence[FixedRun]
) -> None:
    """Refuse to pair arms whose folds trained on different source domains.

    Identical test images are necessary but not sufficient: the Stage 3 plain
    runs score the same 50 held-out images as a Step 4 FiLM run but trained on
    three domains including RIM-ONE-DL, so a difference between them would mix
    "conditioning" with "which domains were available". Only a plain arm run
    under the same active set is the honest pair.
    """

    plain_sources = {
        run.held_out_domain: frozenset(run.source_domains) for run in plain_runs
    }
    for run in film_runs:
        expected = plain_sources.get(run.held_out_domain)
        if expected is None:
            continue  # summarised alone; paired_tests only pairs shared domains
        if frozenset(run.source_domains) != expected:
            raise FixedLodoReportError(
                f"{run.held_out_domain.value}: the FiLM arm trained on "
                f"{sorted(run.source_domains)} but the plain arm on "
                f"{sorted(expected)}; pass a plain arm run under the same "
                "active domains (e.g. stage4_lodo_fixed_budget_plain_unet_3dom)"
            )


def build_two_arm_substrate(
    plain_runs: Sequence[FixedRun], film_runs: Sequence[FixedRun]
) -> Substrate:
    store: dict[tuple[str, Domain, str, str], dict[str, list[float]]] = {}
    seeds: dict[tuple[str, Domain], set[int]] = {}
    for run in plain_runs:
        _accumulate(store, seeds, run.arm, run.held_out_domain, run.run_seed, run.per_image_csv)
    for run in film_runs:
        _accumulate(store, seeds, run.arm, run.held_out_domain, run.run_seed, run.per_image_csv)
    values: dict[tuple[str, Domain, str, str], Mapping[str, float]] = {}
    for key, bucket in store.items():
        expected = len(seeds[(key[0], key[1])])
        for metric, samples in bucket.items():
            if len(samples) != expected:
                raise FixedLodoReportError(
                    f"{key[0]}/{key[1].value} {key[2]} image {key[3]!r} has "
                    f"{len(samples)} {metric} values but the arm ran {expected} seeds"
                )
        values[key] = {m: float(np.mean(v)) for m, v in bucket.items()}
    return Substrate(seed_counts={c: len(v) for c, v in seeds.items()}, values=values)


# --------------------------------------------------------------------------
# Conditioning diagnostics recorded by each FiLM run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningCell:
    held_out_domain: Domain
    seeds: tuple[int, ...]
    vocabulary: tuple[str, ...]
    test_conditioning: str
    chosen_code: str | None  # the per-domain decision (same across seeds)
    selector_val_accuracy: float  # per-image rule on source validation images
    selector_domain_accuracy: float | None  # per-domain rule on source validation images
    assignment_counts: Mapping[str, float]  # codes actually used, mean over seeds
    nearest_image_counts: Mapping[str, float] | None  # per-image rule, diagnostic
    nearest_minus_best: Mapping[str, float]  # per structure, mean over seeds
    sweep_spread: Mapping[str, float]  # per structure: max - min fixed-code Dice
    best_fixed_code: Mapping[str, Mapping[str, int]]  # structure -> code -> votes


def _conditioning_payload(run: FixedRun) -> dict[str, Any]:
    payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    conditioning = payload.get("conditioning")
    if not isinstance(conditioning, dict):
        raise FixedLodoReportError(
            f"{run.label} ({run.arm}) has no conditioning block; is it a FiLM run?"
        )
    return conditioning


def build_conditioning_cells(film_runs: Sequence[FixedRun]) -> tuple[ConditioningCell, ...]:
    cells: list[ConditioningCell] = []
    for domain in sorted({r.held_out_domain for r in film_runs}, key=lambda d: d.value):
        runs = sorted((r for r in film_runs if r.held_out_domain == domain), key=lambda r: r.run_seed)
        payloads = [_conditioning_payload(run) for run in runs]
        vocabularies = {tuple(p["vocabulary"]) for p in payloads}
        if len(vocabularies) != 1:
            raise FixedLodoReportError(
                f"{domain.value}: seeds disagree on the code vocabulary {sorted(vocabularies)}"
            )
        vocabulary = next(iter(vocabularies))
        policies = {str(p["test_conditioning"]) for p in payloads}
        if len(policies) != 1:
            raise FixedLodoReportError(
                f"{domain.value}: seeds disagree on test_conditioning {sorted(policies)}"
            )
        decisions = {
            (p.get("domain_decision") or {}).get("chosen_domain") for p in payloads
        }
        if len(decisions) != 1:
            # The decision depends only on the fold's images, so seeds must agree.
            raise FixedLodoReportError(
                f"{domain.value}: seeds disagree on the held-out code {sorted(map(str, decisions))}"
            )
        accuracies = [float(p["selector_validation"]["accuracy"]) for p in payloads]
        domain_accuracies = [
            p["selector_validation"].get("domain_level_accuracy") for p in payloads
        ]
        counts = {
            code: float(np.mean([p["test"]["assignment_counts"][code] for p in payloads]))
            for code in vocabulary
        }
        per_image = [p["test"].get("nearest_image_counts") for p in payloads]
        nearest_image_counts = (
            {
                code: float(np.mean([entry[code] for entry in per_image]))
                for code in vocabulary
            }
            if all(entry is not None for entry in per_image)
            else None
        )
        nearest_minus_best = {
            s: float(np.mean([p["nearest_domain_minus_best_fixed_code_dice"][s] for p in payloads]))
            for s in CHANNEL_NAMES
        }
        spread = {}
        for s in CHANNEL_NAMES:
            per_seed = []
            for p in payloads:
                dice = [float(p["fixed_code_sweep"][code][s]["dice_mean"]) for code in vocabulary]
                per_seed.append(max(dice) - min(dice))
            spread[s] = float(np.mean(per_seed))
        votes = {
            s: {code: sum(1 for p in payloads if p["best_fixed_code"][s] == code) for code in vocabulary}
            for s in CHANNEL_NAMES
        }
        cells.append(
            ConditioningCell(
                held_out_domain=domain,
                seeds=tuple(r.run_seed for r in runs),
                vocabulary=vocabulary,
                test_conditioning=next(iter(policies)),
                chosen_code=next(iter(decisions)),
                selector_val_accuracy=float(np.mean(accuracies)),
                selector_domain_accuracy=(
                    float(np.mean([float(a) for a in domain_accuracies]))
                    if all(a is not None for a in domain_accuracies)
                    else None
                ),
                assignment_counts=counts,
                nearest_image_counts=nearest_image_counts,
                nearest_minus_best=nearest_minus_best,
                sweep_spread=spread,
                best_fixed_code=votes,
            )
        )
    return tuple(cells)


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def _cell_index(cells: Sequence[DomainCell]) -> dict[tuple[Domain, str], DomainCell]:
    return {(c.held_out_domain, c.structure): c for c in cells}


def render_side_by_side(
    plain_cells: Sequence[DomainCell],
    film_cells: Sequence[DomainCell],
    results: Sequence[PairedResult],
    plain_arm: str,
    film_arm: str,
) -> str:
    plain = _cell_index(plain_cells)
    film = _cell_index(film_cells)
    tests = {(r.held_out_domain, r.structure): r for r in results}
    lines = [
        "| Held-out domain | Structure | Images | Plain Dice, mean ± seed SD | "
        "FiLM Dice, mean ± seed SD | Δ (FiLM − plain) | p | p (Holm) | Significant |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    domains = sorted({k[0] for k in plain} | {k[0] for k in film}, key=lambda d: d.value)
    for domain in domains:
        for structure in CHANNEL_NAMES:
            p = plain.get((domain, structure))
            f = film.get((domain, structure))
            t = tests.get((domain, structure))
            images = (p or f).test_image_count if (p or f) else 0

            def fmt(cell: DomainCell | None) -> str:
                if cell is None or "dice" not in cell.intervals:
                    return "—"
                d = cell.intervals["dice"]
                return f"{d.mean:.4f} ± {d.std:.4f}"

            if t is None:
                delta, pv, ph, sig = "—", "—", "—", "—"
            else:
                delta = f"{t.mean_difference:+.4f}"
                pv, ph = f"{t.p_value:.4g}", f"{t.p_adjusted:.4g}"
                sig = "**yes**" if t.significant else "no"
            lines.append(
                f"| `{domain.value}` | {structure} | {images} | {fmt(p)} | {fmt(f)} | "
                f"{delta} | {pv} | {ph} | {sig} |"
            )
    lines.append("")
    lines.append(f"Plain arm: `{plain_arm}`. FiLM arm: `{film_arm}`.")
    return "\n".join(lines)


def render_conditioning_table(cells: Sequence[ConditioningCell]) -> str:
    lines = [
        "| Held-out domain | Seeds | Code used | Selector accuracy on source val (per domain / per image) | "
        "Per-image nearest source among held-out images | Fixed-code sweep spread, disc / cup | "
        "Used − best fixed code, disc / cup | Best fixed code votes, disc / cup |",
        "|---|---|---|---:|---|---:|---:|---|",
    ]
    for c in cells:
        code = f"`{c.chosen_code}`" if c.chosen_code else f"({c.test_conditioning})"
        domain_acc = "—" if c.selector_domain_accuracy is None else f"{c.selector_domain_accuracy:.2f}"
        per_image = (
            ", ".join(f"`{k}`: {c.nearest_image_counts[k]:.1f}" for k in c.vocabulary)
            if c.nearest_image_counts
            else "—"
        )
        votes = " / ".join(
            ", ".join(f"`{k}`×{n}" for k, n in c.best_fixed_code[s].items() if n)
            for s in CHANNEL_NAMES
        )
        lines.append(
            f"| `{c.held_out_domain.value}` | {len(c.seeds)} | {code} | "
            f"{domain_acc} / {c.selector_val_accuracy:.3f} | {per_image} | "
            f"{c.sweep_spread['disc']:.4f} / {c.sweep_spread['cup']:.4f} | "
            f"{c.nearest_minus_best['disc']:+.4f} / {c.nearest_minus_best['cup']:+.4f} | {votes} |"
        )
    return "\n".join(lines)


def render_markdown_report(
    plain_cells: Sequence[DomainCell],
    film_cells: Sequence[DomainCell],
    results: Sequence[PairedResult],
    conditioning: Sequence[ConditioningCell],
    plain_runs: Sequence[FixedRun],
    film_runs: Sequence[FixedRun],
    manifest_path: Path,
    method: str,
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Step 4: Global FiLM against the plain U-Net, fixed-budget leave-one-domain-out")
    add("")
    add(
        "**Evidence boundary.** Every figure is computed by `aggregate_stage4_film.py` "
        f"from the per-image metric CSVs of {len(plain_runs)} plain and {len(film_runs)} "
        "Global FiLM runs, validated against the shared budgeted manifest "
        f"`{manifest_path.name}`. Interpretation is written by hand."
    )
    add("")
    sources = len(film_runs[0].source_domains)
    active = sorted({film_runs[0].held_out_domain.value, *film_runs[0].source_domains})
    add("## 1. Held-out Dice side by side")
    add("")
    add(
        f"Leave-one-domain-out over {len(active)} domains (`{'`, `'.join(active)}`): "
        f"train on {sources}, test on the held-out one, "
        f"{film_runs[0].train_budget * sources} / {film_runs[0].val_budget * sources} / "
        f"{film_runs[0].test_budget} images per fold. "
        "Same backbone, folds, budget, seeds, augmentation, optimiser and test images; "
        "the arms differ only in the conditioning. Δ is FiLM minus plain on per-image "
        f"Dice with seeds averaged first; p-values are {method}, Holm-adjusted over "
        f"{len(results)} tests, significance at α = {ALPHA}."
    )
    add("")
    add(render_side_by_side(plain_cells, film_cells, results, plain_runs[0].arm, film_runs[0].arm))
    add("")
    add("## 2. Did the conditioning do anything, and did the selector find it?")
    add("")
    add(
        "The held-out domain's code is decided once, from the mean colour statistics "
        "of its unlabelled reference sample (its budgeted training partition, labels "
        "unused, disjoint from the test images), as the source domain with the nearest "
        "training centroid. Selector accuracy is measured on source-domain validation "
        "images whose true domain is known, for the per-domain rule (one decision per "
        "domain) and the per-image rule. The per-image column shows how the held-out "
        "test images would individually be assigned; unanimity means the domain-level "
        "decision is uncontroversial. The fixed-code sweep scores the held-out set once "
        "under each source code: a spread near zero means the codes are interchangeable "
        "and the decision is irrelevant; a large spread with a negative 'used − best' "
        "means the rule picked a worse code than was available."
    )
    add("")
    add(render_conditioning_table(conditioning))
    add("")
    add("## 3. Findings")
    add("")
    add("<!-- TODO: written by hand; the tool does not infer this. -->")
    add("")
    return "\n".join(lines)


def write_csv(
    results: Sequence[PairedResult], conditioning: Sequence[ConditioningCell], path: Path
) -> Path:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "kind": "paired_test",
                "held_out_domain": r.held_out_domain.value,
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
    for c in conditioning:
        row: dict[str, Any] = {
            "kind": "conditioning",
            "held_out_domain": c.held_out_domain.value,
            "seeds": " ".join(str(s) for s in c.seeds),
            "test_conditioning": c.test_conditioning,
            "chosen_code": c.chosen_code or "",
            "selector_val_accuracy": f"{c.selector_val_accuracy:.6g}",
            "selector_domain_accuracy": (
                "" if c.selector_domain_accuracy is None else f"{c.selector_domain_accuracy:.6g}"
            ),
        }
        for code in c.vocabulary:
            row[f"assigned_{code}"] = f"{c.assignment_counts[code]:.6g}"
        for s in CHANNEL_NAMES:
            row[f"sweep_spread_{s}"] = f"{c.sweep_spread[s]:.6g}"
            row[f"nearest_minus_best_{s}"] = f"{c.nearest_minus_best[s]:.6g}"
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
        plain = load_arm(roots, args.plain_arm, args.expected_seeds, manifest_path)
        film = load_arm(roots, args.film_arm, args.expected_seeds, manifest_path)
        require_same_folds(plain, film)
        plain_cells = build_domain_cells(plain)
        film_cells = build_domain_cells(film)
        substrate = build_two_arm_substrate(plain, film)
        results = paired_tests(substrate, method=args.method, reference_arm=args.film_arm)
        conditioning = build_conditioning_cells(film)
    except (FixedLodoReportError, OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    print(f"plain: {len(plain)} runs | film: {len(film)} runs | paired tests: {len(results)}")
    print()
    print(render_side_by_side(plain_cells, film_cells, results, args.plain_arm, args.film_arm))
    print()
    print(render_conditioning_table(conditioning))
    if args.report_out is not None:
        report = render_markdown_report(
            plain_cells, film_cells, results, conditioning, plain, film, manifest_path, args.method
        )
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.report_out}")
    if args.csv_out is not None:
        print(f"wrote {write_csv(results, conditioning, args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
