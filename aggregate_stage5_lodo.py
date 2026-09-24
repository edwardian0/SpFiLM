#!/usr/bin/env python3
"""Aggregate the Step 5 leave-one-domain-out runs: a conditioned arm next to plain.

Both arms come from ``run_stage5_lodo.py``: the fixed-budget LODO folds over the
three active domains, so for each held-out domain they score the identical 50
test images and differ only in the conditioning. This reads the Step 5 runner's
``stage5_lodo`` metadata block alone -- Stage 3 and Step 4 runs sharing a run
root are invisible to it -- and reuses the Step 4 LODO aggregator for the rest:
membership proven against the locked manifest, per-domain Dice with the seed
spread, conditioned minus plain on per-image Dice with seeds averaged first
(Wilcoxon, Holm-adjusted), and the conditioning diagnostics (the code the
held-out domain was given, selector accuracy, the fixed-code sweep). A single
seed per arm is accepted, so seed 42 can be read before the grid is launched.

``--film-arm`` names the conditioned arm and ``--plain-arm`` its comparator; a
positive difference reads "the conditioned arm helped". SpFiLM, once it is a
Step 5 arm, is compared with plain by passing its experiment name as
``--film-arm``.

    python aggregate_stage5_lodo.py --expected-seeds 42 --report-out run_reports/stage5_lodo.md
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_stage3_fixed import (  # noqa: E402
    DEFAULT_EXPECTED_SEEDS,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOTS,
    PAIRED_METHODS,
    FixedLodoReportError,
    FixedRun,
    build_domain_cells,
    paired_tests,
)
from aggregate_stage4_film import (  # noqa: E402
    build_conditioning_cells,
    build_two_arm_substrate,
    load_arm,
    render_conditioning_table,
    render_markdown_report,
    render_side_by_side,
    require_same_folds,
    write_csv,
)
from run_stage5_lodo import (  # noqa: E402
    STAGE5_LODO_METADATA_KEY,
    STAGE5_LODO_PROTOCOL_NAME,
)


DEFAULT_PLAIN_ARM = "stage5_lodo_fixed_budget_plain_unet_3dom"
DEFAULT_FILM_ARM = "stage5_lodo_fixed_budget_global_film_3dom"
REPORT_TITLE = "Step 5: Global FiLM against the plain U-Net, leave-one-domain-out"


def load_stage5_arm(
    roots: Sequence[Path],
    arm: str,
    expected_seeds: Sequence[int],
    manifest_path: Path,
) -> tuple[FixedRun, ...]:
    """One Step 5 arm's runs; a Stage 3 or Step 4 run is never picked up."""

    return load_arm(
        roots,
        arm,
        expected_seeds,
        manifest_path,
        metadata_key=STAGE5_LODO_METADATA_KEY,
        protocol=STAGE5_LODO_PROTOCOL_NAME,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        help="Directory to search for run outputs (repeatable; default artifacts/)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--plain-arm", default=DEFAULT_PLAIN_ARM, help="Comparator arm (its experiment_name)"
    )
    parser.add_argument(
        "--film-arm", default=DEFAULT_FILM_ARM, help="Conditioned arm (its experiment_name)"
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXPECTED_SEEDS),
        help="Seeds both arms must have; pass a subset (e.g. 42) before the grid is complete",
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
        plain = load_stage5_arm(roots, args.plain_arm, args.expected_seeds, manifest_path)
        film = load_stage5_arm(roots, args.film_arm, args.expected_seeds, manifest_path)
        require_same_folds(plain, film)
        plain_cells = build_domain_cells(plain, allow_single_seed=True)
        film_cells = build_domain_cells(film, allow_single_seed=True)
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
            plain_cells,
            film_cells,
            results,
            conditioning,
            plain,
            film,
            manifest_path,
            args.method,
            title=REPORT_TITLE,
            tool="aggregate_stage5_lodo.py",
        )
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.report_out}")
    if args.csv_out is not None:
        print(f"wrote {write_csv(results, conditioning, args.csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
