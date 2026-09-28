#!/usr/bin/env python3
"""Aggregate the Step 5 leave-one-domain-out runs: two arms side by side.

Every arm comes from ``run_stage5_lodo.py``: the fixed-budget LODO folds over
the three active domains, so for each held-out domain the arms score the
identical 50 test images and differ only in the conditioning. This reads the
Step 5 runner's ``stage5_lodo`` metadata block alone -- Stage 3 and Step 4 runs
sharing a run root are invisible to it -- and reuses the Step 4 LODO aggregator
for the rest: membership proven against the locked manifest, per-domain Dice
with the seed spread, arm B minus arm A on per-image Dice with seeds averaged
first (Wilcoxon, Holm-adjusted over the table's 6 tests), and the conditioning
diagnostics of every conditioned arm (the signal the held-out domain was given,
selector accuracy, the sweep over training signals). Arms are labelled from
what their runs recorded. A single seed per arm is accepted, so seed 42 can be
read before the grid is launched.

``--arm-b`` names the reference arm and ``--arm-a`` its comparator, so a
positive difference reads "B helped"; ``--film-arm`` / ``--plain-arm`` are the
older spellings. The defaults are Global FiLM against plain. SpFiLM:

    python aggregate_stage5_lodo.py --arm-b stage5_lodo_fixed_budget_spatial_film_k8_3dom \\
        --expected-seeds 42 --report-out run_reports/stage5_lodo_spfilm_vs_plain.md
    python aggregate_stage5_lodo.py --arm-a stage5_lodo_fixed_budget_global_film_3dom \\
        --arm-b stage5_lodo_fixed_budget_spatial_film_k8_3dom --expected-seeds 42
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
    PAIRED_METHODS,
    FixedRun,
)
from aggregate_stage4_film import add_arm_arguments, load_arm, run  # noqa: E402
from run_stage5_lodo import (  # noqa: E402
    STAGE5_LODO_METADATA_KEY,
    STAGE5_LODO_PROTOCOL_NAME,
)


DEFAULT_PLAIN_ARM = "stage5_lodo_fixed_budget_plain_unet_3dom"
DEFAULT_FILM_ARM = "stage5_lodo_fixed_budget_global_film_3dom"
SPATIAL_FILM_ARM = "stage5_lodo_fixed_budget_spatial_film_k8_3dom"
# Filled with the arms' labels, e.g. "SpFiLM against Global FiLM".
REPORT_TITLE = "Step 5: {arm_b} against {arm_a}, leave-one-domain-out"
REPORT_TOOL = "aggregate_stage5_lodo.py"


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
    add_arm_arguments(parser, DEFAULT_PLAIN_ARM, DEFAULT_FILM_ARM)
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
    return run(
        parse_args(argv),
        metadata_key=STAGE5_LODO_METADATA_KEY,
        protocol=STAGE5_LODO_PROTOCOL_NAME,
        title=REPORT_TITLE,
        tool=REPORT_TOOL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
