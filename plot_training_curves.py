#!/usr/bin/env python3
"""Draw training curves for one or more runs from their ``history.csv``.

The engine rewrites ``history.csv`` after every epoch and refreshes
``training_curves.png`` every few epochs on its own. This script renders the
same figure on demand -- for a run that is still training, one that was
preempted before it could finish, or a finished run whose plot you want again
(or somewhere else). It reads only the CSV, so it also works on a run directory
synced down from CREATE without the checkpoints.

    python plot_training_curves.py artifacts/runs/allf_s4_seed_42_12345
    python plot_training_curves.py artifacts/runs/allp_s4_seed_42_* artifacts/runs/allf_s4_seed_42_*
    python plot_training_curves.py some/history.csv --out /tmp/curves.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.visualization import (  # noqa: E402
    HISTORY_FILENAME,
    TRAINING_CURVES_FILENAME,
    best_epoch_from_history,
    early_stop_epoch_from_history,
    load_history,
    save_training_curves,
)


FINISHED_MARKER = "test_metrics.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render training curves from a run's history.csv, mid-run or after"
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories (holding history.csv) or history.csv paths",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output PNG; only with a single run. Default: training_curves.png "
            "beside the history"
        ),
    )
    parser.add_argument("--title", help="Figure title; only with a single run")
    return parser.parse_args(argv)


def resolve_history(run: Path) -> Path:
    if run.is_dir():
        return run / HISTORY_FILENAME
    return run


def run_status(history_path: Path) -> str:
    """'finished' once the engine has written its test report, else 'in progress'."""

    return "finished" if (history_path.parent / FINISHED_MARKER).is_file() else "in progress"


def render(run: Path, out: Path | None = None, title: str | None = None) -> Path:
    history_path = resolve_history(run)
    if not history_path.is_file():
        raise FileNotFoundError(f"No {HISTORY_FILENAME} at {history_path}")
    history = load_history(history_path)
    status = run_status(history_path)
    if title is None:
        title = f"{history_path.parent.name} | {len(history)} epochs | {status}"
    output = out if out is not None else history_path.parent / TRAINING_CURVES_FILENAME
    save_training_curves(history, output, title=title)
    last = history[-1]
    print(
        f"{output} | {status} | epochs={len(history)} "
        f"best_epoch={best_epoch_from_history(history)} "
        f"early_stop_epoch={early_stop_epoch_from_history(history)} "
        f"last val_loss={last['val_loss']:.4f} "
        f"disc={last['val_disc_dice']:.4f} cup={last['val_cup_dice']:.4f}"
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.runs) > 1 and (args.out is not None or args.title is not None):
        print("FATAL: --out and --title apply to a single run only", file=sys.stderr)
        return 64
    failures = 0
    for run in args.runs:
        try:
            render(run, args.out, args.title)
        except (OSError, ValueError) as error:
            print(f"FATAL: {run}: {error}", file=sys.stderr)
            failures += 1
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
