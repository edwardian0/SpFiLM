#!/usr/bin/env python3
"""How large a within-domain intensity distance is pure sampling noise.

``analyze_domain_shift.py --split-diagnosis`` reports a Wasserstein-1 distance
between a domain's glaucoma and non-glaucoma intensity curves. Balanced, that
comparison rests on 31 images a side, and a distance computed from 31 images is
not zero even when the two subsets are drawn from the same population. Without a
floor there is no way to tell a real diagnosis effect from the draw.

This measures the floor directly: split one domain's images *of a single
diagnosis class* into two disjoint random halves of the same size the split
uses, and take the distance between them. Both halves are the same domain and
the same diagnosis, so whatever distance appears is sampling noise. A
glaucoma-versus-non-glaucoma distance at or below this figure is not evidence of
a diagnosis effect.

Only a class with at least ``2 * --subset`` images can be split this way, so the
floor is reported for whichever (domain, class) pools are large enough; the
noise floor depends on the subset size, not on which class was resampled.
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from analyze_domain_shift import (  # noqa: E402
    DIAGNOSIS_CLASSES,
    _write_csv,
    group_by_diagnosis,
    refuge_validation_labels,
    wasserstein_distance,
)
from spfilm.global_histograms import CHANNELS, POPULATIONS, accumulate_domain  # noqa: E402
from spfilm.stage3 import (  # noqa: E402
    Stage3ConfigError,
    Stage3DataError,
    Stage3LodoConfig,
    discover_lodo_records,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "domain_shift_dx_global"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--working-size", type=int, default=256)
    parser.add_argument(
        "--subset",
        type=int,
        default=31,
        help=(
            "Images per half. Match this to the balanced split's cap, or the "
            "floor describes a different sample size than the one in question"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent disjoint splits per pool; the CSV reports mean and max",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = Stage3LodoConfig.from_json(args.config.expanduser().resolve())
        records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
        extra_labels = refuge_validation_labels(config, PROJECT_ROOT)
    except (Stage3ConfigError, Stage3DataError, OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    for domain in sorted(records_by_domain, key=lambda item: item.value):
        grouped, _ = group_by_diagnosis(records_by_domain[domain], extra_labels)
        for label in DIAGNOSIS_CLASSES:
            pool = grouped[label]
            if len(pool) < 2 * args.subset:
                print(
                    f"NOTE: {domain.value} {label}: {len(pool)} images cannot "
                    f"give two disjoint halves of {args.subset}",
                    flush=True,
                )
                continue
            print(
                f"resampling {domain.value} {label}: {args.repeats} disjoint "
                f"{args.subset}v{args.subset} splits of {len(pool)} images",
                flush=True,
            )
            measured: dict[tuple[str, str], list[float]] = {
                (population, channel): []
                for population in POPULATIONS
                for channel in CHANNELS
            }
            for repeat in range(args.repeats):
                # Seeded on the pool as well as the repeat, so one domain's
                # draws do not shadow another's.
                rng = np.random.default_rng(
                    [args.seed, repeat, *map(ord, f"{domain.value}:{label}")]
                )
                order = rng.permutation(len(pool))
                halves = [
                    accumulate_domain(
                        domain.value,
                        [pool[index] for index in chunk],
                        args.working_size,
                        args.workers,
                    )
                    for chunk in (
                        order[: args.subset],
                        order[args.subset : 2 * args.subset],
                    )
                ]
                for population in POPULATIONS:
                    for channel in CHANNELS:
                        measured[(population, channel)].append(
                            wasserstein_distance(
                                halves[0].density(population, channel),
                                halves[1].density(population, channel),
                            )
                        )
            for (population, channel), values in measured.items():
                rows.append(
                    {
                        "domain": domain.value,
                        "resampled_class": label,
                        "population": population,
                        "channel": channel,
                        "subset": args.subset,
                        "repeats": args.repeats,
                        "wasserstein_1_mean": round(float(np.mean(values)), 6),
                        "wasserstein_1_max": round(float(np.max(values)), 6),
                    }
                )

    if not rows:
        print("FATAL: no pool was large enough to split", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    path = _write_csv(output_dir / "diagnosis_noise_floor.csv", rows)

    print()
    print(
        f"same-domain, same-diagnosis W1 at n={args.subset} a side (fov pixels); "
        "a glaucoma-vs-non distance at or below this is not a diagnosis effect:"
    )
    print(f"  {'domain':18s}{'class':14s}" + "".join(f"{c:>9s}" for c in CHANNELS))
    for domain_name, label in sorted(
        {(str(row["domain"]), str(row["resampled_class"])) for row in rows}
    ):
        selected = {
            str(row["channel"]): row
            for row in rows
            if row["domain"] == domain_name
            and row["resampled_class"] == label
            and row["population"] == "fov"
        }
        print(
            f"  {domain_name:18s}{label:14s}"
            + "".join(
                f"{selected[channel]['wasserstein_1_mean']:9.4f}"
                for channel in CHANNELS
            )
            + "   max"
            + "".join(
                f"{selected[channel]['wasserstein_1_max']:8.4f}"
                for channel in CHANNELS
            )
        )
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
