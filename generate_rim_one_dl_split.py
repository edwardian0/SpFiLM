#!/usr/bin/env python3
"""One-shot generator for the committed RIM-ONE-DL Step 2 split manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import (  # noqa: E402
    FundusRecord,
    RIM_ONE_DL_IMAGE_COUNT,
    RIM_ONE_DL_SPLIT_COUNTS,
    discover_rim_one_dl,
    load_rim_one_dl_split_manifest,
)


def _largest_remainder(
    group_sizes: dict[str, int], target_count: int
) -> dict[str, int]:
    """Allocate an exact total proportionally with deterministic tie-breaking."""

    total = sum(group_sizes.values())
    if not 0 < target_count < total:
        raise ValueError(f"target_count must be in (0, {total}), got {target_count}")
    quotient_remainder = {
        stratum: divmod(size * target_count, total)
        for stratum, size in group_sizes.items()
    }
    allocation = {
        stratum: quotient for stratum, (quotient, _) in quotient_remainder.items()
    }
    remaining = target_count - sum(allocation.values())
    ranked = sorted(
        group_sizes,
        key=lambda stratum: (-quotient_remainder[stratum][1], stratum),
    )
    for stratum in ranked[:remaining]:
        allocation[stratum] += 1
    return allocation


def build_partitions(
    records: list[FundusRecord], seed: int
) -> dict[str, list[str]]:
    if len(records) != RIM_ONE_DL_IMAGE_COUNT:
        raise ValueError(
            f"Expected {RIM_ONE_DL_IMAGE_COUNT} records, found {len(records)}"
        )
    grouped: dict[str, list[FundusRecord]] = {}
    for record in records:
        grouped.setdefault(record.stratum, []).append(record)
    group_sizes = {stratum: len(rows) for stratum, rows in grouped.items()}
    test_counts = _largest_remainder(
        group_sizes, RIM_ONE_DL_SPLIT_COUNTS["test"]
    )
    remaining_sizes = {
        stratum: group_sizes[stratum] - test_counts[stratum]
        for stratum in group_sizes
    }
    val_counts = _largest_remainder(
        remaining_sizes, RIM_ONE_DL_SPLIT_COUNTS["val"]
    )

    rng = random.Random(seed)
    partitions = {"train": [], "val": [], "test": []}
    for stratum in sorted(grouped):
        rows = sorted(grouped[stratum], key=lambda record: record.sample_id)
        rng.shuffle(rows)
        test_count = test_counts[stratum]
        val_count = val_counts[stratum]
        partitions["test"].extend(record.sample_id for record in rows[:test_count])
        partitions["val"].extend(
            record.sample_id
            for record in rows[test_count : test_count + val_count]
        )
        partitions["train"].extend(
            record.sample_id for record in rows[test_count + val_count :]
        )
    for split in partitions:
        partitions[split].sort()
    actual = {split: len(stems) for split, stems in partitions.items()}
    if actual != RIM_ONE_DL_SPLIT_COUNTS:
        raise AssertionError(
            f"Generator produced {actual}, expected {RIM_ONE_DL_SPLIT_COUNTS}"
        )
    return partitions


def partition_distribution(
    stems: list[str], record_by_id: dict[str, FundusRecord]
) -> dict[str, object]:
    records = [record_by_id[stem] for stem in stems]
    return {
        "total": len(records),
        "release_prefix": dict(
            sorted(Counter(record.release_prefix for record in records).items())
        ),
        "diagnosis_class": dict(
            sorted(Counter(record.diagnosis_class for record in records).items())
        ),
        "joint_stratum": dict(
            sorted(Counter(record.stratum for record in records).items())
        ),
        "hospital_split": dict(
            sorted(Counter(record.hospital_split for record in records).items())
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Parent containing RIM-ONE_DL_images and RIM-ONE-DL_masks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "splits" / "rim_one_dl.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing manifest"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Refusing to regenerate committed manifest {output_path}; use --force"
        )

    records = discover_rim_one_dl(args.data_root)
    partitions = build_partitions(records, seed=args.seed)
    record_by_id = {record.sample_id: record for record in records}
    distributions = {
        split: partition_distribution(stems, record_by_id)
        for split, stems in partitions.items()
    }
    payload = {
        "schema_version": 1,
        "dataset": "rim_one_dl",
        "seed": args.seed,
        "policy": (
            "one-time random 70/10/20 split across all 485 hospital-tree images; "
            "jointly stratified by release prefix and glaucoma/normal class"
        ),
        "source_record_count": len(records),
        "partition_distributions": distributions,
        "partitions": partitions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_rim_one_dl_split_manifest(records, output_path)

    print(f"manifest={output_path}")
    print(f"seed={args.seed}")
    for split in ("train", "val", "test"):
        print(f"{split}={json.dumps(distributions[split], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
