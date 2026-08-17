#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import (  # noqa: E402
    DatasetLayoutError,
    audit_records,
    discover_drishti,
    discover_refuge_training,
    inspect_rim_download,
)
from spfilm.engine import Stage2Config, run_experiment  # noqa: E402
from spfilm.visualization import save_mask_contact_sheet  # noqa: E402


def audit_all(output_path: Path) -> dict[str, object]:
    datasets_root = WORKSPACE_ROOT / "datasets"
    report: dict[str, object] = {}
    for name, discover, root in (
        ("refuge", discover_refuge_training, datasets_root / "REFUGE"),
        ("drishti", discover_drishti, datasets_root / "DRISHTI-GS"),
    ):
        try:
            report[name] = audit_records(discover(root))
        except (DatasetLayoutError, FileNotFoundError) as error:
            report[name] = {"status": "blocked", "error": str(error)}
    report["rim_one"] = inspect_rim_download(
        datasets_root / "RIM-ONE_DL_images"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"audit_report={output_path}")
    return report


def inspect_dataset(dataset: str, output_path: Path, count: int, seed: int) -> Path:
    datasets_root = WORKSPACE_ROOT / "datasets"
    if dataset == "refuge":
        records = discover_refuge_training(datasets_root / "REFUGE")
    elif dataset == "drishti":
        records = discover_drishti(datasets_root / "DRISHTI-GS")
    else:
        raise ValueError(
            "RIM-ONE inspection is blocked until the 159-image segmentation "
            "release and explicit mask manifest are present"
        )
    result = save_mask_contact_sheet(
        records, output_path, count=count, seed=seed, image_size=320
    )
    print(f"mask_contact_sheet={result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: a reproducible, single-domain plain U-Net baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="Validate all local dataset layouts and mask contracts"
    )
    audit_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit.json",
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Render normalized disc/cup masks for visual review"
    )
    inspect_parser.add_argument("--dataset", choices=("refuge", "drishti"), required=True)
    inspect_parser.add_argument("--count", type=int, default=12)
    inspect_parser.add_argument("--seed", type=int, default=42)
    inspect_parser.add_argument("--output", type=Path)

    for command in ("train", "all"):
        train_parser = subparsers.add_parser(
            command,
            help=(
                "Run the configured baseline"
                if command == "train"
                else "Audit all data, then run the configured baseline"
            ),
        )
        train_parser.add_argument(
            "--config",
            type=Path,
            default=PROJECT_ROOT / "configs" / "stage2_refuge.json",
        )
        train_parser.add_argument(
            "--smoke",
            action="store_true",
            help="Run one small batch through forward, backward, checkpoint, and metrics",
        )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        audit_all(args.output.resolve())
        return
    if args.command == "inspect":
        output = args.output or (
            PROJECT_ROOT / "artifacts" / f"{args.dataset}_mask_contact_sheet.png"
        )
        inspect_dataset(args.dataset, output.resolve(), args.count, args.seed)
        return
    if args.command == "all":
        audit_all((PROJECT_ROOT / "artifacts" / "data_audit.json").resolve())
    config = Stage2Config.from_json(args.config.resolve())
    report = run_experiment(config, PROJECT_ROOT, smoke=args.smoke)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
