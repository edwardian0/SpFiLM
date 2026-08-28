#!/usr/bin/env python3
"""Step 2: train and test the plain U-Net backbone in-domain.

The frozen config selects REFUGE, Drishti-GS, or RIM-ONE-DL and the matching
in-domain split policy. The complete source image is aspect-preserving
letterboxed to a square canvas; the pipeline adds no crop. No SpFiLM, no Global
FiLM, and no conditioning of any kind.

This file is a thin entry point. The orchestration (audit, split, manifest,
contact sheet, train, early stop, reload best, test, figures, JSON) lives in
``spfilm.engine.run_experiment``; everything here is argument resolution and the
terminal report.

Known deviation from the project brief, recorded deliberately: deep supervision
is DEFERRED for this run. ``PlainUNet.forward`` returns a single logits tensor and
``BCEDiceLoss`` consumes a single logits tensor. Deep supervision will be added to
every arm together at a later step so the arms stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from spfilm.data import FundusRecord, FundusSegmentationDataset  # noqa: E402
from spfilm.engine import (  # noqa: E402
    Stage2Config,
    build_splits,
    choose_device,
    discover_config_records,
    run_experiment,
    seed_everything,
)
from spfilm.metrics import CHANNEL_NAMES, DEGENERATE_POLICY, metric_frame  # noqa: E402


DEFAULT_BATCH_SIZE = 8
SMOKE_BATCH_SIZE = 2
SMOKE_EPOCHS = 2
SMOKE_RECORD_COUNT = 4
FULL_POOL = {"refuge": 400, "drishti": 101, "rim_one_dl": 485}
FULL_SPLIT_EXPECTED = {
    "refuge": {"train": 256, "val": 64, "test": 80},
    "drishti": {"train": 40, "val": 10, "test": 51},
    "rim_one_dl": {"train": 340, "val": 48, "test": 97},
}
DEEP_SUPERVISION_NOTE = (
    "deep supervision is DEFERRED for this run (single-head output, single-tensor "
    "loss); it will be added to all arms together at a later step"
)
CONDITIONING_NOTE = (
    "no conditioning: this is the plain U-Net comparator, not SpFiLM or Global FiLM"
)
INPUT_POLICY_NOTE = (
    "full image, no ROI crop: the whole fundus is aspect-preserving letterboxed "
    "onto a square canvas (2124x2056 -> 512x496 centre-pasted on 512x512)"
)


def input_policy_note(dataset: str) -> str:
    if dataset == "rim_one_dl":
        return (
            "full source image, no additional ROI crop: each already ONH-cropped "
            "square source is resized onto the configured square letterbox canvas"
        )
    return INPUT_POLICY_NOTE


def labels_for_dataset(dataset: str) -> dict[str, str]:
    """Return the disk-facing labels for a supported Stage 2 dataset."""

    labels = {
        "refuge": {
            "artifact_prefix": "refuge_s2",
            "domain": "refuge_zeiss",
            "provenance": "REFUGE Training400, Zeiss Visucam",
        },
        "drishti": {
            "artifact_prefix": "drishti_s2",
            "domain": "drishti_gs",
            "provenance": "Drishti-GS",
        },

        "rim_one_dl": {
            "artifact_prefix": "rim_one_s2",
            "domain": "rim_one_dl",
            "provenance": "RIM-ONE DL (r1/r2/r3 mixture)",
        },
    }
    try:
        return labels[dataset]
    except KeyError:
        raise SystemExit(f"unsupported dataset for labels: {dataset!r}") from None


def _run(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unavailable"


def git_revision() -> str:
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit == "unavailable":
        return commit
    dirty = _run(["git", "status", "--porcelain"])
    suffix = "-dirty" if dirty and dirty != "unavailable" else ""
    return f"{commit}{suffix}"


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} {torch.cuda.get_device_name(index)}"
    if device.type == "mps":
        return f"mps Apple Metal on {platform.machine()}"
    return f"cpu {platform.processor() or platform.machine()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2 in-domain baseline: train and test PlainUNet on the "
            "config-selected fundus dataset, full-image letterboxed, no conditioning"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage2_refuge.json",
        help="Frozen JSON config; every flag below overrides it",
    )
    parser.add_argument("--seed", type=int, help="Split, init, and augmentation seed")
    parser.add_argument("--epochs", type=int, help="Epoch budget for the run")
    parser.add_argument(
        "--early-stopping-mode",
        choices=("monitor", "terminate"),
        help=(
            "monitor: run the full epoch budget, early stopping only selects the "
            "best checkpoint (default, set in config). terminate: restore the old "
            "behaviour where the rule breaks out of the loop."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Training batch size (default {DEFAULT_BATCH_SIZE}, {SMOKE_BATCH_SIZE} under --smoke)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Square letterbox canvas edge in pixels (the config's 512 is the design)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Exact run directory, must stay inside the repo "
            "(default artifacts/runs/<dataset>_s2_<timestamp>)"
        ),
    )
    parser.add_argument("--num-workers", type=int, help="DataLoader workers (default 0)")
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), help="Compute device"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Tiny end-to-end rehearsal: {SMOKE_RECORD_COUNT} images, {SMOKE_EPOCHS} "
            "epochs, full metric/CSV/table path"
        ),
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> tuple[Stage2Config, Path]:
    """Apply the CLI overrides to the frozen config and pick the run directory."""

    config = Stage2Config.from_json(args.config.resolve())
    labels = labels_for_dataset(config.dataset)
    # The full run takes batch_size/num_workers/fractions from the frozen config;
    # only --smoke overrides them. num_workers=0 is a macOS shared-memory artefact
    # and must not be forced onto Linux, where decode is the epoch bottleneck.
    overrides: dict[str, Any] = {}
    if args.smoke:
        # A four-image subset cannot survive the production split fractions; these
        # give train=2, val=1, test=1 so every stage of the path is exercised.
        overrides.update(
            batch_size=SMOKE_BATCH_SIZE,
            num_workers=0,
            epochs=SMOKE_EPOCHS,
            patience=SMOKE_EPOCHS,
            min_epochs=0,
            test_fraction=0.25,
            val_fraction=0.5,
        )
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
        overrides["patience"] = min(config.patience, args.epochs)
        overrides["min_epochs"] = min(config.min_epochs, args.epochs)
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.resolution is not None:
        overrides["image_size"] = args.resolution
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.device is not None:
        overrides["requested_device"] = args.device
    if args.early_stopping_mode is not None:
        overrides["early_stopping_mode"] = args.early_stopping_mode

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_smoke" if args.smoke else ""
    if args.out_dir is not None:
        run_dir = (
            args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
        )
    else:
        run_dir = (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / f"{labels['artifact_prefix']}{suffix}_{timestamp}"
        )
    run_dir = run_dir.resolve()
    try:
        relative = run_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        raise SystemExit(
            f"--out-dir must stay inside {PROJECT_ROOT}, got {run_dir}"
        ) from None
    overrides["experiment_name"] = (
        f"{labels['artifact_prefix']}_plain_unet{suffix}_{timestamp}"
    )
    overrides["output_dir"] = str(relative)
    return replace(config, **overrides), run_dir.resolve()


def select_smoke_records(
    records: Sequence[FundusRecord], count: int = SMOKE_RECORD_COUNT
) -> list[FundusRecord]:
    """Take a tiny subset that still splits into non-empty train/val/test.

    ``stratified_partition`` rounds within each stratum, so an even 2/2 subset would
    round the test split to zero and fail validation. Taking count-1 from the first
    stratum and one from the second leaves every split occupied.
    """

    grouped: dict[str, list[FundusRecord]] = {}
    for record in sorted(records, key=lambda record: record.sample_id):
        grouped.setdefault(record.stratum, []).append(record)
    strata = sorted(grouped)
    if len(strata) < 2 or len(grouped[strata[0]]) < count - 1:
        selected = sorted(records, key=lambda record: record.sample_id)[:count]
    else:
        selected = grouped[strata[0]][: count - 1] + grouped[strata[1]][:1]
    return sorted(selected, key=lambda record: record.sample_id)


def select_manifest_smoke_splits(
    splits: dict[str, list[FundusRecord]],
) -> dict[str, list[FundusRecord]]:
    """Take 2/1/1 records without discarding committed split membership."""

    return {
        "train": sorted(splits["train"], key=lambda record: record.sample_id)[:2],
        "val": sorted(splits["val"], key=lambda record: record.sample_id)[:1],
        "test": sorted(splits["test"], key=lambda record: record.sample_id)[:1],
    }


def print_header(config: Stage2Config, device: torch.device, run_dir: Path, smoke: bool) -> None:
    labels = labels_for_dataset(config.dataset)
    print("=" * 96, flush=True)
    print(
        "SpFiLM Step 2 | in-domain plain U-Net baseline | "
        f"{labels['provenance']}",
        flush=True,
    )
    print("=" * 96, flush=True)
    for label, value in (
        ("git_commit", git_revision()),
        ("timestamp", datetime.now().astimezone().isoformat(timespec="seconds")),
        ("hostname", socket.gethostname()),
        ("platform", f"{platform.platform()} python {platform.python_version()}"),
        ("torch", torch.__version__),
        ("device", device_description(device)),
        ("seed", config.seed),
        ("mode", "SMOKE rehearsal" if smoke else "full run"),
        ("run_dir", run_dir),
    ):
        print(f"{label:<16} {value}", flush=True)
    print("-" * 96, flush=True)
    print("resolved config (JSON defaults after CLI overrides):", flush=True)
    for line in json.dumps(asdict(config), indent=2).splitlines():
        print(f"  {line}", flush=True)
    print("-" * 96, flush=True)
    for note in (
        CONDITIONING_NOTE,
        input_policy_note(config.dataset),
        DEEP_SUPERVISION_NOTE,
    ):
        print(f"note             {note}", flush=True)
    print("=" * 96, flush=True)


def print_data_block(
    config: Stage2Config,
    records: Sequence[FundusRecord],
    strata: Sequence[str],
    *,
    smoke: bool,
    precomputed_splits: dict[str, list[FundusRecord]] | None = None,
) -> None:
    """Report the splits, the first training batch, and the target pixel balance."""

    expected_split: dict[str, int] = {}
    if not smoke:
        try:
            expected_pool = FULL_POOL[config.dataset]
            expected_split = FULL_SPLIT_EXPECTED[config.dataset]
        except KeyError:
            raise SystemExit(
                f"split check FAILED: unknown dataset {config.dataset!r}"
            ) from None
        if len(records) != expected_pool:
            raise SystemExit(
                f"split check FAILED: expected {expected_pool} records for "
                f"{config.dataset!r}, got {len(records)}"
            )

    splits = (
        build_splits(config, list(records), PROJECT_ROOT)
        if precomputed_splits is None
        else precomputed_splits
    )
    print("data", flush=True)
    header = f"  {'split':<8}{'n':>6}" + "".join(f"{name:>16}" for name in strata)
    print(header, flush=True)
    for name in ("train", "val", "test"):
        counts = [
            sum(record.stratum == stratum for record in splits[name])
            for stratum in strata
        ]
        row = f"  {name:<8}{len(splits[name]):>6}" + "".join(f"{value:>16}" for value in counts)
        print(row, flush=True)
    print(
        f"  {'total':<8}{len(records):>6}"
        + "".join(
            f"{sum(record.stratum == stratum for record in records):>16}"
            for stratum in strata
        ),
        flush=True,
    )
    print(f"  domain           {sorted({record.domain for record in records})}", flush=True)

    # Guard against a full run silently inheriting --smoke fractions or using a
    # partial provider pool. Smoke runs continue to bypass this production gate.
    if not smoke:
        actual = {name: len(splits[name]) for name in ("train", "val", "test")}
        if config.dataset == "rim_one_dl":
            print(
                "  split_source     committed splits/rim_one_dl.json "
                f"-> train/val/test = {actual['train']}/{actual['val']}/{actual['test']}",
                flush=True,
            )
        else:
            print(
                f"  fractions        test={config.test_fraction} "
                f"val={config.val_fraction} -> train/val/test = "
                f"{actual['train']}/{actual['val']}/{actual['test']}",
                flush=True,
            )
        if actual != expected_split:
            raise SystemExit(
                f"split check FAILED: expected {expected_split}, got {actual}. "
                "The full run must not inherit the --smoke fractions (0.25/0.5)."
            )
        print(
            "  split_check      PASS "
            f"({expected_split['train']}/{expected_split['val']}/"
            f"{expected_split['test']}, smoke overrides not applied)",
            flush=True,
        )

    # Same dataset construction the engine uses, so these are the tensors that will
    # actually be trained on. The engine re-seeds before building its own loaders.
    dataset = FundusSegmentationDataset(
        splits["train"],
        image_size=config.image_size,
        augment=True,
        horizontal_flip_probability=config.horizontal_flip_probability,
        rotation_degrees=config.rotation_degrees,
        brightness_contrast=config.brightness_contrast,
    )
    batch_size = min(config.batch_size, len(dataset))
    samples = [dataset[index] for index in range(batch_size)]
    images = torch.stack([sample[0] for sample in samples])
    masks = torch.stack([sample[1] for sample in samples])
    print(
        f"  first batch      images {tuple(images.shape)} {images.dtype} "
        f"[{images.min():.3f}, {images.max():.3f}] | "
        f"targets {tuple(masks.shape)} {masks.dtype} "
        f"values {sorted(set(masks.unique().tolist()))}",
        flush=True,
    )
    for channel, name in enumerate(CHANNEL_NAMES):
        fraction = float(masks[:, channel].mean())
        print(
            f"  {name + ' fraction':<16} {fraction:.5f} of the "
            f"{config.image_size}x{config.image_size} canvas "
            f"(first training batch, augmented)",
            flush=True,
        )
    print("=" * 96, flush=True)


def epoch_line_header() -> str:
    return (
        f"{'epoch':>7} | {'lr':>9} | {'train_loss':>10} | {'val_loss':>10} | "
        f"{'val_dice_disc':>13} | {'val_dice_cup':>12} | {'time':>8} | best"
    )


def make_epoch_printer(total_epochs: int):
    def print_epoch(row: dict[str, float], is_best: bool) -> None:
        print(
            f"{int(row['epoch']):>3d}/{total_epochs:<3d} | "
            f"{row['learning_rate']:>9.3e} | "
            f"{row['train_loss']:>10.4f} | "
            f"{row['val_loss']:>10.4f} | "
            f"{row['val_disc_dice']:>13.4f} | "
            f"{row['val_cup_dice']:>12.4f} | "
            f"{row['epoch_seconds']:>7.1f}s | "
            f"{'*' if is_best else ''}",
            flush=True,
        )

    return print_epoch


def _cell(mean: float | None, std: float | None, count: int) -> str:
    if mean is None or std is None:
        return f"undefined (n={count})"
    return f"{mean:.4f} ± {std:.4f} (n={count})"


def _hd95_cell(structure: dict[str, Any]) -> str:
    kept = int(structure["hd95_sample_count"])
    excluded = int(structure["hd95_excluded_count"])
    mean = structure["hd95_mean"]
    std = structure["hd95_std"]
    # HD95 never prints bare: the exclusion count is part of the number.
    if mean is None or std is None:
        return f"undefined (n={kept}, excluded={excluded})"
    return f"{mean:.2f} ± {std:.2f} (n={kept}, excluded={excluded})"


def format_final_block(report: dict[str, Any], config: Stage2Config) -> str:
    test = report["test"]
    labels = labels_for_dataset(config.dataset)
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append(
        f"TEST RESULTS (in-domain: trained and tested on {labels['domain']})"
    )
    lines.append("=" * 96)
    lines.append(f"experiment        {report['experiment_name']}")
    lines.append(f"device            {report['device']}")
    lines.append(f"parameters        {report['parameter_count']:,}")
    lines.append(
        f"best epoch        {report['best_epoch']} selected on "
        f"{report['checkpoint_selection']}"
    )
    lines.append(f"training seconds  {report['training_seconds']:.1f}")
    lines.append(f"split counts      {report['split_counts']}")
    repairs = report["cup_within_disc_repairs"]
    lines.append(
        f"cup<=disc repairs {repairs['repaired_samples']} of "
        f"{repairs['drawn_samples']} augmented samples drawn "
        f"({repairs['repaired_pixels']} pixels total, repaired with cup &= disc)"
    )
    lines.append("")
    lines.append(f"metric frame      {test['metric_frame']}")
    lines.append("degenerate-case policy:")
    for key, value in DEGENERATE_POLICY.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append(f"threshold         {config.threshold} (hard Dice, distinct from the soft Dice in the loss)")
    lines.append("reporting rule    " + report["reporting_rule"])
    lines.append("")

    hd95_label = (
        "HD95 (native px)" if config.dataset == "rim_one_dl" else "HD95 (px)"
    )
    columns = ("Dice", "IoU", hd95_label, "Accuracy")
    widths = (26, 26, 34, 26)
    head = f"{'structure':<10}" + "".join(
        f"{name:<{width}}" for name, width in zip(columns, widths)
    )
    lines.append(head)
    lines.append("-" * len(head))
    for name in CHANNEL_NAMES:
        structure = test[name]
        count = int(structure["sample_count"])
        cells = (
            _cell(structure["dice_mean"], structure["dice_std"], count),
            _cell(structure["iou_mean"], structure["iou_std"], count),
            _hd95_cell(structure),
            _cell(structure["accuracy_mean"], structure["accuracy_std"], count),
        )
        lines.append(
            f"{name:<10}"
            + "".join(f"{cell:<{width}}" for cell, width in zip(cells, widths))
        )
    lines.append("-" * len(head))
    lines.append("mean ± std over per-image values; disc and cup are never averaged together")
    lines.append(f"per-image source  {test['per_image_csv']}")
    lines.append("")
    lines.append(f"known deviation   {DEEP_SUPERVISION_NOTE}")
    lines.append("=" * 96)
    return "\n".join(lines)


def write_run_notes(run_dir: Path, config: Stage2Config, report: dict[str, Any]) -> Path:
    path = run_dir / "RUN_NOTES.md"
    repairs = report["cup_within_disc_repairs"]
    labels = labels_for_dataset(config.dataset)
    path.write_text(
        "\n".join(
            (
                f"# {report['experiment_name']}",
                "",
                f"- Step 2, in-domain: trained and tested on `{labels['domain']}` "
                f"({labels['provenance']}) only.",
                f"- Git commit: `{git_revision()}`",
                f"- Device: {report['device']}; seed {config.seed}; "
                f"{config.image_size}px letterbox canvas.",
                f"- Splits: {report['split_counts']}.",
                "",
                "## Known deviations",
                "",
                f"- **Deep supervision: deferred.** {DEEP_SUPERVISION_NOTE}.",
                "",
                "## Design decisions in force",
                "",
                "- 2-channel sigmoid head (disc, cup); not softmax over exclusive classes.",
                "- InstanceNorm (affine) after every convolution; no BatchNorm.",
                "- Loss: equal-weight BCEWithLogits + soft Dice.",
                "- Metrics per image, disc and cup reported separately, hard Dice at "
                f"threshold {config.threshold}.",
                f"- {input_policy_note(config.dataset)}.",
                f"- {CONDITIONING_NOTE}.",
                "",
                "## Data integrity",
                "",
                f"- Cup-within-disc repairs: {repairs['repaired_samples']} of "
                f"{repairs['drawn_samples']} augmented samples drawn "
                f"({repairs['repaired_pixels']} pixels), repaired in place with "
                "`cup &= disc` rather than aborting the run.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    args = parse_args()
    config, run_dir = resolve_config(args)
    device = choose_device(config.requested_device)

    print_header(config, device, run_dir, args.smoke)

    seed_everything(config.seed)
    records = discover_config_records(config, PROJECT_ROOT)
    smoke_splits: dict[str, list[FundusRecord]] | None = None
    if args.smoke:
        if config.dataset == "rim_one_dl":
            full_splits = build_splits(config, records, PROJECT_ROOT)
            smoke_splits = select_manifest_smoke_splits(full_splits)
            records = sorted(
                (
                    record
                    for split in ("train", "val", "test")
                    for record in smoke_splits[split]
                ),
                key=lambda record: record.sample_id,
            )
        else:
            records = select_smoke_records(records)
        print(
            f"smoke subset      {len(records)} records: "
            f"{[record.sample_id for record in records]}",
            flush=True,
        )
    strata = sorted({record.stratum for record in records})
    print_data_block(
        config,
        records,
        strata,
        smoke=args.smoke,
        precomputed_splits=smoke_splits,
    )

    total_epochs = config.epochs
    print("training (no tqdm; one line per epoch, * marks a new best checkpoint)", flush=True)
    print(epoch_line_header(), flush=True)
    print("-" * len(epoch_line_header()), flush=True)

    # smoke=False on purpose: the engine's own smoke mode shrinks the image to 128px
    # and truncates every loader to one batch, which would skip most of the metric
    # path. --smoke here shrinks the record list instead and runs the path in full.
    report = run_experiment(
        config,
        PROJECT_ROOT,
        smoke=False,
        records=records,
        split_records=smoke_splits,
        epoch_callback=make_epoch_printer(total_epochs),
    )

    summary = format_final_block(report, config)
    print(summary, flush=True)
    (run_dir / "summary_table.txt").write_text(summary + "\n", encoding="utf-8")
    write_run_notes(run_dir, config, report)
    print("artifacts", flush=True)
    for name, path in sorted(report["artifacts"].items()):
        print(f"  {name:<26} {path}", flush=True)
    for name in ("summary_table.txt", "RUN_NOTES.md", "test_metrics.json", "resolved_config.json"):
        print(f"  {name:<26} {run_dir / name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
