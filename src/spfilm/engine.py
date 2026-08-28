from __future__ import annotations

import csv
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import (
    FundusRecord,
    FundusSegmentationDataset,
    audit_records,
    discover_drishti,
    discover_refuge_training,
    discover_rim_one_dl,
    load_rim_one_dl_split_manifest,
    load_rim_one_r3_manifest,
    provider_partition,
    seed_worker,
    stratified_partition,
    validate_splits,
)
from .losses import BCEDiceLoss
from .metrics import (
    CHANNEL_NAMES,
    DEGENERATE_POLICY,
    OverlapAccumulator,
    metric_frame,
    summarise_per_image_csv,
)
from .model import PlainUNet
from .visualization import (
    save_mask_contact_sheet,
    save_prediction_gallery,
    save_training_curves,
)


@dataclass(frozen=True)
class Stage2Config:
    experiment_name: str
    dataset: str
    data_root: str
    output_dir: str
    seed: int = 42
    image_size: int = 512
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 40
    patience: int = 8
    min_epochs: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    base_channels: int = 16
    test_fraction: float = 0.20
    val_fraction: float = 0.20
    threshold: float = 0.50
    horizontal_flip_probability: float = 0.50
    rotation_degrees: float = 10.0
    brightness_contrast: float = 0.10
    requested_device: str = "auto"
    rim_manifest: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "Stage2Config":
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        return cls(**values)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _safe_output_path(project_root: Path, value: str) -> Path:
    output_path = _resolve(project_root, value)
    try:
        output_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"output_dir must remain inside {project_root}, got {output_path}"
        ) from error
    return output_path


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError("requested_device must be auto, cuda, mps, or cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def discover_config_records(
    config: Stage2Config, project_root: Path
) -> list[FundusRecord]:
    data_root = _resolve(project_root, config.data_root)
    if config.dataset == "refuge":
        return discover_refuge_training(data_root)
    if config.dataset == "drishti":
        return discover_drishti(data_root)
    if config.dataset == "rim_one_dl":
        return discover_rim_one_dl(data_root)
    if config.dataset == "rim_one_r3":
        if config.rim_manifest is None:
            raise ValueError(
                "rim_one_r3 requires rim_manifest so the annotation policy is explicit"
            )
        return load_rim_one_r3_manifest(
            data_root, _resolve(project_root, config.rim_manifest)
        )
    raise ValueError(f"Unsupported dataset {config.dataset!r}")


def build_splits(
    config: Stage2Config,
    records: list[FundusRecord],
    project_root: str | Path | None = None,
) -> dict[str, list[FundusRecord]]:
    if config.dataset == "refuge":
        return stratified_partition(
            records,
            seed=config.seed,
            test_fraction=config.test_fraction,
            val_fraction_of_remaining=config.val_fraction,
        )
    if config.dataset == "rim_one_dl":
        if config.rim_manifest is None:
            raise ValueError("rim_one_dl requires a committed rim_manifest split")
        if project_root is None:
            raise ValueError("rim_one_dl split resolution requires project_root")
        manifest_path = _resolve(Path(project_root).resolve(), config.rim_manifest)
        return load_rim_one_dl_split_manifest(records, manifest_path)
    return provider_partition(records, seed=config.seed, val_fraction=config.val_fraction)


def write_split_manifest(
    splits: dict[str, list[FundusRecord]], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "split",
                "sample_id",
                "domain",
                "stratum",
                "image_path",
                "mask_paths",
                "mask_encoding",
            ),
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            for record in splits[split]:
                writer.writerow(
                    {
                        "split": split,
                        "sample_id": record.sample_id,
                        "domain": record.domain,
                        "stratum": record.stratum,
                        "image_path": record.image_path,
                        "mask_paths": "|".join(str(path) for path in record.mask_paths),
                        "mask_encoding": record.mask_encoding,
                    }
                )
    return output_path


def _make_dataset(
    records: list[FundusRecord], config: Stage2Config, augment: bool
) -> FundusSegmentationDataset:
    return FundusSegmentationDataset(
        records,
        image_size=config.image_size,
        augment=augment,
        horizontal_flip_probability=config.horizontal_flip_probability,
        rotation_degrees=config.rotation_degrees,
        brightness_contrast=config.brightness_contrast,
    )


def _make_loader(
    dataset: FundusSegmentationDataset,
    config: Stage2Config,
    device: torch.device,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: int | None = None,
    repair_counter: dict[str, int] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch_index, (images, targets, metadata) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.shape[0]
        sample_count += images.shape[0]
        if repair_counter is not None:
            _accumulate_cup_repairs(repair_counter, metadata)
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if sample_count == 0:
        raise RuntimeError("Training loader yielded no samples")
    return total_loss / sample_count


def _accumulate_cup_repairs(counter: dict[str, int], metadata: dict[str, Any]) -> None:
    """Tally the cup-within-disc repairs the dataset applied to this batch."""

    repairs = metadata.get("cup_repair_pixels")
    if repairs is None:
        return
    pixels = [int(value) for value in repairs]
    counter["repaired_samples"] += sum(1 for value in pixels if value > 0)
    counter["repaired_pixels"] += sum(pixels)
    counter["drawn_samples"] += len(pixels)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    max_batches: int | None = None,
    per_image_csv: str | Path | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    overlap = OverlapAccumulator(threshold=threshold)
    sample_ids: list[str] = []
    image_size = 0
    hd95_unit: str | None = None
    for batch_index, (images, targets, metadata) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.shape[0]
        sample_count += images.shape[0]
        batch_hd95_unit = "letterboxed-grid pixels"
        hd95_multipliers: list[float] | None = None
        if "letterbox_scale" in metadata:
            scales = [float(value) for value in metadata["letterbox_scale"]]
            if len(scales) != images.shape[0] or any(
                not math.isfinite(scale) or scale <= 0 for scale in scales
            ):
                raise RuntimeError(
                    "RIM-ONE-DL letterbox scales must be finite, positive, and "
                    "present once per evaluated image"
                )
            hd95_multipliers = [1.0 / scale for scale in scales]
            batch_hd95_unit = "native pixels"
        if hd95_unit is None:
            hd95_unit = batch_hd95_unit
        elif hd95_unit != batch_hd95_unit:
            raise RuntimeError("Evaluation mixed incompatible HD95 coordinate frames")
        overlap.update(
            logits,
            targets,
            image_ids=metadata["sample_id"],
            hd95_multipliers=hd95_multipliers,
        )
        sample_ids.extend(metadata["sample_id"])
        image_size = targets.shape[-1]
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if sample_count == 0:
        raise RuntimeError("Evaluation loader yielded no samples")
    metrics: dict[str, Any] = {
        "loss": total_loss / sample_count,
        "evaluated_sample_count": sample_count,
        "sample_ids": sample_ids,
        "metric_frame": metric_frame(image_size),
        "degenerate_case_policy": DEGENERATE_POLICY,
    }
    if hd95_unit == "native pixels":
        metrics["hd95_unit"] = hd95_unit
    if per_image_csv is None:
        metrics.update(overlap.compute())
    else:
        # The written CSV is the single source the summary is reduced from.
        csv_path = overlap.write_per_image_csv(per_image_csv)
        metrics["per_image_csv"] = str(csv_path)
        metrics.update(summarise_per_image_csv(csv_path))
    return metrics


RIM_ONE_DL_PER_IMAGE_CONTEXT = (
    "release_prefix",
    "hospital_split",
    "diagnosis_class",
    "native_width",
    "native_height",
    "letterbox_scale",
    "hd95_unit",
)


def _append_rim_one_dl_per_image_context(
    csv_path: str | Path,
    records: Sequence[FundusRecord],
    image_size: int,
) -> Path:
    """Add RIM-only provenance and native-to-letterbox scale to metric rows."""

    csv_path = Path(csv_path)
    record_by_id = {record.sample_id: record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("Cannot annotate metrics for duplicate RIM-ONE-DL IDs")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames or any(field in fieldnames for field in RIM_ONE_DL_PER_IMAGE_CONTEXT):
        raise ValueError(f"Unexpected per-image metric schema in {csv_path}")

    for row in rows:
        sample_id = row["image_id"]
        try:
            record = record_by_id[sample_id]
        except KeyError:
            raise ValueError(
                f"Per-image metric row {sample_id!r} has no RIM-ONE-DL record"
            ) from None
        if (
            record.release_prefix is None
            or record.hospital_split is None
            or record.diagnosis_class is None
            or record.native_size is None
        ):
            raise ValueError(f"RIM-ONE-DL record {sample_id!r} lacks metric context")
        row.update(
            {
                "release_prefix": record.release_prefix,
                "hospital_split": record.hospital_split,
                "diagnosis_class": record.diagnosis_class,
                "native_width": record.native_size[0],
                "native_height": record.native_size[1],
                "letterbox_scale": f"{image_size / max(record.native_size):.12g}",
                "hd95_unit": "native_px",
            }
        )

    output_fieldnames = [fieldnames[0], *RIM_ONE_DL_PER_IMAGE_CONTEXT, *fieldnames[1:]]
    temporary_path = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)
    return csv_path


def _write_history(history: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def run_experiment(
    config: Stage2Config,
    project_root: str | Path,
    smoke: bool = False,
    records: Sequence[FundusRecord] | None = None,
    split_records: dict[str, list[FundusRecord]] | None = None,
    epoch_callback: Callable[[dict[str, float], bool], None] | None = None,
) -> dict[str, Any]:
    """Audit, split, train, and evaluate the Stage 2 single-domain baseline.

    ``records`` lets a caller supply an already-discovered record list (the same
    discovery this function would run) instead of reading the dataset twice.
    ``epoch_callback`` receives each epoch's history row and whether it was the new
    best; when given it replaces the default per-epoch print.
    """

    project_root = Path(project_root).expanduser().resolve()
    if smoke:
        config = replace(
            config,
            experiment_name=f"{config.experiment_name}_smoke",
            output_dir=f"{config.output_dir}_smoke",
            image_size=min(config.image_size, 128),
            batch_size=1,
            epochs=1,
            patience=1,
            base_channels=min(config.base_channels, 8),
        )
    output_dir = _safe_output_path(project_root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = 1 if smoke else None

    seed_everything(config.seed)
    device = choose_device(config.requested_device)
    records = (
        discover_config_records(config, project_root)
        if records is None
        else list(records)
    )
    audit = audit_records(records)
    if split_records is None:
        splits = build_splits(config, records, project_root)
    else:
        splits = {
            name: list(split_records[name]) for name in ("train", "val", "test")
        }
        validate_splits(splits, records)
    split_counts = {name: len(values) for name, values in splits.items()}

    audit["split_counts"] = split_counts
    split_policies = {
        "refuge": "deterministic stratified split inside REFUGE Training400 only",
        "rim_one_dl": (
            "committed 340/48/97 stem manifest, jointly stratified by release "
            "prefix and glaucoma/normal class"
        ),
    }
    audit["split_policy"] = split_policies.get(
        config.dataset,
        "provider test locked; validation stratified from provider train",
    )
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    write_split_manifest(splits, output_dir / "split_manifest.csv")
    save_mask_contact_sheet(
        records,
        output_dir / "mask_contact_sheet.png",
        count=12,
        seed=config.seed,
        image_size=min(config.image_size, 320),
    )

    train_dataset = _make_dataset(splits["train"], config, augment=True)
    val_dataset = _make_dataset(splits["val"], config, augment=False)
    test_dataset = _make_dataset(splits["test"], config, augment=False)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = _make_loader(train_dataset, config, device, True, generator)
    val_loader = _make_loader(val_dataset, config, device, False, generator)
    test_loader = _make_loader(test_dataset, config, device, False, generator)

    model = PlainUNet(base_channels=config.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.epochs,
        eta_min=1e-6, 
    )
    #For Adam and ReduceOnPlateuaLR() Combo# optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            # Can adjust patience (by increasing) and increas the factor to increas the time taken
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    cup_repairs = {"repaired_samples": 0, "repaired_pixels": 0, "drawn_samples": 0}
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            max_batches=max_batches,
            repair_counter=cup_repairs,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=config.threshold,
            max_batches=max_batches,
        )
        val_loss = float(val_metrics["loss"])
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Validation loss became non-finite at epoch {epoch}")
        # scheduler.step(val_loss)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_disc_dice": float(val_metrics["disc"]["dice_mean"]),
            "val_cup_dice": float(val_metrics["cup"]["dice_mean"]),
            "learning_rate": float(scheduler.get_last_lr()[0]), # float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        _write_history(history, output_dir / "history.csv")
        is_best = val_loss < best_val_loss - 1e-5
        if epoch_callback is None:
            print(
                f"epoch={epoch:02d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"disc_dice={row['val_disc_dice']:.4f} "
                f"cup_dice={row['val_cup_dice']:.4f}",
                flush=True,
            )
        else:
            epoch_callback(row, is_best)

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "config": asdict(config),
                    "channel_order": ["disc", "cup"],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        # min_epochs gates only the early-stop trigger; LR scheduling and
        # checkpointing above are untouched. Val loss is dominated by disc, so cup
        # Dice can sit near zero for many epochs before soft Dice pulls it out.
        if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
            print(f"early_stopping best_epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=config.threshold,
        max_batches=max_batches,
        per_image_csv=output_dir / "test_per_image_metrics.csv",
    )
    if config.dataset == "rim_one_dl":
        _append_rim_one_dl_per_image_context(
            output_dir / "test_per_image_metrics.csv",
            splits["test"],
            config.image_size,
        )
        rim_metric_frame = (
            f"metrics computed on the {config.image_size}px full-source-image grid; "
            "each square ONH-cropped source is resized to "
            f"{config.image_size}x{config.image_size}, and the per-image native-to-grid "
            "letterbox_scale is recorded in the metrics CSV; each HD95 value is "
            "divided by that scale and reported in native-source pixels, not "
            "letterboxed-grid pixels or millimetres"
        )
        if test_metrics.get("hd95_unit") != "native pixels":
            raise RuntimeError(
                "RIM-ONE-DL evaluation did not convert HD95 to native pixels"
            )
        test_metrics["metric_frame"] = rim_metric_frame
        test_metrics["per_image_context_fields"] = list(
            RIM_ONE_DL_PER_IMAGE_CONTEXT
        )
    save_training_curves(history, output_dir / "training_curves.png")
    save_prediction_gallery(
        model,
        test_dataset,
        device,
        output_dir / "test_predictions.png",
        threshold=config.threshold,
        count=1 if smoke else 6,
    )

    report = {
        "experiment_name": config.experiment_name,
        "smoke_test": smoke,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "checkpoint_selection": "lowest validation BCE + soft Dice loss",
        "training_seconds": time.perf_counter() - training_started,
        "split_counts": split_counts,
        "cup_within_disc_repairs": {
            **cup_repairs,
            "policy": (
                "augmented training samples whose cup leaked outside the disc were "
                "repaired in place with cup &= disc rather than raising"
            ),
        },
        "test": test_metrics,
        "reporting_rule": "Disc and cup metrics are separate; no combined Dice is reported.",
        "metric_frame": test_metrics["metric_frame"],
        "degenerate_case_policy": DEGENERATE_POLICY,
        "artifacts": {
            "test_per_image_metrics": str(output_dir / "test_per_image_metrics.csv"),
            "checkpoint": str(checkpoint_path),
            "history": str(output_dir / "history.csv"),
            "split_manifest": str(output_dir / "split_manifest.csv"),
            "data_audit": str(output_dir / "data_audit.json"),
            "mask_contact_sheet": str(output_dir / "mask_contact_sheet.png"),
            "training_curves": str(output_dir / "training_curves.png"),
            "test_predictions": str(output_dir / "test_predictions.png"),
        },
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    _print_test_results(test_metrics, config.image_size)
    return report


def _print_test_results(test_metrics: dict[str, Any], image_size: int) -> None:
    if test_metrics.get("hd95_unit") == "native pixels":
        print(
            "test results | Dice and IoU unitless | HD95 in per-image native "
            "source pixels (not mm, not letterboxed-grid pixels) | accuracy over "
            "all letterboxed-grid pixels | disc and cup separate",
            flush=True,
        )
    else:
        print(
            f"test results | Dice and IoU unitless | HD95 in {image_size}x{image_size} "
            "letterboxed-grid pixels (not mm, not native pixels) | accuracy over all "
            "pixels | disc and cup separate",
            flush=True,
        )
    for name in CHANNEL_NAMES:
        structure = test_metrics[name]
        hd95_mean = structure["hd95_mean"]
        hd95_suffix = (
            " native-px"
            if test_metrics.get("hd95_unit") == "native pixels"
            else "px"
        )
        hd95_text = (
            "undefined" if hd95_mean is None else f"{hd95_mean:.2f}{hd95_suffix}"
        )
        print(
            f"  {name:<4} dice={structure['dice_mean']:.4f} "
            f"iou={structure['iou_mean']:.4f} "
            f"hd95={hd95_text} "
            f"acc={structure['accuracy_mean']:.4f} "
            f"hd95_excluded={structure['hd95_excluded_count']}"
            f"/{structure['sample_count']}",
            flush=True,
        )
