from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    FundusRecord,
    FundusSegmentationDataset,
    audit_records,
    discover_drishti,
    discover_refuge_training,
    load_rim_one_r3_manifest,
    provider_partition,
    seed_worker,
    stratified_partition,
)
from .losses import BCEDiceLoss
from .metrics import OverlapAccumulator
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
    config: Stage2Config, records: list[FundusRecord]
) -> dict[str, list[FundusRecord]]:
    if config.dataset == "refuge":
        return stratified_partition(
            records,
            seed=config.seed,
            test_fraction=config.test_fraction,
            val_fraction_of_remaining=config.val_fraction,
        )
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
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch_index, (images, targets, _) in enumerate(loader):
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
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if sample_count == 0:
        raise RuntimeError("Training loader yielded no samples")
    return total_loss / sample_count


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    overlap = OverlapAccumulator(threshold=threshold)
    sample_ids: list[str] = []
    for batch_index, (images, targets, metadata) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.shape[0]
        sample_count += images.shape[0]
        overlap.update(logits, targets)
        sample_ids.extend(metadata["sample_id"])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if sample_count == 0:
        raise RuntimeError("Evaluation loader yielded no samples")
    metrics: dict[str, Any] = {
        "loss": total_loss / sample_count,
        "evaluated_sample_count": sample_count,
        "sample_ids": sample_ids,
    }
    metrics.update(overlap.compute())
    return metrics


def _write_history(history: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def run_experiment(
    config: Stage2Config,
    project_root: str | Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Audit, split, train, and evaluate the Stage 2 single-domain baseline."""

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
    records = discover_config_records(config, project_root)
    audit = audit_records(records)
    splits = build_splits(config, records)
    split_counts = {name: len(values) for name, values in splits.items()}

    audit["split_counts"] = split_counts
    audit["split_policy"] = (
        "deterministic stratified split inside REFUGE Training400 only"
        if config.dataset == "refuge"
        else "provider test locked; validation stratified from provider train"
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
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
        scheduler.step(val_loss)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_disc_dice": float(val_metrics["disc"]["dice_mean"]),
            "val_cup_dice": float(val_metrics["cup"]["dice_mean"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        _write_history(history, output_dir / "history.csv")
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"disc_dice={row['val_disc_dice']:.4f} "
            f"cup_dice={row['val_cup_dice']:.4f}"
        )

        if val_loss < best_val_loss - 1e-5:
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
        if epochs_without_improvement >= config.patience:
            print(f"early_stopping best_epoch={best_epoch}")
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
        "test": test_metrics,
        "reporting_rule": "Disc and cup metrics are separate; no combined Dice is reported.",
        "artifacts": {
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
    print(
        f"test_disc_dice={test_metrics['disc']['dice_mean']:.4f} "
        f"test_cup_dice={test_metrics['cup']['dice_mean']:.4f}"
    )
    return report
