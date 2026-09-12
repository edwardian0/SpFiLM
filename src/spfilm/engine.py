from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
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
from .film.conditioning import (
    DESCRIPTOR_NAMES,
    DESCRIPTOR_POLICY,
    SELECTION_POLICY,
    ConditionResult,
    ConditioningError,
    DomainVocabulary,
    FixedCondition,
    NearestCondition,
    NearestDomainSelector,
    OracleCondition,
)
from .losses import BCEDiceLoss
from .metrics import (
    CHANNEL_NAMES,
    DEGENERATE_POLICY,
    OverlapAccumulator,
    metric_frame,
    summarise_per_image_csv,
)
from .model import ARMS, build_model
from .visualization import (
    save_mask_contact_sheet,
    save_prediction_gallery,
    save_training_curves,
)


def _save_checkpoint(
    path, model, optimizer, epoch, val_metrics, config, conditioning=None
) -> None:
    """Single writer for both checkpoints so their payload schemas cannot drift.

    ``conditioning`` (conditioned arms only) carries the domain vocabulary and
    the fitted nearest-domain selector, because a FiLM checkpoint is unusable
    without knowing which code means which domain and how test images get one.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "validation_metrics": val_metrics,
        "config": asdict(config),
        "channel_order": ["disc", "cup"],
        "arm": config.arm,
    }
    if conditioning is not None:
        payload["conditioning"] = conditioning
    torch.save(payload, path)


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
    # "monitor": the stopping rule is evaluated and logged every epoch but never
    # ends the loop, so every run consumes the full epoch budget and the schedule
    # is identical across datasets, seeds and (later) conditioning arms.
    # "terminate": pre-2026-08 behaviour, the rule breaks out of the loop.
    early_stopping_mode: str = "monitor"
    early_stopping_min_delta: float = 1e-5
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
    # Conditioning arm. "plain" is the Stage 2/3 backbone with no conditioning;
    # "global_film" adds channel-wise FiLM after each encoder block (Step 4).
    # The film_* fields are ignored by the plain arm and, so that in-flight plain
    # runs keep resuming, are left out of the plain resume fingerprint.
    arm: str = "plain"
    film_levels: int = 5
    film_embedding_dim: int = 64
    film_hidden_dim: int = 256
    film_clamp: float = 5.0
    # How held-out test images get a domain code. "nearest_domain": the source
    # domain whose training descriptor centroid is nearest (the supervisor's
    # policy for unseen domains). "oracle": the true code, valid only when every
    # test image's domain is in the training vocabulary (in-domain checks).
    test_conditioning: str = "nearest_domain"

    @classmethod
    def from_json(cls, path: str | Path) -> "Stage2Config":
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        return cls(**values)


FILM_CONFIG_FIELDS = (
    "arm",
    "film_levels",
    "film_embedding_dim",
    "film_hidden_dim",
    "film_clamp",
    "test_conditioning",
)
TEST_CONDITIONING_POLICIES = ("nearest_domain", "oracle")


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


RESUME_STATE_FILENAME = "resume_state.pt"


def _resume_fingerprint(config: Stage2Config, split_counts: dict[str, int]) -> str:
    """Identity of the run a resume file belongs to; a mismatch must never resume.

    The plain arm's fingerprint is computed exactly as before the conditioning
    fields existed, so a plain run preempted under the old code resumes under
    the new. A conditioned arm hashes every field.
    """

    config_payload = asdict(config)
    if config.arm == "plain":
        for field in FILM_CONFIG_FIELDS:
            config_payload.pop(field, None)
    payload = json.dumps(
        {"config": config_payload, "split_counts": split_counts}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_resume_state(path: Path, **state: Any) -> None:
    """Write everything needed to continue after preemption, atomically.

    Preemption can land mid-write, so the payload goes to a temporary file and is
    renamed into place. A half-written resume file would be worse than none.
    """

    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            **state,
        },
        temporary,
    )
    temporary.replace(path)


def _load_resume_state(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """Return resume state only when it provably belongs to this exact run."""

    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported resume-state schema in {path}")
    if state.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Resume state in {path} was written for a different config or split. "
            "Delete the run directory and start again rather than resuming into it."
        )
    return state


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


ConditionFn = Callable[[torch.Tensor, Mapping[str, Any]], ConditionResult]


def _forward(
    model: torch.nn.Module,
    images: torch.Tensor,
    condition: ConditionResult | None,
) -> torch.Tensor:
    """The plain arm takes images alone; a conditioned arm also takes its codes."""

    if condition is None:
        return model(images)
    return model(images, condition.indices)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: int | None = None,
    repair_counter: dict[str, int] | None = None,
    condition_fn: ConditionFn | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch_index, (images, targets, metadata) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        condition = None if condition_fn is None else condition_fn(images, metadata)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            logits = _forward(model, images, condition)
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
    native_hd95: bool = False,
    condition_fn: ConditionFn | None = None,
) -> dict[str, Any]:
    """Score one loader. With ``condition_fn`` the per-image codes are returned too.

    ``metrics["conditioning"]`` then holds one row per image recording which
    code was used, where it came from, and (when a selector chose it) the
    distances and descriptor it saw, so a held-out result can later be broken
    down by the code it was scored under.
    """

    model.eval()
    total_loss = 0.0
    sample_count = 0
    overlap = OverlapAccumulator(threshold=threshold)
    sample_ids: list[str] = []
    conditioning_rows: list[dict[str, Any]] = []
    image_size = 0
    hd95_unit: str | None = None
    for batch_index, (images, targets, metadata) in enumerate(loader):
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        condition = None if condition_fn is None else condition_fn(images, metadata)
        if condition is not None:
            conditioning_rows.extend(
                _conditioning_rows(condition, metadata, condition_fn.vocabulary)
            )
        logits = _forward(model, images, condition)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.shape[0]
        sample_count += images.shape[0]
        batch_hd95_unit = (
            "native pixels" if native_hd95 else "letterboxed-grid pixels"
        )
        hd95_multipliers: list[float] | None = None
        if native_hd95:
            domains = [str(value) for value in metadata["domain"]]
            if any(domain != "rim_one_dl" for domain in domains):
                raise RuntimeError(
                    "Native-pixel HD95 is only defined for RIM-ONE-DL batches"
                )
            scales = [float(value) for value in metadata["letterbox_scale"]]
            if len(scales) != images.shape[0] or any(
                not math.isfinite(scale) or scale <= 0 for scale in scales
            ):
                raise RuntimeError(
                    "RIM-ONE-DL letterbox scales must be finite, positive, and "
                    "present once per evaluated image"
                )
            hd95_multipliers = [1.0 / scale for scale in scales]
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
    if condition_fn is not None:
        metrics["conditioning"] = {
            "source": condition_fn.source,
            "vocabulary": condition_fn.vocabulary.to_json(),
            "rows": conditioning_rows,
        }
    return metrics


def _conditioning_rows(
    condition: ConditionResult,
    metadata: Mapping[str, Any],
    vocabulary: DomainVocabulary,
) -> list[dict[str, Any]]:
    """One record per image: the code used and what the selector saw, if any."""

    sample_ids = [str(value) for value in metadata["sample_id"]]
    true_domains = [str(value) for value in metadata["domain"]]
    indices = condition.indices.detach().cpu().tolist()
    if len(indices) != len(sample_ids):
        raise ConditioningError(
            f"{len(indices)} codes for {len(sample_ids)} images"
        )
    distances = (
        None if condition.distances is None else condition.distances.detach().cpu()
    )
    descriptors = (
        None
        if condition.descriptors is None
        else condition.descriptors.detach().cpu()
    )
    rows: list[dict[str, Any]] = []
    for position, (sample_id, true_domain, index) in enumerate(
        zip(sample_ids, true_domains, indices)
    ):
        row: dict[str, Any] = {
            "image_id": sample_id,
            "true_domain": true_domain,
            "selected_domain": vocabulary.domains[int(index)],
            "condition_source": condition.source,
        }
        for domain_position, domain in enumerate(vocabulary.domains):
            row[f"distance_{domain}"] = (
                float(distances[position, domain_position])
                if distances is not None
                else ""
            )
        for name_position, name in enumerate(DESCRIPTOR_NAMES):
            row[name] = (
                float(descriptors[position, name_position])
                if descriptors is not None
                else ""
            )
        rows.append(row)
    return rows


def write_conditioning_csv(
    rows: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Write the per-image conditioning records next to the metric CSV."""

    if not rows:
        raise RuntimeError("Cannot write conditioning records before any samples")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def summarise_conditioning(
    rows: Sequence[Mapping[str, Any]], vocabulary: DomainVocabulary
) -> dict[str, Any]:
    """Assignment counts and, when the selector ran, the mean nearest distance."""

    counts = {domain: 0 for domain in vocabulary.domains}
    nearest: list[float] = []
    for row in rows:
        counts[str(row["selected_domain"])] += 1
        candidates = [
            row[f"distance_{domain}"]
            for domain in vocabulary.domains
            if row[f"distance_{domain}"] != ""
        ]
        if candidates:
            nearest.append(min(float(value) for value in candidates))
    true_domains = sorted({str(row["true_domain"]) for row in rows})
    return {
        "sample_count": len(rows),
        "true_domains": true_domains,
        "true_domains_in_vocabulary": all(
            domain in vocabulary.domains for domain in true_domains
        ),
        "assignment_counts": counts,
        "mean_nearest_distance": (
            float(np.mean(nearest)) if nearest else None
        ),
    }


def selector_confusion(
    rows: Sequence[Mapping[str, Any]], vocabulary: DomainVocabulary
) -> dict[str, Any]:
    """How often the selector recovers the true domain on source-domain images."""

    confusion = {
        true: {selected: 0 for selected in vocabulary.domains}
        for true in vocabulary.domains
    }
    correct = 0
    for row in rows:
        true = str(row["true_domain"])
        selected = str(row["selected_domain"])
        if true not in confusion:
            raise ConditioningError(
                f"Selector accuracy is only defined on vocabulary domains, got {true!r}"
            )
        confusion[true][selected] += 1
        correct += int(true == selected)
    return {
        "sample_count": len(rows),
        "accuracy": (correct / len(rows)) if rows else None,
        "confusion": confusion,
    }


RIM_ONE_DL_PER_IMAGE_CONTEXT = (
    "release_prefix",
    "hospital_split",
    "diagnosis_class",
    "native_width",
    "native_height",
    "letterbox_scale",
    "hd95_unit",
)
HD95_UNIT_NATIVE = "native pixels"


def _rim_one_dl_metric_frame(image_size: int) -> str:
    return (
        f"metrics computed on the {image_size}px full-source-image grid; "
        "each square ONH-cropped source is resized to "
        f"{image_size}x{image_size}, and the per-image native-to-grid "
        "letterbox_scale is recorded in the metrics CSV; each HD95 value is "
        "divided by that scale and reported in native-source pixels, not "
        "letterboxed-grid pixels or millimetres"
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


def evaluate_named_test_set(
    model: torch.nn.Module,
    records: Sequence[FundusRecord],
    config: Stage2Config,
    device: torch.device,
    criterion: torch.nn.Module,
    output_dir: Path,
    name: str,
    smoke: bool = False,
    condition_fn: ConditionFn | None = None,
) -> dict[str, Any]:
    """Score one named test set on its own and write its metrics and overlays.

    The single-source arm scores several unseen target domains with one trained
    model. Each is evaluated separately rather than pooled, because Dice averaged
    over a mixture of domains hides exactly the per-domain gap the experiment
    exists to measure, and because HD95 is only comparable within one coordinate
    frame: RIM-ONE-DL is reported in native source pixels and the others in
    letterboxed-grid pixels, so a pooled HD95 would mix units.
    """

    if not records:
        raise ValueError(f"Named test set {name!r} is empty")
    dataset = _make_dataset(list(records), config, augment=False)
    generator = torch.Generator().manual_seed(config.seed)
    loader = _make_loader(dataset, config, device, False, generator)
    native_hd95 = all(record.domain == "rim_one_dl" for record in records)
    per_image_csv = output_dir / f"test_{name}_per_image_metrics.csv"
    metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        threshold=config.threshold,
        max_batches=1 if smoke else None,
        per_image_csv=per_image_csv,
        native_hd95=native_hd95,
        condition_fn=condition_fn,
    )
    if native_hd95:
        _append_rim_one_dl_per_image_context(
            per_image_csv, list(records), config.image_size
        )
        if metrics.get("hd95_unit") != HD95_UNIT_NATIVE:
            raise RuntimeError(
                f"{name} evaluation did not convert HD95 to native pixels"
            )
        metrics["metric_frame"] = _rim_one_dl_metric_frame(config.image_size)
        metrics["per_image_context_fields"] = list(RIM_ONE_DL_PER_IMAGE_CONTEXT)
    gallery_path = output_dir / f"test_{name}_predictions.png"
    save_prediction_gallery(
        model,
        dataset,
        device,
        gallery_path,
        threshold=config.threshold,
        count=1 if smoke else 6,
        predict=_gallery_predictor(model, condition_fn),
    )
    metrics["artifacts"] = {
        "per_image_metrics": str(per_image_csv),
        "predictions": str(gallery_path),
    }
    if condition_fn is not None:
        conditioning_csv = output_dir / f"test_{name}_conditioning_per_image.csv"
        rows = metrics["conditioning"].pop("rows")
        write_conditioning_csv(rows, conditioning_csv)
        metrics["conditioning"].update(
            summarise_conditioning(rows, condition_fn.vocabulary)
        )
        metrics["conditioning"]["per_image_csv"] = str(conditioning_csv)
        metrics["artifacts"]["conditioning_per_image"] = str(conditioning_csv)
    return metrics


def _gallery_predictor(
    model: torch.nn.Module, condition_fn: ConditionFn | None
) -> Callable[[torch.Tensor, Mapping[str, Any]], torch.Tensor] | None:
    """A conditioned arm's overlays must show the prediction under its real code."""

    if condition_fn is None:
        return None

    def predict(images: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        return _forward(model, images, condition_fn(images, metadata))

    return predict


def _conditioning_report(
    *,
    model: torch.nn.Module,
    config: Stage2Config,
    device: torch.device,
    criterion: torch.nn.Module,
    vocabulary: DomainVocabulary,
    selector: NearestDomainSelector,
    test_condition: ConditionFn,
    test_metrics: dict[str, Any],
    val_loader: DataLoader,
    score_test: Callable[[Path, ConditionFn | None], dict[str, Any]],
    output_dir: Path,
    max_batches: int | None,
) -> dict[str, Any]:
    """Everything needed to interpret a conditioned arm's held-out score.

    Three things are recorded. (1) Which code each test image was scored under,
    so Dice can be broken down by assigned domain. (2) The selector's accuracy
    on source-domain validation images, whose true domain is known: if it cannot
    tell the source domains apart, "nearest domain" on an unseen one means
    little. (3) A fixed-code sweep, scoring the held-out set once under each
    source code. If all codes give the same Dice the conditioning is inert and
    the selector is irrelevant; if they differ, the sweep shows whether the
    selector found the best code. These are inference-only and cheap.
    """

    artifacts: dict[str, str] = {}

    test_rows = test_metrics["conditioning"].pop("rows")
    conditioning_csv = write_conditioning_csv(
        test_rows, output_dir / "test_conditioning_per_image.csv"
    )
    test_metrics["conditioning"].update(summarise_conditioning(test_rows, vocabulary))
    test_metrics["conditioning"]["per_image_csv"] = str(conditioning_csv)
    artifacts["test_conditioning_per_image"] = str(conditioning_csv)

    validation_rows: list[dict[str, Any]] = []
    probe = NearestCondition(selector)
    with torch.inference_mode():
        for batch_index, (images, _targets, metadata) in enumerate(val_loader):
            images = images.to(device, non_blocking=device.type == "cuda")
            validation_rows.extend(
                _conditioning_rows(probe(images, metadata), metadata, vocabulary)
            )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    validation_csv = write_conditioning_csv(
        validation_rows, output_dir / "val_selector_per_image.csv"
    )
    artifacts["val_selector_per_image"] = str(validation_csv)
    selector_validation = selector_confusion(validation_rows, vocabulary)
    selector_validation["per_image_csv"] = str(validation_csv)

    sweep: dict[str, Any] = {}
    for domain in vocabulary.domains:
        csv_path = output_dir / f"test_fixed_code_{domain}_per_image_metrics.csv"
        metrics = score_test(csv_path, FixedCondition(vocabulary, domain))
        metrics.pop("conditioning", None)
        metrics.pop("sample_ids", None)
        sweep[domain] = metrics
        artifacts[f"test_fixed_code_{domain}_per_image_metrics"] = str(csv_path)

    best_fixed_code: dict[str, str] = {}
    nearest_matches_best: dict[str, bool] = {}
    nearest_minus_best: dict[str, float] = {}
    for structure in CHANNEL_NAMES:
        best = max(
            vocabulary.domains,
            key=lambda domain: float(sweep[domain][structure]["dice_mean"]),
        )
        best_fixed_code[structure] = best
        nearest_dice = float(test_metrics[structure]["dice_mean"])
        best_dice = float(sweep[best][structure]["dice_mean"])
        nearest_minus_best[structure] = nearest_dice - best_dice
        nearest_matches_best[structure] = math.isclose(
            nearest_dice, best_dice, rel_tol=0.0, abs_tol=1e-9
        )

    return {
        "arm": config.arm,
        "vocabulary": vocabulary.to_json(),
        "train_val_conditioning": OracleCondition.source,
        "test_conditioning": config.test_conditioning,
        "test_condition_source": test_condition.source,
        "descriptor": list(DESCRIPTOR_NAMES),
        "descriptor_policy": DESCRIPTOR_POLICY,
        "selection_policy": SELECTION_POLICY,
        "film": {
            "levels": config.film_levels,
            "embedding_dim": config.film_embedding_dim,
            "hidden_dim": config.film_hidden_dim,
            "clamp": config.film_clamp,
        },
        "selector": {
            "path": str(output_dir / "domain_selector.json"),
            "fitted_counts": dict(
                zip(vocabulary.domains, selector.fitted_counts)
            ),
            "centroids": {
                domain: dict(zip(DESCRIPTOR_NAMES, selector.centroids[i].tolist()))
                for i, domain in enumerate(vocabulary.domains)
            },
        },
        "test": test_metrics["conditioning"],
        "selector_validation": selector_validation,
        "fixed_code_sweep": {
            domain: {
                structure: {
                    key: sweep[domain][structure][key]
                    for key in ("dice_mean", "iou_mean", "hd95_mean", "sample_count")
                }
                for structure in CHANNEL_NAMES
            }
            for domain in vocabulary.domains
        },
        "best_fixed_code": best_fixed_code,
        "nearest_domain_matches_best_fixed_code": nearest_matches_best,
        "nearest_domain_minus_best_fixed_code_dice": nearest_minus_best,
        "artifacts": artifacts,
    }


def run_experiment(
    config: Stage2Config,
    project_root: str | Path,
    smoke: bool = False,
    records: Sequence[FundusRecord] | None = None,
    split_records: dict[str, list[FundusRecord]] | None = None,
    epoch_callback: Callable[[dict[str, float], bool], None] | None = None,
    split_policy: str | None = None,
    allow_resume: bool = False,
    extra_test_sets: Mapping[str, Sequence[FundusRecord]] | None = None,
) -> dict[str, Any]:
    """Audit, split, train, and evaluate the Stage 2 single-domain baseline.

    ``records`` lets a caller supply an already-discovered record list (the same
    discovery this function would run) instead of reading the dataset twice.
    ``epoch_callback`` receives each epoch's history row and whether it was the new
    best; when given it replaces the default per-epoch print. ``extra_test_sets``
    names further already-unseen test sets to score with the same selected
    checkpoint, each reported separately under ``test_by_name``.
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
    audit["split_policy"] = split_policy or split_policies.get(
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
    val_native_hd95 = all(
        record.domain == "rim_one_dl" for record in splits["val"]
    )
    test_native_hd95 = all(
        record.domain == "rim_one_dl" for record in splits["test"]
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = _make_loader(train_dataset, config, device, True, generator)
    val_loader = _make_loader(val_dataset, config, device, False, generator)
    test_loader = _make_loader(test_dataset, config, device, False, generator)

    if config.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {config.arm!r}")
    if config.test_conditioning not in TEST_CONDITIONING_POLICIES:
        raise ValueError(
            f"test_conditioning must be one of {TEST_CONDITIONING_POLICIES}, got "
            f"{config.test_conditioning!r}"
        )
    conditioned = config.arm != "plain"
    vocabulary: DomainVocabulary | None = None
    selector: NearestDomainSelector | None = None
    train_condition: ConditionFn | None = None
    test_condition: ConditionFn | None = None
    checkpoint_conditioning: dict[str, Any] | None = None
    if conditioned:
        # Codes are fold-local: only the domains that will see gradients get one.
        vocabulary = DomainVocabulary.from_domains(
            record.domain for record in splits["train"]
        )
        if config.test_conditioning == "oracle":
            missing = sorted(
                {record.domain for record in splits["test"]} - set(vocabulary.domains)
            )
            if missing:
                raise ConditioningError(
                    "test_conditioning='oracle' needs every test domain in the "
                    f"training vocabulary {list(vocabulary.domains)}; missing {missing}"
                )
        # The nearest-domain selector depends only on the training images, never
        # on the network, so it is fitted once up front (on un-augmented images,
        # so the brightness/contrast jitter cannot inflate the reference spread)
        # and is recomputed identically on a resume.
        selector_loader = _make_loader(
            _make_dataset(splits["train"], config, augment=False),
            config,
            device,
            False,
            torch.Generator().manual_seed(config.seed),
        )
        selector = NearestDomainSelector.fit_from_loader(selector_loader, vocabulary)
        selector.save(output_dir / "domain_selector.json")
        train_condition = OracleCondition(vocabulary)
        test_condition = (
            NearestCondition(selector)
            if config.test_conditioning == "nearest_domain"
            else OracleCondition(vocabulary)
        )
        checkpoint_conditioning = {
            "domain_vocabulary": vocabulary.to_json(),
            "domain_selector": selector.to_json(),
            "train_val_conditioning": OracleCondition.source,
            "test_conditioning": config.test_conditioning,
        }
        print(
            f"conditioning | arm={config.arm} | codes={list(vocabulary.domains)} | "
            f"train/val=oracle | test={config.test_conditioning} | "
            f"film_levels={config.film_levels}",
            flush=True,
        )

    model = build_model(
        config.arm,
        config.base_channels,
        num_domains=len(vocabulary) if vocabulary is not None else None,
        film_levels=config.film_levels,
        embedding_dim=config.film_embedding_dim,
        hidden_dim=config.film_hidden_dim,
        clamp=config.film_clamp,
    ).to(device)
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

    if config.early_stopping_mode not in {"monitor", "terminate"}:
        raise ValueError(
            "early_stopping_mode must be 'monitor' or 'terminate', got "
            f"{config.early_stopping_mode!r}"
        )
    checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"
    resume_path = output_dir / RESUME_STATE_FILENAME
    fingerprint = _resume_fingerprint(config, split_counts)
    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    would_have_stopped_at_epoch: int | None = None
    cup_repairs = {"repaired_samples": 0, "repaired_pixels": 0, "drawn_samples": 0}
    start_epoch = 1
    resumed_from_epoch: int | None = None
    previously_elapsed = 0.0

    resume_state = (
        _load_resume_state(resume_path, fingerprint) if allow_resume else None
    )
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        # The loaders hold this generator by reference and only read it when
        # iterated, so restoring its state here restores the shuffle stream.
        generator.set_state(resume_state["generator_state"])
        history = [dict(row) for row in resume_state["history"]]
        best_val_loss = float(resume_state["best_val_loss"])
        best_epoch = int(resume_state["best_epoch"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        would_have_stopped_at_epoch = resume_state["would_have_stopped_at_epoch"]
        cup_repairs = dict(resume_state["cup_repairs"])
        previously_elapsed = float(resume_state["elapsed_seconds"])
        rng = resume_state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        start_epoch = int(resume_state["epoch"]) + 1
        resumed_from_epoch = start_epoch
        print(
            f"resuming after preemption: {len(history)} epochs already done, "
            f"continuing from epoch {start_epoch}/{config.epochs}",
            flush=True,
        )

    training_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
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
            condition_fn=train_condition,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=config.threshold,
            max_batches=max_batches,
            native_hd95=val_native_hd95,
            condition_fn=train_condition,
        )
        # Per-image code rows are for the test report, not for every epoch's
        # checkpoint payload.
        val_metrics.pop("conditioning", None)
        val_loss = float(val_metrics["loss"])
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Validation loss became non-finite at epoch {epoch}")
        # CosineAnnealingLR is epoch-driven and takes no metric; passing one
        # would be read as the epoch number and the LR would never decay.
        scheduler.step()

        is_best = val_loss < best_val_loss - config.early_stopping_min_delta
        epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1
        # The stopping rule is evaluated in full every epoch regardless of mode.
        # Under "monitor" its only effect is to record the epoch it first fired,
        # so the early-stopped model stays reportable after the fact.
        # min_epochs gates only this rule; LR scheduling and checkpointing are
        # untouched. Val loss is dominated by disc, so cup Dice can sit near zero
        # for many epochs before soft Dice pulls it out.
        stop_rule_met = (
            epoch >= config.min_epochs
            and epochs_without_improvement >= config.patience
        )
        if stop_rule_met and would_have_stopped_at_epoch is None:
            would_have_stopped_at_epoch = epoch

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_disc_dice": float(val_metrics["disc"]["dice_mean"]),
            "val_cup_dice": float(val_metrics["cup"]["dice_mean"]),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "epochs_without_improvement": float(epochs_without_improvement),
            "would_have_stopped_at_epoch": (
                float(would_have_stopped_at_epoch)
                if would_have_stopped_at_epoch is not None
                else -1.0
            ),
        }
        history.append(row)
        _write_history(history, output_dir / "history.csv")
        if epoch_callback is None:
            print(
                f"epoch={epoch:03d}/{config.epochs} "
                f"lr={row['learning_rate']:.3e} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"disc_dice={row['val_disc_dice']:.4f} "
                f"cup_dice={row['val_cup_dice']:.4f} "
                f"patience={epochs_without_improvement}/{config.patience} "
                f"would_have_stopped_at_epoch={would_have_stopped_at_epoch}",
                flush=True,
            )
        else:
            epoch_callback(row, is_best)

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
                conditioning=checkpoint_conditioning,
            )
        # last_model.pt is rewritten every epoch so the final-epoch weights are
        # recoverable without re-running, whatever the monitor decided.
        _save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            val_metrics,
            config,
            conditioning=checkpoint_conditioning,
        )
        if allow_resume:
            _save_resume_state(
                resume_path,
                fingerprint=fingerprint,
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(),
                generator_state=generator.get_state(),
                history=history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                would_have_stopped_at_epoch=would_have_stopped_at_epoch,
                cup_repairs=cup_repairs,
                elapsed_seconds=(
                    previously_elapsed + (time.perf_counter() - training_started)
                ),
            )

        if config.early_stopping_mode == "terminate" and stop_rule_met:
            print(f"early_stopping best_epoch={best_epoch}", flush=True)
            break

    epochs_run = len(history)
    print(
        f"epoch budget: ran {epochs_run} of {config.epochs} configured epochs "
        f"(early_stopping_mode={config.early_stopping_mode}, "
        f"would_have_stopped_at_epoch={would_have_stopped_at_epoch})",
        flush=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    def _score_test(
        per_image_csv: Path, condition_fn: ConditionFn | None
    ) -> dict[str, Any]:
        metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            threshold=config.threshold,
            max_batches=max_batches,
            per_image_csv=per_image_csv,
            native_hd95=test_native_hd95,
            condition_fn=condition_fn,
        )
        if test_native_hd95:
            _append_rim_one_dl_per_image_context(
                per_image_csv, splits["test"], config.image_size
            )
            if metrics.get("hd95_unit") != HD95_UNIT_NATIVE:
                raise RuntimeError(
                    "RIM-ONE-DL evaluation did not convert HD95 to native pixels"
                )
            metrics["metric_frame"] = _rim_one_dl_metric_frame(config.image_size)
            metrics["per_image_context_fields"] = list(RIM_ONE_DL_PER_IMAGE_CONTEXT)
        return metrics

    test_metrics = _score_test(
        output_dir / "test_per_image_metrics.csv", test_condition
    )
    conditioning_report: dict[str, Any] | None = None
    if conditioned:
        assert vocabulary is not None and selector is not None
        assert test_condition is not None
        conditioning_report = _conditioning_report(
            model=model,
            config=config,
            device=device,
            criterion=criterion,
            vocabulary=vocabulary,
            selector=selector,
            test_condition=test_condition,
            test_metrics=test_metrics,
            val_loader=val_loader,
            score_test=_score_test,
            output_dir=output_dir,
            max_batches=max_batches,
        )
    test_by_name: dict[str, Any] = {}
    for name in sorted(extra_test_sets or {}):
        test_by_name[name] = evaluate_named_test_set(
            model,
            list((extra_test_sets or {})[name]),
            config,
            device,
            criterion,
            output_dir,
            name,
            smoke=smoke,
            condition_fn=test_condition,
        )
    save_training_curves(history, output_dir / "training_curves.png")
    save_prediction_gallery(
        model,
        test_dataset,
        device,
        output_dir / "test_predictions.png",
        threshold=config.threshold,
        count=1 if smoke else 6,
        predict=_gallery_predictor(model, test_condition),
    )

    report = {
        "experiment_name": config.experiment_name,
        "arm": config.arm,
        "smoke_test": smoke,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "epochs_configured": config.epochs,
        "early_stopping": {
            "mode": config.early_stopping_mode,
            "metric": "val_loss",
            "direction": "min",
            "min_delta": config.early_stopping_min_delta,
            "patience": config.patience,
            "min_epochs": config.min_epochs,
            "would_have_stopped_at_epoch": would_have_stopped_at_epoch,
            "terminated_training": epochs_run < config.epochs,
        },
        "lr_schedule": {
            "name": "CosineAnnealingLR",
            "t_max": config.epochs,
            "eta_min": 1e-6,
            "initial_lr": config.learning_rate,
        },
        "checkpoint_selection": "lowest validation BCE + soft Dice loss",
        "training_seconds": (
            previously_elapsed + (time.perf_counter() - training_started)
        ),
        "resumed_from_epoch": resumed_from_epoch,
        "split_counts": split_counts,
        "cup_within_disc_repairs": {
            **cup_repairs,
            "policy": (
                "augmented training samples whose cup leaked outside the disc were "
                "repaired in place with cup &= disc rather than raising"
            ),
        },
        "test": test_metrics,
        "test_by_name": test_by_name,
        "conditioning": conditioning_report,
        "reporting_rule": "Disc and cup metrics are separate; no combined Dice is reported.",
        "metric_frame": test_metrics["metric_frame"],
        "degenerate_case_policy": DEGENERATE_POLICY,
        "artifacts": {
            "test_per_image_metrics": str(output_dir / "test_per_image_metrics.csv"),
            "checkpoint": str(checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "history": str(output_dir / "history.csv"),
            "split_manifest": str(output_dir / "split_manifest.csv"),
            "data_audit": str(output_dir / "data_audit.json"),
            "mask_contact_sheet": str(output_dir / "mask_contact_sheet.png"),
            "training_curves": str(output_dir / "training_curves.png"),
            "test_predictions": str(output_dir / "test_predictions.png"),
            **{
                f"test_{name}_{artifact}": path
                for name, metrics in test_by_name.items()
                for artifact, path in metrics["artifacts"].items()
            },
            **(
                conditioning_report["artifacts"]
                if conditioning_report is not None
                else {}
            ),
        },
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    if allow_resume and resume_path.is_file():
        # The run finished. A stale resume file would let a later invocation
        # "resume" a completed run instead of refusing to overwrite it.
        resume_path.unlink()
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
