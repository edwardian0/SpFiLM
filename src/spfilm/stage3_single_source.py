from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import FundusRecord, validate_splits
from .engine import Stage2Config
from .lodo import Domain, SampleKey, load_lodo_manifest
from .single_source import SingleSourceFold, SingleSourceManifest
from .stage3 import (
    LodoDomainConfig,
    Stage3ConfigError,
    Stage3DataError,
    _domain,
    _fraction,
    _is_int,
    _mapping,
    _non_negative_float,
    _non_negative_int,
    _parse_domain_config,
    _positive_float,
    _positive_int,
    _string,
    build_lodo_manifest,
    resolve_path,
    resolve_project_output,
)


@dataclass(frozen=True)
class Stage3SingleSourceConfig:
    """Configuration for fixed-budget train-on-one, test-on-three runs."""

    experiment_name: str
    output_dir: str
    parent_manifest_dir: str
    fold_manifest_dir: str
    source_domains: tuple[Domain, ...]
    run_seeds: tuple[int, ...]
    train_budget: int
    val_budget: int
    test_budget: int | None
    subsample_seed: int
    domains: tuple[LodoDomainConfig, ...]
    image_size: int
    batch_size: int
    num_workers: int
    epochs: int
    patience: int
    min_epochs: int
    early_stopping_mode: str
    early_stopping_min_delta: float
    learning_rate: float
    weight_decay: float
    base_channels: int
    threshold: float
    horizontal_flip_probability: float
    rotation_degrees: float
    brightness_contrast: float
    requested_device: str

    @classmethod
    def from_json(cls, config_path: str | Path) -> "Stage3SingleSourceConfig":
        config_path = Path(config_path).expanduser().resolve()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage3ConfigError(
                f"Cannot read Stage 3 single-source config {config_path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise Stage3ConfigError(
                "Stage 3 single-source config must be a JSON object"
            )
        if payload.get("stage") != "single_source":
            raise Stage3ConfigError(
                "Stage 3 single-source config must set stage='single_source'"
            )
        if payload.get("arm") != "plain":
            raise Stage3ConfigError(
                "Only the plain Stage 3 single-source arm is currently supported"
            )

        protocol = _mapping(payload.get("protocol"), "protocol")
        if protocol.get("source_test_policy") != "exclude":
            raise Stage3ConfigError(
                "protocol.source_test_policy must be 'exclude'"
            )
        raw_sources = protocol.get("source_domains")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise Stage3ConfigError(
                "protocol.source_domains must be a non-empty array"
            )
        source_domains = tuple(
            _domain(value, "protocol.source_domains") for value in raw_sources
        )
        if len(source_domains) != len(set(source_domains)):
            raise Stage3ConfigError("protocol.source_domains contains duplicates")

        raw_seeds = protocol.get("seeds")
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise Stage3ConfigError("protocol.seeds must be a non-empty array")
        if any(not _is_int(seed) for seed in raw_seeds):
            raise Stage3ConfigError("protocol.seeds must contain only integers")
        run_seeds = tuple(raw_seeds)
        if len(run_seeds) != len(set(run_seeds)):
            raise Stage3ConfigError("protocol.seeds contains duplicates")
        if run_seeds != tuple(sorted(run_seeds)):
            raise Stage3ConfigError("protocol.seeds must be sorted")

        budget = _mapping(protocol.get("budget"), "protocol.budget")
        train_budget = _positive_int(budget, "train")
        val_budget = _positive_int(budget, "val")
        if "test" not in budget:
            raise Stage3ConfigError(
                "protocol.budget.test must be a positive integer or null"
            )
        raw_test_budget = budget.get("test")
        test_budget = (
            None
            if raw_test_budget is None
            else _positive_int(budget, "test")
        )
        subsample_seed = budget.get("subsample_seed")
        if not _is_int(subsample_seed):
            raise Stage3ConfigError(
                "protocol.budget.subsample_seed must be an integer"
            )

        raw_domains = _mapping(payload.get("domains"), "domains")
        domain_configs = tuple(
            sorted(
                (
                    _parse_domain_config(raw_domain, raw_config)
                    for raw_domain, raw_config in raw_domains.items()
                ),
                key=lambda config: config.domain,
            )
        )
        configured_domains = {config.domain for config in domain_configs}
        if configured_domains != set(source_domains):
            raise Stage3ConfigError(
                "Configured domains must exactly match protocol.source_domains"
            )
        if configured_domains != set(Domain):
            raise Stage3ConfigError(
                "Stage 3 single-source requires every domain in the closed "
                "Domain enum"
            )

        config = cls(
            experiment_name=_string(payload, "experiment_name"),
            output_dir=_string(payload, "output_dir"),
            parent_manifest_dir=_string(payload, "parent_manifest_dir"),
            fold_manifest_dir=_string(payload, "fold_manifest_dir"),
            source_domains=source_domains,
            run_seeds=run_seeds,
            train_budget=train_budget,
            val_budget=val_budget,
            test_budget=test_budget,
            subsample_seed=subsample_seed,
            domains=domain_configs,
            image_size=_positive_int(payload, "image_size"),
            batch_size=_positive_int(payload, "batch_size"),
            num_workers=_non_negative_int(payload, "num_workers"),
            epochs=_positive_int(payload, "epochs"),
            patience=_positive_int(payload, "patience"),
            min_epochs=_non_negative_int(payload, "min_epochs"),
            early_stopping_mode=_string(payload, "early_stopping_mode"),
            early_stopping_min_delta=_non_negative_float(
                payload, "early_stopping_min_delta"
            ),
            learning_rate=_positive_float(payload, "learning_rate"),
            weight_decay=_non_negative_float(payload, "weight_decay"),
            base_channels=_positive_int(payload, "base_channels"),
            threshold=_fraction(payload, "threshold", allow_endpoints=True),
            horizontal_flip_probability=_fraction(
                payload,
                "horizontal_flip_probability",
                allow_endpoints=True,
            ),
            rotation_degrees=_non_negative_float(payload, "rotation_degrees"),
            brightness_contrast=_fraction(
                payload,
                "brightness_contrast",
                allow_endpoints=True,
            ),
            requested_device=_string(payload, "requested_device"),
        )
        if config.early_stopping_mode not in {"monitor", "terminate"}:
            raise Stage3ConfigError(
                "early_stopping_mode must be 'monitor' or 'terminate'"
            )
        if config.min_epochs > config.epochs:
            raise Stage3ConfigError("min_epochs cannot exceed epochs")
        if config.requested_device not in {"auto", "cpu", "cuda", "mps"}:
            raise Stage3ConfigError(
                "requested_device must be auto, cpu, cuda, or mps"
            )
        return config

    def domain_config(self, domain: Domain) -> LodoDomainConfig:
        for config in self.domains:
            if config.domain == domain:
                return config
        raise Stage3ConfigError(f"No configuration for domain {domain.value}")

    def training_config(
        self,
        source_domain: Domain,
        run_seed: int,
        output_dir: str,
        requested_device: str | None = None,
    ) -> Stage2Config:
        """Adapt the source domain and shared hyperparameters to the engine."""

        domain_config = self.domain_config(source_domain)
        dataset = {
            Domain.REFUGE_ZEISS: "refuge",
            Domain.REFUGE_CANON_VAL: "refuge",
            Domain.DRISHTI_GS: "drishti",
            Domain.RIM_ONE_DL: "rim_one_dl",
        }[source_domain]
        return Stage2Config(
            experiment_name=(
                f"{self.experiment_name}_{source_domain.value}_seed_{run_seed}"
            ),
            dataset=dataset,
            data_root=domain_config.data_root,
            output_dir=output_dir,
            seed=run_seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            epochs=self.epochs,
            patience=self.patience,
            min_epochs=self.min_epochs,
            early_stopping_mode=self.early_stopping_mode,
            early_stopping_min_delta=self.early_stopping_min_delta,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            base_channels=self.base_channels,
            test_fraction=(
                domain_config.test_fraction
                if domain_config.test_fraction is not None
                else 0.2
            ),
            val_fraction=(
                domain_config.val_fraction
                if domain_config.val_fraction is not None
                else 0.2
            ),
            threshold=self.threshold,
            horizontal_flip_probability=self.horizontal_flip_probability,
            rotation_degrees=self.rotation_degrees,
            brightness_contrast=self.brightness_contrast,
            requested_device=requested_device or self.requested_device,
            rim_manifest=(
                domain_config.split_manifest
                if source_domain == Domain.RIM_ONE_DL
                else None
            ),
        )


def parent_lodo_manifest_path(
    config: Stage3SingleSourceConfig,
    project_root: str | Path,
) -> Path:
    manifest_dir = resolve_project_output(project_root, config.parent_manifest_dir)
    return manifest_dir / "lodo_manifest.json"


def single_source_manifest_path(
    config: Stage3SingleSourceConfig,
    project_root: str | Path,
) -> Path:
    manifest_dir = resolve_project_output(project_root, config.fold_manifest_dir)
    return manifest_dir / "single_source_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_single_source_manifest(
    config: Stage3SingleSourceConfig,
    records_by_domain: dict[Domain, list[FundusRecord]],
    project_root: str | Path,
) -> SingleSourceManifest:
    """Derive fixed-budget partitions from the locked parent membership."""

    project_root = Path(project_root).expanduser().resolve()
    parent_path = parent_lodo_manifest_path(config, project_root)
    if not parent_path.is_file():
        raise Stage3DataError(
            f"Locked parent LODO manifest is missing: {parent_path}"
        )
    committed_parent = load_lodo_manifest(parent_path)
    rebuilt_parent = build_lodo_manifest(config, records_by_domain, project_root)
    if committed_parent != rebuilt_parent:
        raise Stage3DataError(
            "The locked parent LODO manifest does not match the current config "
            "and discovered data"
        )

    strata: dict[SampleKey, str] = {}
    for domain, records in sorted(records_by_domain.items()):
        for record in records:
            key = SampleKey(domain, record.sample_id)
            if key in strata:
                raise Stage3DataError(
                    f"Duplicate discovered sample key while collecting strata: {key!r}"
                )
            strata[key] = record.stratum

    return SingleSourceManifest.build(
        _sha256(parent_path),
        rebuilt_parent.domain_partitions,
        config.train_budget,
        config.val_budget,
        config.test_budget,
        strata,
        config.subsample_seed,
    )


def validate_single_source_manifest_against_config(
    config: Stage3SingleSourceConfig,
    manifest: SingleSourceManifest,
    records_by_domain: dict[Domain, list[FundusRecord]],
    project_root: str | Path,
) -> None:
    """Rebuild the stratified draw and reject any config or data drift."""

    expected = build_single_source_manifest(
        config,
        records_by_domain,
        project_root,
    )
    if manifest != expected:
        raise Stage3DataError(
            "The locked single-source manifest does not match the current "
            "config, parent manifest, and discovered data; inspect the change "
            "and run prepare --force only if new membership is intentional"
        )


def single_source_fold_splits(
    fold: SingleSourceFold,
    records_by_key: dict[SampleKey, FundusRecord],
) -> dict[str, list[FundusRecord]]:
    """Resolve source train/val and each target test set independently."""

    views: dict[str, list[FundusRecord]] = {}
    keyed_views: tuple[tuple[str, tuple[SampleKey, ...]], ...] = (
        ("train", fold.train),
        ("val", fold.val),
        *((domain.value, samples) for domain, samples in fold.tests),
    )
    for name, keys in keyed_views:
        try:
            views[name] = [records_by_key[key] for key in keys]
        except KeyError as error:
            raise Stage3DataError(
                f"Single-source fold references unresolved sample {error.args[0]!r}"
            ) from error

    pooled_test = [
        record
        for domain in fold.target_domains
        for record in views[domain.value]
    ]
    included_records = [*views["train"], *views["val"], *pooled_test]
    validate_splits(
        {
            "train": views["train"],
            "val": views["val"],
            "test": pooled_test,
        },
        included_records,
    )
    return views


def select_single_source_smoke_splits(
    splits: dict[str, list[FundusRecord]],
) -> dict[str, list[FundusRecord]]:
    """Keep one deterministic source and target sample in every partition."""

    if "train" not in splits or "val" not in splits:
        raise Stage3DataError("Smoke selection requires train and val partitions")
    target_names = sorted(set(splits) - {"train", "val"})
    if not target_names:
        raise Stage3DataError(
            "Smoke selection requires at least one named target test partition"
        )
    if any(not splits[name] for name in splits):
        raise Stage3DataError("Smoke selection cannot use an empty partition")

    source_domains = {
        record.domain for name in ("train", "val") for record in splits[name]
    }
    if len(source_domains) != 1:
        raise Stage3DataError(
            "Single-source train and val partitions must share exactly one domain"
        )
    source_domain = next(iter(source_domains))
    for name in target_names:
        try:
            target_domain = Domain(name)
        except ValueError as error:
            raise Stage3DataError(
                f"Unknown named target test partition {name!r}"
            ) from error
        if target_domain.value == source_domain:
            raise Stage3DataError("The source domain cannot be a target test set")
        if {record.domain for record in splits[name]} != {target_domain.value}:
            raise Stage3DataError(
                f"Target partition {name!r} contains another domain"
            )

    all_keys = [
        (record.domain, record.sample_id)
        for name in ("train", "val", *target_names)
        for record in splits[name]
    ]
    if len(all_keys) != len(set(all_keys)):
        raise Stage3DataError(
            "Single-source train, val, and target test partitions must be disjoint"
        )

    return {
        name: [
            min(
                splits[name],
                key=lambda record: (record.domain, record.sample_id),
            )
        ]
        for name in ("train", "val", *target_names)
    }
