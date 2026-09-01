from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import (
    FundusRecord,
    audit_records,
    discover_drishti,
    discover_refuge_training,
    discover_refuge_validation,
    discover_rim_one_dl,
    load_rim_one_dl_split_manifest,
    provider_partition,
    stratified_partition,
    validate_splits,
)
from .engine import Stage2Config
from .lodo import (
    Domain,
    DomainPartitions,
    LodoFold,
    LodoManifest,
    SampleKey,
)


class Stage3ConfigError(ValueError):
    """Raised when the Stage 3 config cannot define the locked protocol."""


class Stage3DataError(ValueError):
    """Raised when discovered data disagrees with the locked manifest."""


@dataclass(frozen=True)
class LodoDomainConfig:
    domain: Domain
    adapter: str
    data_root: str
    split_policy: str
    split_seed: int | None = None
    test_fraction: float | None = None
    val_fraction: float | None = None
    split_manifest: str | None = None
    image_subdir: str | None = None
    mask_subdir: str | None = None


@dataclass(frozen=True)
class Stage3LodoConfig:
    experiment_name: str
    output_dir: str
    fold_manifest_dir: str
    held_out_domains: tuple[Domain, ...]
    run_seeds: tuple[int, ...]
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
    def from_json(cls, config_path: str | Path) -> "Stage3LodoConfig":
        config_path = Path(config_path).expanduser().resolve()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Stage3ConfigError(
                f"Cannot read Stage 3 config {config_path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise Stage3ConfigError("Stage 3 config must be a JSON object")
        if payload.get("stage") != "lodo":
            raise Stage3ConfigError("Stage 3 config must set stage='lodo'")
        if payload.get("arm") != "plain":
            raise Stage3ConfigError("Only the plain Stage 3 arm is currently supported")

        protocol = _mapping(payload.get("protocol"), "protocol")
        if protocol.get("source_test_policy") != "exclude":
            raise Stage3ConfigError(
                "protocol.source_test_policy must be 'exclude'"
            )
        raw_held_out = protocol.get("held_out_domains")
        if not isinstance(raw_held_out, list) or not raw_held_out:
            raise Stage3ConfigError(
                "protocol.held_out_domains must be a non-empty array"
            )
        held_out_domains = tuple(
            _domain(value, "protocol.held_out_domains")
            for value in raw_held_out
        )
        if len(held_out_domains) != len(set(held_out_domains)):
            raise Stage3ConfigError("protocol.held_out_domains contains duplicates")

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

        raw_domains = _mapping(payload.get("domains"), "domains")
        domain_configs = tuple(
            _parse_domain_config(raw_domain, raw_config)
            for raw_domain, raw_config in raw_domains.items()
        )
        domain_configs = tuple(
            sorted(domain_configs, key=lambda config: config.domain)
        )
        configured_domains = {config.domain for config in domain_configs}
        if configured_domains != set(held_out_domains):
            raise Stage3ConfigError(
                "Configured domains must exactly match protocol.held_out_domains"
            )
        if configured_domains != set(Domain):
            raise Stage3ConfigError(
                "Stage 3 LODO requires every domain in the closed Domain enum"
            )

        config = cls(
            experiment_name=_string(payload, "experiment_name"),
            output_dir=_string(payload, "output_dir"),
            fold_manifest_dir=_string(payload, "fold_manifest_dir"),
            held_out_domains=held_out_domains,
            run_seeds=run_seeds,
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
        held_out_domain: Domain,
        run_seed: int,
        output_dir: str,
        requested_device: str | None = None,
    ) -> Stage2Config:
        """Adapt shared hyperparameters to the existing split-driven engine."""

        domain_config = self.domain_config(held_out_domain)
        dataset = {
            Domain.REFUGE_ZEISS: "refuge",
            Domain.REFUGE_CANON_VAL: "refuge",
            Domain.DRISHTI_GS: "drishti",
            Domain.RIM_ONE_DL: "rim_one_dl",
        }[held_out_domain]
        return Stage2Config(
            experiment_name=(
                f"{self.experiment_name}_{held_out_domain.value}_seed_{run_seed}"
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
                if held_out_domain == Domain.RIM_ONE_DL
                else None
            ),
        )


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Stage3ConfigError(f"{context} must be a JSON object")
    return value


def _domain(value: object, context: str) -> Domain:
    if not isinstance(value, str):
        raise Stage3ConfigError(f"{context} must contain domain strings")
    try:
        return Domain(value)
    except ValueError as error:
        raise Stage3ConfigError(f"Unknown domain {value!r} in {context}") from error


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Stage3ConfigError(f"{key} must be a non-empty string")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not _is_int(value) or value <= 0:
        raise Stage3ConfigError(f"{key} must be a positive integer")
    return value


def _non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not _is_int(value) or value < 0:
        raise Stage3ConfigError(f"{key} must be a non-negative integer")
    return value


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Stage3ConfigError(f"{key} must be numeric")
    return float(value)


def _positive_float(payload: dict[str, Any], key: str) -> float:
    value = _number(payload, key)
    if value <= 0:
        raise Stage3ConfigError(f"{key} must be positive")
    return value


def _non_negative_float(payload: dict[str, Any], key: str) -> float:
    value = _number(payload, key)
    if value < 0:
        raise Stage3ConfigError(f"{key} must be non-negative")
    return value


def _fraction(
    payload: dict[str, Any],
    key: str,
    allow_endpoints: bool = False,
) -> float:
    value = _number(payload, key)
    valid = 0 <= value <= 1 if allow_endpoints else 0 < value < 1
    if not valid:
        qualifier = (
            "between zero and one inclusive"
            if allow_endpoints
            else "between zero and one"
        )
        raise Stage3ConfigError(f"{key} must be {qualifier}")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise Stage3ConfigError(f"{key} must be a non-empty string or null")
    return value


def _optional_seed(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not _is_int(value):
        raise Stage3ConfigError(f"{key} must be an integer or null")
    return value


def _optional_fraction(payload: dict[str, Any], key: str) -> float | None:
    if payload.get(key) is None:
        return None
    return _fraction(payload, key)


def _parse_domain_config(
    raw_domain: str,
    raw_config: object,
) -> LodoDomainConfig:
    domain = _domain(raw_domain, "domains")
    payload = _mapping(raw_config, f"domains.{domain.value}")
    if payload.get("blocked_on"):
        raise Stage3ConfigError(
            f"{domain.value} remains blocked: {payload['blocked_on']}"
        )
    config = LodoDomainConfig(
        domain=domain,
        adapter=_string(payload, "adapter"),
        data_root=_string(payload, "data_root"),
        split_policy=_string(payload, "split_policy"),
        split_seed=_optional_seed(payload, "split_seed"),
        test_fraction=_optional_fraction(payload, "test_fraction"),
        val_fraction=_optional_fraction(payload, "val_fraction"),
        split_manifest=_optional_string(payload, "split_manifest"),
        image_subdir=_optional_string(payload, "image_subdir"),
        mask_subdir=_optional_string(payload, "mask_subdir"),
    )
    expected_adapters = {
        Domain.REFUGE_ZEISS: "refuge",
        Domain.REFUGE_CANON_VAL: "refuge_canon_val",
        Domain.DRISHTI_GS: "drishti",
        Domain.RIM_ONE_DL: "rim_one_dl",
    }
    if config.adapter != expected_adapters[domain]:
        raise Stage3ConfigError(
            f"{domain.value} adapter must be {expected_adapters[domain]!r}"
        )
    if config.split_policy == "stratified":
        if (
            config.split_seed is None
            or config.test_fraction is None
            or config.val_fraction is None
        ):
            raise Stage3ConfigError(
                f"{domain.value} stratified policy requires split_seed and fractions"
            )
    elif config.split_policy == "provider":
        if config.split_seed is None or config.val_fraction is None:
            raise Stage3ConfigError(
                f"{domain.value} provider policy requires split_seed and val_fraction"
            )
    elif config.split_policy == "committed_manifest":
        if config.split_manifest is None:
            raise Stage3ConfigError(
                f"{domain.value} committed policy requires split_manifest"
            )
    else:
        raise Stage3ConfigError(
            f"Unsupported split policy for {domain.value}: {config.split_policy!r}"
        )
    return config


def resolve_path(project_root: str | Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root).expanduser().resolve() / path
    return path.resolve()


def resolve_project_output(project_root: str | Path, value: str) -> Path:
    project_root = Path(project_root).expanduser().resolve()
    path = resolve_path(project_root, value)
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise Stage3ConfigError(
            f"Generated output must remain inside {project_root}, got {path}"
        ) from error
    return path


def lodo_manifest_path(
    config: Stage3LodoConfig,
    project_root: str | Path,
) -> Path:
    manifest_dir = resolve_project_output(project_root, config.fold_manifest_dir)
    return manifest_dir / "lodo_manifest.json"


def discover_lodo_records(
    config: Stage3LodoConfig,
    project_root: str | Path,
) -> dict[Domain, list[FundusRecord]]:
    """Discover every configured domain without creating any split membership."""

    project_root = Path(project_root).expanduser().resolve()
    discovered: dict[Domain, list[FundusRecord]] = {}
    for domain_config in config.domains:
        root = resolve_path(project_root, domain_config.data_root)
        if domain_config.adapter == "refuge":
            records = discover_refuge_training(root)
        elif domain_config.adapter == "refuge_canon_val":
            records = discover_refuge_validation(
                root,
                image_subdir=domain_config.image_subdir,
                mask_subdir=domain_config.mask_subdir,
            )
        elif domain_config.adapter == "drishti":
            records = discover_drishti(root)
        elif domain_config.adapter == "rim_one_dl":
            records = discover_rim_one_dl(root)
        else:
            raise Stage3ConfigError(
                f"Unsupported adapter {domain_config.adapter!r}"
            )
        if not records:
            raise Stage3DataError(f"{domain_config.domain.value} discovery is empty")
        if any(record.domain != domain_config.domain.value for record in records):
            raise Stage3DataError(
                f"{domain_config.domain.value} adapter returned another domain"
            )
        sample_ids = [record.sample_id for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise Stage3DataError(
                f"{domain_config.domain.value} contains duplicate sample IDs"
            )
        discovered[domain_config.domain] = sorted(
            records, key=lambda record: record.sample_id
        )
    return discovered


def build_lodo_manifest(
    config: Stage3LodoConfig,
    records_by_domain: dict[Domain, list[FundusRecord]],
    project_root: str | Path,
) -> LodoManifest:
    """Create locked per-domain partitions using only configured split policies."""

    if set(records_by_domain) != {item.domain for item in config.domains}:
        raise Stage3DataError(
            "Discovered record domains do not match the Stage 3 config"
        )
    project_root = Path(project_root).expanduser().resolve()
    domain_partitions: list[DomainPartitions] = []
    split_seeds: dict[Domain, int | None] = {}
    for domain_config in config.domains:
        records = records_by_domain[domain_config.domain]
        if domain_config.split_policy == "stratified":
            if (
                domain_config.split_seed is None
                or domain_config.test_fraction is None
                or domain_config.val_fraction is None
            ):
                raise Stage3ConfigError(
                    f"Incomplete stratified policy for {domain_config.domain.value}"
                )
            splits = stratified_partition(
                records,
                seed=domain_config.split_seed,
                test_fraction=domain_config.test_fraction,
                val_fraction_of_remaining=domain_config.val_fraction,
            )
        elif domain_config.split_policy == "provider":
            if (
                domain_config.split_seed is None
                or domain_config.val_fraction is None
            ):
                raise Stage3ConfigError(
                    f"Incomplete provider policy for {domain_config.domain.value}"
                )
            splits = provider_partition(
                records,
                seed=domain_config.split_seed,
                val_fraction=domain_config.val_fraction,
            )
        elif domain_config.split_policy == "committed_manifest":
            if domain_config.split_manifest is None:
                raise Stage3ConfigError(
                    f"Missing split manifest for {domain_config.domain.value}"
                )
            splits = load_rim_one_dl_split_manifest(
                records,
                resolve_path(project_root, domain_config.split_manifest),
            )
        else:
            raise Stage3ConfigError(
                f"Unsupported split policy {domain_config.split_policy!r}"
            )
        validate_splits(splits, records)
        domain_partitions.append(
            DomainPartitions(
                domain=domain_config.domain,
                train=tuple(
                    SampleKey(domain_config.domain, record.sample_id)
                    for record in splits["train"]
                ),
                val=tuple(
                    SampleKey(domain_config.domain, record.sample_id)
                    for record in splits["val"]
                ),
                test=tuple(
                    SampleKey(domain_config.domain, record.sample_id)
                    for record in splits["test"]
                ),
            )
        )
        split_seeds[domain_config.domain] = domain_config.split_seed
    return LodoManifest.build(tuple(domain_partitions), split_seeds)


def resolve_manifest_records(
    manifest: LodoManifest,
    records_by_domain: dict[Domain, list[FundusRecord]],
) -> dict[SampleKey, FundusRecord]:
    """Resolve locked sample keys and prove exact coverage of current discovery."""

    manifest_domains = {
        partition.domain for partition in manifest.domain_partitions
    }
    if set(records_by_domain) != manifest_domains:
        raise Stage3DataError(
            "Manifest domains do not exactly match discovered domains"
        )
    resolved: dict[SampleKey, FundusRecord] = {}
    for partition in manifest.domain_partitions:
        records = records_by_domain[partition.domain]
        discovered = {
            SampleKey(partition.domain, record.sample_id): record
            for record in records
        }
        if len(discovered) != len(records):
            raise Stage3DataError(
                f"{partition.domain.value} discovery contains duplicate sample IDs"
            )
        listed = set(partition.train + partition.val + partition.test)
        if listed != set(discovered):
            missing = sorted(set(discovered) - listed)
            unknown = sorted(listed - set(discovered))
            raise Stage3DataError(
                f"{partition.domain.value} manifest/discovery mismatch: "
                f"unlisted={missing[:5]}, unknown={unknown[:5]}"
            )
        resolved.update(discovered)
    return resolved


def validate_manifest_against_config(
    config: Stage3LodoConfig,
    manifest: LodoManifest,
    records_by_domain: dict[Domain, list[FundusRecord]],
    project_root: str | Path,
) -> None:
    """Reject a manifest produced from different split settings or source data."""

    expected = build_lodo_manifest(config, records_by_domain, project_root)
    if manifest != expected:
        raise Stage3DataError(
            "The locked LODO manifest does not match the current config and "
            "discovered data; inspect the change and run prepare --force only if "
            "new membership is intentional"
        )


def fold_record_splits(
    fold: LodoFold,
    records_by_key: dict[SampleKey, FundusRecord],
) -> dict[str, list[FundusRecord]]:
    """Resolve one immutable fold into the record lists consumed by the engine."""

    splits: dict[str, list[FundusRecord]] = {}
    for partition_name in ("train", "val", "test"):
        keys = getattr(fold, partition_name)
        try:
            splits[partition_name] = [records_by_key[key] for key in keys]
        except KeyError as error:
            raise Stage3DataError(
                f"Fold references unresolved sample {error.args[0]!r}"
            ) from error
    records = [
        record
        for partition_name in ("train", "val", "test")
        for record in splits[partition_name]
    ]
    validate_splits(splits, records)
    return splits


def select_lodo_smoke_splits(
    splits: dict[str, list[FundusRecord]],
) -> dict[str, list[FundusRecord]]:
    """Keep one deterministic sample per represented domain in each partition."""

    if set(splits) != {"train", "val", "test"}:
        raise Stage3DataError("Smoke selection requires train, val, and test splits")
    selected: dict[str, list[FundusRecord]] = {}
    for partition_name in ("train", "val", "test"):
        by_domain: dict[str, list[FundusRecord]] = {}
        for record in splits[partition_name]:
            by_domain.setdefault(record.domain, []).append(record)
        selected[partition_name] = [
            sorted(records, key=lambda record: record.sample_id)[0]
            for _, records in sorted(by_domain.items())
        ]
    selected_records = [
        record
        for partition_name in ("train", "val", "test")
        for record in selected[partition_name]
    ]
    validate_splits(selected, selected_records)
    return selected


def audit_lodo_domains(
    records_by_domain: dict[Domain, list[FundusRecord]],
) -> dict[str, object]:
    """Run the expensive source-layout/mask audit once per discovered domain."""

    return {
        domain.value: audit_records(records)
        for domain, records in sorted(records_by_domain.items())
    }
