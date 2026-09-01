from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


LODO_MANIFEST_SCHEMA_VERSION = 1
LODO_PROTOCOL_NAME = "leave_one_domain_out_locked_test"


class LodoManifestError(ValueError):
    """Raised when a LODO manifest violates its schema or protocol contract."""


class Domain(str, Enum):
    """Acquisition-domain keys permitted in Stage 3 folds and manifests."""

    REFUGE_ZEISS = "refuge_zeiss"
    REFUGE_CANON_VAL = "refuge_canon_val"
    DRISHTI_GS = "drishti_gs"
    RIM_ONE_DL = "rim_one_dl"


@dataclass(frozen=True, order=True)
class SampleKey:
    """Stable, portable identity of one sample within an acquisition domain."""

    domain: Domain
    sample_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Domain):
            raise TypeError("domain type must be valid")
        if not isinstance(self.sample_id, str):
            raise TypeError("sample_id must be string")
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.sample_id != self.sample_id.strip():
            raise ValueError("sample_id must not contain surrounding whitespace")


@dataclass(frozen=True)
class DomainPartitions:
    """Contract for the partitions for each domain."""

    domain: Domain
    train: tuple[SampleKey, ...]
    val: tuple[SampleKey, ...]
    test: tuple[SampleKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Domain):
            raise TypeError("domain must be a valid domain")

        if not all(
            isinstance(partition, tuple)
            for partition in (self.train, self.val, self.test)
        ):
            raise TypeError("Partitions must be tuples")

        partitions = (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        )

        for partition_name, partition in partitions:
            if not all(isinstance(sample, SampleKey) for sample in partition):
                raise TypeError(
                    f"{partition_name} must contain only SampleKey objects."
                )

            if any(sample.domain != self.domain for sample in partition):
                raise ValueError(f"{partition_name} contains samples from another domain.")

            if not partition:
                raise ValueError(f"{partition_name} partition is empty.")

            if len(partition) != len(set(partition)):
                raise ValueError(
                    f"{partition_name} contains duplicate SampleKey objects."
                )

        if (
            not set(self.train).isdisjoint(self.val)
            or not set(self.train).isdisjoint(self.test)
            or not set(self.val).isdisjoint(self.test)
        ):
            raise ValueError("Partitions are not pairwise disjoint.")


@dataclass(frozen=True)
class LodoFold:
    """Immutable flattened membership for one leave-one-domain-out fold."""

    held_out_domain: Domain
    train: tuple[SampleKey, ...]
    val: tuple[SampleKey, ...]
    test: tuple[SampleKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.held_out_domain, Domain):
            raise TypeError("held_out_domain must be a valid Domain")
        if not all(
            isinstance(partition, tuple)
            for partition in (self.train, self.val, self.test)
        ):
            raise TypeError("Partitions must be tuples")

        partitions = (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        )

        for partition_name, partition in partitions:
            if not partition:
                raise ValueError(f"{partition_name} partition is empty.")

            if not all(isinstance(sample, SampleKey) for sample in partition):
                raise TypeError(
                    f"{partition_name} must contain only SampleKey objects."
                )

            if len(partition) != len(set(partition)):
                raise ValueError(
                    f"{partition_name} contains duplicate SampleKey objects."
                )

        for partition_name, partition in (
            ("train", self.train),
            ("val", self.val),
        ):
            if any(sample.domain == self.held_out_domain for sample in partition):
                raise ValueError(
                    f"{partition_name} contains samples from the held-out domain."
                )

        if any(sample.domain != self.held_out_domain for sample in self.test):
            raise ValueError(
                "test must contain only samples from the held-out domain."
            )

        if (
            not set(self.train).isdisjoint(self.val)
            or not set(self.train).isdisjoint(self.test)
            or not set(self.val).isdisjoint(self.test)
        ):
            raise ValueError("Partitions are not pairwise disjoint.")


def compose_lodo_fold(
    domain_partitions: tuple[DomainPartitions, ...],
    held_out_domain: Domain,
) -> LodoFold:
    """Compose one deterministic LODO fold from locked domain partitions."""

    if not isinstance(held_out_domain, Domain):
        raise TypeError("held_out_domain must be a valid Domain")
    if not isinstance(domain_partitions, tuple):
        raise TypeError("domain_partitions must be a tuple")
    if not all(
        isinstance(partition, DomainPartitions)
        for partition in domain_partitions
    ):
        raise TypeError("domain_partitions must contain only DomainPartitions")

    domains = tuple(partition.domain for partition in domain_partitions)
    if len(domains) != len(set(domains)):
        raise ValueError("domain_partitions contains duplicate domains")

    held_out_matches = tuple(
        partition
        for partition in domain_partitions
        if partition.domain == held_out_domain
    )
    if len(held_out_matches) != 1:
        raise ValueError("Expected exactly one held-out domain partition")

    source_partitions = tuple(
        partition
        for partition in domain_partitions
        if partition.domain != held_out_domain
    )
    if not source_partitions:
        raise ValueError("At least one source domain partition is required")

    target_partition = held_out_matches[0]
    train = tuple(
        sorted(
            sample
            for partition in source_partitions
            for sample in partition.train
        )
    )
    val = tuple(
        sorted(
            sample
            for partition in source_partitions
            for sample in partition.val
        )
    )
    test = tuple(sorted(target_partition.test))

    return LodoFold(
        held_out_domain=held_out_domain,
        train=train,
        val=val,
        test=test,
    )


def compose_all_lodo_folds(
    domain_partitions: tuple[DomainPartitions, ...],
) -> tuple[LodoFold, ...]:
    """Compose one deterministic LODO fold per supplied domain."""

    if not isinstance(domain_partitions, tuple):
        raise TypeError("domain_partitions must be a tuple")
    if not all(
        isinstance(partition, DomainPartitions)
        for partition in domain_partitions
    ):
        raise TypeError("domain_partitions must contain only DomainPartitions")
    if len(domain_partitions) < 2:
        raise ValueError("At least two domain partitions are required")

    folds: list[LodoFold] = []

    for partition in sorted(
        domain_partitions,
        key=lambda item: item.domain,
    ):
        held_out_domain = partition.domain
        fold = compose_lodo_fold(domain_partitions, held_out_domain)
        folds.append(fold)

    return tuple(folds)


@dataclass(frozen=True)
class LodoManifest:
    """Canonical serializable evidence for locked domain partitions and folds."""

    schema_version: int
    split_seeds: tuple[tuple[Domain, int | None], ...]
    domain_partitions: tuple[DomainPartitions, ...]
    folds: tuple[LodoFold, ...]

    def __post_init__(self) -> None:
        if self.schema_version != LODO_MANIFEST_SCHEMA_VERSION:
            raise LodoManifestError(
                "Unsupported LODO manifest schema version: "
                f"{self.schema_version!r}"
            )
        if not isinstance(self.split_seeds, tuple):
            raise LodoManifestError("split_seeds must be a tuple")
        if not isinstance(self.domain_partitions, tuple):
            raise LodoManifestError("domain_partitions must be a tuple")
        if not isinstance(self.folds, tuple):
            raise LodoManifestError("folds must be a tuple")

        canonical_partitions = tuple(
            sorted(self.domain_partitions, key=lambda partition: partition.domain)
        )
        if self.domain_partitions != canonical_partitions:
            raise LodoManifestError("domain_partitions must be sorted by domain")
        for partition in self.domain_partitions:
            for partition_name in ("train", "val", "test"):
                samples = getattr(partition, partition_name)
                if samples != tuple(sorted(samples)):
                    raise LodoManifestError(
                        f"{partition.domain.value} {partition_name} must be sorted"
                    )

        parsed_seeds: list[tuple[Domain, int | None]] = []
        for item in self.split_seeds:
            if not isinstance(item, tuple) or len(item) != 2:
                raise LodoManifestError(
                    "Each split_seeds entry must be a (Domain, seed) tuple"
                )
            domain, seed = item
            if not isinstance(domain, Domain):
                raise LodoManifestError("split_seeds keys must be Domain values")
            if seed is not None and (
                not isinstance(seed, int) or isinstance(seed, bool)
            ):
                raise LodoManifestError("Split seeds must be integers or null")
            parsed_seeds.append((domain, seed))
        if tuple(parsed_seeds) != tuple(
            sorted(parsed_seeds, key=lambda item: item[0])
        ):
            raise LodoManifestError("split_seeds must be sorted by domain")

        partition_domains = tuple(
            partition.domain for partition in self.domain_partitions
        )
        if tuple(domain for domain, _ in parsed_seeds) != partition_domains:
            raise LodoManifestError(
                "split_seeds domains must exactly match domain_partitions"
            )

        expected_folds = compose_all_lodo_folds(self.domain_partitions)
        if self.folds != expected_folds:
            raise LodoManifestError(
                "Manifest folds do not match recomposed domain partitions"
            )

    @classmethod
    def build(
        cls,
        domain_partitions: tuple[DomainPartitions, ...],
        split_seeds: Mapping[Domain, int | None],
    ) -> "LodoManifest":
        """Canonicalize locked partitions and derive their complete fold set."""

        if not isinstance(domain_partitions, tuple):
            raise TypeError("domain_partitions must be a tuple")
        if not isinstance(split_seeds, Mapping):
            raise TypeError("split_seeds must be a mapping")
        canonical_partitions = tuple(
            DomainPartitions(
                domain=partition.domain,
                train=tuple(sorted(partition.train)),
                val=tuple(sorted(partition.val)),
                test=tuple(sorted(partition.test)),
            )
            for partition in sorted(
                domain_partitions,
                key=lambda partition: partition.domain,
            )
        )
        canonical_seeds = tuple(
            sorted(split_seeds.items(), key=lambda item: item[0])
        )
        return cls(
            schema_version=LODO_MANIFEST_SCHEMA_VERSION,
            split_seeds=canonical_seeds,
            domain_partitions=canonical_partitions,
            folds=compose_all_lodo_folds(canonical_partitions),
        )


def _sample_key_payload(sample: SampleKey) -> dict[str, str]:
    return {"domain": sample.domain.value, "sample_id": sample.sample_id}


def lodo_manifest_payload(manifest: LodoManifest) -> dict[str, object]:
    """Return the strict JSON-compatible representation of a manifest."""

    if not isinstance(manifest, LodoManifest):
        raise TypeError("manifest must be a LodoManifest")
    return {
        "schema_version": manifest.schema_version,
        "protocol": LODO_PROTOCOL_NAME,
        "split_seeds": {
            domain.value: seed for domain, seed in manifest.split_seeds
        },
        "domain_partitions": {
            partition.domain.value: {
                partition_name: [
                    sample.sample_id
                    for sample in getattr(partition, partition_name)
                ]
                for partition_name in ("train", "val", "test")
            }
            for partition in manifest.domain_partitions
        },
        "folds": [
            {
                "held_out_domain": fold.held_out_domain.value,
                **{
                    partition_name: [
                        _sample_key_payload(sample)
                        for sample in getattr(fold, partition_name)
                    ]
                    for partition_name in ("train", "val", "test")
                },
            }
            for fold in manifest.folds
        ],
    }


def write_lodo_manifest(
    manifest: LodoManifest,
    output_path: str | Path,
) -> Path:
    """Atomically write a canonical LODO manifest as UTF-8 JSON."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            lodo_manifest_payload(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LodoManifestError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: object,
    expected: set[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LodoManifestError(f"{context} must be a JSON object")
    if set(value) != expected:
        raise LodoManifestError(
            f"{context} keys must be exactly {sorted(expected)}, "
            f"got {sorted(value)}"
        )
    return value


def _parse_domain(value: object, context: str) -> Domain:
    if not isinstance(value, str):
        raise LodoManifestError(f"{context} must be a domain string")
    try:
        return Domain(value)
    except ValueError as error:
        raise LodoManifestError(f"Unknown domain in {context}: {value!r}") from error


def _parse_sample_keys(value: object, context: str) -> tuple[SampleKey, ...]:
    if not isinstance(value, list):
        raise LodoManifestError(f"{context} must be a JSON array")
    samples: list[SampleKey] = []
    for index, item in enumerate(value):
        row = _require_exact_keys(
            item,
            {"domain", "sample_id"},
            f"{context}[{index}]",
        )
        domain = _parse_domain(row["domain"], f"{context}[{index}].domain")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str):
            raise LodoManifestError(
                f"{context}[{index}].sample_id must be a string"
            )
        try:
            samples.append(SampleKey(domain, sample_id))
        except (TypeError, ValueError) as error:
            raise LodoManifestError(f"Invalid sample in {context}: {error}") from error
    return tuple(samples)


def load_lodo_manifest(manifest_path: str | Path) -> LodoManifest:
    """Load a manifest and revalidate every stored partition and derived fold."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise LodoManifestError(
            f"Cannot read LODO manifest {manifest_path}: {error}"
        ) from error
    root = _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol",
            "split_seeds",
            "domain_partitions",
            "folds",
        },
        "manifest",
    )
    if root["protocol"] != LODO_PROTOCOL_NAME:
        raise LodoManifestError(
            f"Unsupported LODO protocol: {root['protocol']!r}"
        )
    schema_version = root["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise LodoManifestError("schema_version must be an integer")

    seed_payload = root["split_seeds"]
    if not isinstance(seed_payload, dict):
        raise LodoManifestError("split_seeds must be a JSON object")
    split_seeds: list[tuple[Domain, int | None]] = []
    for raw_domain, seed in seed_payload.items():
        domain = _parse_domain(raw_domain, "split_seeds key")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise LodoManifestError(
                f"Split seed for {domain.value} must be an integer or null"
            )
        split_seeds.append((domain, seed))

    partitions_payload = root["domain_partitions"]
    if not isinstance(partitions_payload, dict):
        raise LodoManifestError("domain_partitions must be a JSON object")
    domain_partitions: list[DomainPartitions] = []
    for raw_domain, raw_partitions in partitions_payload.items():
        domain = _parse_domain(raw_domain, "domain_partitions key")
        partition_rows = _require_exact_keys(
            raw_partitions,
            {"train", "val", "test"},
            f"domain_partitions.{domain.value}",
        )
        parsed: dict[str, tuple[SampleKey, ...]] = {}
        for partition_name in ("train", "val", "test"):
            sample_ids = partition_rows[partition_name]
            if not isinstance(sample_ids, list) or not all(
                isinstance(sample_id, str) for sample_id in sample_ids
            ):
                raise LodoManifestError(
                    f"domain_partitions.{domain.value}.{partition_name} "
                    "must be an array of sample IDs"
                )
            try:
                parsed[partition_name] = tuple(
                    SampleKey(domain, sample_id) for sample_id in sample_ids
                )
            except (TypeError, ValueError) as error:
                raise LodoManifestError(
                    f"Invalid {domain.value} {partition_name}: {error}"
                ) from error
        try:
            domain_partitions.append(
                DomainPartitions(
                    domain=domain,
                    train=parsed["train"],
                    val=parsed["val"],
                    test=parsed["test"],
                )
            )
        except (TypeError, ValueError) as error:
            raise LodoManifestError(
                f"Invalid partitions for {domain.value}: {error}"
            ) from error

    folds_payload = root["folds"]
    if not isinstance(folds_payload, list):
        raise LodoManifestError("folds must be a JSON array")
    folds: list[LodoFold] = []
    for index, raw_fold in enumerate(folds_payload):
        fold_row = _require_exact_keys(
            raw_fold,
            {"held_out_domain", "train", "val", "test"},
            f"folds[{index}]",
        )
        held_out_domain = _parse_domain(
            fold_row["held_out_domain"],
            f"folds[{index}].held_out_domain",
        )
        try:
            folds.append(
                LodoFold(
                    held_out_domain=held_out_domain,
                    train=_parse_sample_keys(
                        fold_row["train"], f"folds[{index}].train"
                    ),
                    val=_parse_sample_keys(
                        fold_row["val"], f"folds[{index}].val"
                    ),
                    test=_parse_sample_keys(
                        fold_row["test"], f"folds[{index}].test"
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise LodoManifestError(f"Invalid fold {index}: {error}") from error

    try:
        return LodoManifest(
            schema_version=schema_version,
            split_seeds=tuple(
                sorted(split_seeds, key=lambda item: item[0])
            ),
            domain_partitions=tuple(
                sorted(domain_partitions, key=lambda item: item.domain)
            ),
            folds=tuple(folds),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, LodoManifestError):
            raise
        raise LodoManifestError(f"Invalid LODO manifest: {error}") from error
