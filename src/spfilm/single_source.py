"""Fixed-budget domain partitions and the train-on-one, test-on-the-rest folds.

Every domain's locked Stage 2 partitions are proportional to its own dataset
size (Drishti 40/10/51, REFUGE 256/64/80, RIM-ONE-DL 340/48/97). That is fine
when each domain is its own experiment, but it makes cross-domain folds
incomparable: pooling three domains gives a training set that swings between 552
and 852 images depending on which domain is dropped, confounded with the domain
identity itself.

This module caps every domain to one common budget first -- Drishti is the floor
at 40 train, 10 val, 50 test -- and composes folds from the capped partitions.
The budgeted partitions are the pivot, deliberately:

* ``compose_all_single_source_folds`` turns them into the train-on-one arm
  (40 train, 10 val, 50 test on each of three unseen targets).
* ``spfilm.lodo.compose_all_lodo_folds`` applied to the *same* budgeted
  partitions turns them into a fixed-budget train-on-three arm (120 train, 30
  val, 50 test) without any further code here.

so the two directions differ only in training volume and stay directly
comparable. The subsample is a pure function of the locked manifest, the budget
and a fixed seed -- never of a run seed -- so the same images are used every
time.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .lodo import Domain, DomainPartitions, SampleKey


SINGLE_SOURCE_MANIFEST_SCHEMA_VERSION = 1
SINGLE_SOURCE_PROTOCOL_NAME = "single_source_locked_multi_target_test"
PARTITION_NAMES = ("train", "val", "test")


class SingleSourceManifestError(ValueError):
    """Raised when a manifest violates its schema or protocol contract."""


def stratified_subsample(
    keys: tuple[SampleKey, ...],
    strata: Mapping[SampleKey, str],
    budget: int,
    seed: int,
    label: str,
) -> tuple[SampleKey, ...]:
    """Draw a fixed-size, proportionally stratified subset deterministically.

    The draw depends only on ``seed``, ``label`` and the sorted sample IDs, so the
    same budget always selects the same images. Strata are allocated by largest
    remainder, which keeps each stratum's share as close to its share of the full
    partition as an integer budget allows.
    """

    if budget <= 0:
        raise SingleSourceManifestError(f"{label} budget must be positive")
    ordered = tuple(sorted(keys))
    if len(ordered) != len(set(ordered)):
        raise SingleSourceManifestError(f"{label} contains duplicate samples")
    if budget > len(ordered):
        raise SingleSourceManifestError(
            f"{label} budget {budget} exceeds the {len(ordered)} available samples"
        )
    if budget == len(ordered):
        return ordered

    grouped: dict[str, list[SampleKey]] = {}
    for key in ordered:
        try:
            stratum = strata[key]
        except KeyError:
            raise SingleSourceManifestError(
                f"{label} sample {key.sample_id!r} has no stratum"
            ) from None
        grouped.setdefault(stratum, []).append(key)

    total = len(ordered)
    exact = {
        stratum: len(group) * budget / total for stratum, group in grouped.items()
    }
    allocation = {stratum: math.floor(value) for stratum, value in exact.items()}
    shortfall = budget - sum(allocation.values())
    # Largest remainder: whichever strata were rounded down hardest get the
    # leftover places, ties broken by name so the result never depends on dict
    # ordering.
    by_remainder = sorted(
        grouped,
        key=lambda stratum: (-(exact[stratum] - allocation[stratum]), stratum),
    )
    for stratum in by_remainder[:shortfall]:
        allocation[stratum] += 1

    selected: list[SampleKey] = []
    for stratum in sorted(grouped):
        group = list(grouped[stratum])
        random.Random(f"{seed}|{label}|{stratum}").shuffle(group)
        selected.extend(group[: allocation[stratum]])
    if len(selected) != budget:
        raise SingleSourceManifestError(
            f"{label} subsample produced {len(selected)} of {budget} samples"
        )
    return tuple(sorted(selected))


def budgeted_domain_partitions(
    domain_partitions: tuple[DomainPartitions, ...],
    train_budget: int,
    val_budget: int,
    test_budget: int | None,
    strata: Mapping[SampleKey, str],
    subsample_seed: int,
) -> tuple[DomainPartitions, ...]:
    """Cap every domain's locked partitions to one budget shared by all domains.

    Each partition is drawn from that domain's own locked partition of the same
    role, so a sample never changes role between the uncapped and the budgeted
    view, nor between the two Stage 3 arms. ``test_budget`` of ``None`` keeps
    each domain's whole locked test partition.
    """

    if not isinstance(domain_partitions, tuple):
        raise TypeError("domain_partitions must be a tuple")
    if not all(
        isinstance(partition, DomainPartitions) for partition in domain_partitions
    ):
        raise TypeError("domain_partitions must contain only DomainPartitions")
    domains = tuple(partition.domain for partition in domain_partitions)
    if len(domains) != len(set(domains)):
        raise ValueError("domain_partitions contains duplicate domains")

    budgets = {"train": train_budget, "val": val_budget, "test": test_budget}
    capped: list[DomainPartitions] = []
    for partition in sorted(
        domain_partitions, key=lambda partition: partition.domain
    ):
        drawn: dict[str, tuple[SampleKey, ...]] = {}
        for name in PARTITION_NAMES:
            samples = getattr(partition, name)
            budget = budgets[name]
            drawn[name] = (
                tuple(sorted(samples))
                if budget is None
                else stratified_subsample(
                    samples,
                    strata,
                    budget,
                    subsample_seed,
                    f"{partition.domain.value}.{name}",
                )
            )
        capped.append(
            DomainPartitions(
                domain=partition.domain,
                train=drawn["train"],
                val=drawn["val"],
                test=drawn["test"],
            )
        )
    return tuple(capped)


@dataclass(frozen=True)
class SingleSourceFold:
    """One training domain, its fixed budget, and every unseen target test set."""

    source_domain: Domain
    train: tuple[SampleKey, ...]
    val: tuple[SampleKey, ...]
    tests: tuple[tuple[Domain, tuple[SampleKey, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_domain, Domain):
            raise TypeError("source_domain must be a valid Domain")
        for partition_name in ("train", "val"):
            partition = getattr(self, partition_name)
            if not isinstance(partition, tuple):
                raise TypeError(f"{partition_name} must be a tuple")
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
            if any(sample.domain != self.source_domain for sample in partition):
                raise ValueError(
                    f"{partition_name} contains samples from another domain."
                )
            if partition != tuple(sorted(partition)):
                raise ValueError(f"{partition_name} must be sorted")
        if not set(self.train).isdisjoint(self.val):
            raise ValueError("Partitions are not pairwise disjoint.")

        if not isinstance(self.tests, tuple) or not self.tests:
            raise ValueError("At least one target test partition is required")
        seen: set[Domain] = set()
        for entry in self.tests:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("Each tests entry must be a (Domain, samples) tuple")
            domain, samples = entry
            if not isinstance(domain, Domain):
                raise TypeError("tests keys must be Domain values")
            if domain == self.source_domain:
                raise ValueError(
                    "tests must not contain the source domain; it is never scored "
                    "as an unseen target."
                )
            if domain in seen:
                raise ValueError(f"tests contains duplicate domain {domain.value}")
            seen.add(domain)
            if not isinstance(samples, tuple) or not samples:
                raise ValueError(f"tests[{domain.value}] partition is empty.")
            if not all(isinstance(sample, SampleKey) for sample in samples):
                raise TypeError(
                    f"tests[{domain.value}] must contain only SampleKey objects."
                )
            if len(samples) != len(set(samples)):
                raise ValueError(
                    f"tests[{domain.value}] contains duplicate SampleKey objects."
                )
            if any(sample.domain != domain for sample in samples):
                raise ValueError(
                    f"tests[{domain.value}] contains samples from another domain."
                )
            if samples != tuple(sorted(samples)):
                raise ValueError(f"tests[{domain.value}] must be sorted")
        if tuple(domain for domain, _ in self.tests) != tuple(sorted(seen)):
            raise ValueError("tests must be sorted by domain")

    @property
    def target_domains(self) -> tuple[Domain, ...]:
        return tuple(domain for domain, _ in self.tests)

    @property
    def test(self) -> tuple[SampleKey, ...]:
        """Every target sample, pooled and sorted, for the union evaluation."""

        return tuple(
            sorted(sample for _, samples in self.tests for sample in samples)
        )

    def test_samples(self, domain: Domain) -> tuple[SampleKey, ...]:
        for candidate, samples in self.tests:
            if candidate == domain:
                return samples
        raise KeyError(f"{domain.value} is not a target of this fold")


def compose_single_source_fold(
    budgeted_partitions: tuple[DomainPartitions, ...],
    source_domain: Domain,
) -> SingleSourceFold:
    """Compose one fold from already-budgeted partitions.

    Training and validation come only from ``source_domain``'s own partitions, so
    no target image can influence either the weights or the checkpoint choice.
    The source domain's test partition is excluded entirely: every sample keeps
    the one role the locked manifest gave it.
    """

    if not isinstance(source_domain, Domain):
        raise TypeError("source_domain must be a valid Domain")
    if not isinstance(budgeted_partitions, tuple):
        raise TypeError("budgeted_partitions must be a tuple")
    if not all(
        isinstance(partition, DomainPartitions)
        for partition in budgeted_partitions
    ):
        raise TypeError("budgeted_partitions must contain only DomainPartitions")
    domains = tuple(partition.domain for partition in budgeted_partitions)
    if len(domains) != len(set(domains)):
        raise ValueError("budgeted_partitions contains duplicate domains")

    source_matches = tuple(
        partition
        for partition in budgeted_partitions
        if partition.domain == source_domain
    )
    if len(source_matches) != 1:
        raise ValueError("Expected exactly one source domain partition")
    targets = tuple(
        sorted(
            (
                partition
                for partition in budgeted_partitions
                if partition.domain != source_domain
            ),
            key=lambda partition: partition.domain,
        )
    )
    if not targets:
        raise ValueError("At least one target domain partition is required")

    source = source_matches[0]
    return SingleSourceFold(
        source_domain=source_domain,
        train=tuple(sorted(source.train)),
        val=tuple(sorted(source.val)),
        tests=tuple(
            (partition.domain, tuple(sorted(partition.test)))
            for partition in targets
        ),
    )


def compose_all_single_source_folds(
    budgeted_partitions: tuple[DomainPartitions, ...],
) -> tuple[SingleSourceFold, ...]:
    """Compose one deterministic fold per domain, sorted by source domain."""

    if not isinstance(budgeted_partitions, tuple):
        raise TypeError("budgeted_partitions must be a tuple")
    if len(budgeted_partitions) < 2:
        raise ValueError("At least two domain partitions are required")
    return tuple(
        compose_single_source_fold(budgeted_partitions, partition.domain)
        for partition in sorted(
            budgeted_partitions, key=lambda partition: partition.domain
        )
    )


@dataclass(frozen=True)
class SingleSourceManifest:
    """Canonical evidence for the fixed budget and every scored target set.

    ``parent_manifest_sha256`` pins the locked LODO manifest these partitions
    were drawn from, so both Stage 3 arms provably share one membership
    authority. The stratified draw itself is revalidated against the config and
    freshly discovered data by ``check``; it is not re-derivable from this file
    alone, because the strata live on the discovered records.
    """

    schema_version: int
    parent_manifest_sha256: str
    train_budget: int
    val_budget: int
    test_budget: int | None
    subsample_seed: int
    budgeted_partitions: tuple[DomainPartitions, ...]
    folds: tuple[SingleSourceFold, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SINGLE_SOURCE_MANIFEST_SCHEMA_VERSION:
            raise SingleSourceManifestError(
                "Unsupported single-source manifest schema version: "
                f"{self.schema_version!r}"
            )
        if (
            not isinstance(self.parent_manifest_sha256, str)
            or len(self.parent_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.parent_manifest_sha256
            )
        ):
            raise SingleSourceManifestError(
                "parent_manifest_sha256 must be a lowercase hex SHA-256 digest"
            )
        for name in ("train_budget", "val_budget"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SingleSourceManifestError(f"{name} must be a positive integer")
        if self.test_budget is not None and (
            not isinstance(self.test_budget, int)
            or isinstance(self.test_budget, bool)
            or self.test_budget <= 0
        ):
            raise SingleSourceManifestError(
                "test_budget must be a positive integer or null"
            )
        if not isinstance(self.subsample_seed, int) or isinstance(
            self.subsample_seed, bool
        ):
            raise SingleSourceManifestError("subsample_seed must be an integer")
        if not isinstance(self.budgeted_partitions, tuple):
            raise SingleSourceManifestError("budgeted_partitions must be a tuple")
        if not isinstance(self.folds, tuple):
            raise SingleSourceManifestError("folds must be a tuple")

        canonical = tuple(
            sorted(self.budgeted_partitions, key=lambda partition: partition.domain)
        )
        if self.budgeted_partitions != canonical:
            raise SingleSourceManifestError(
                "budgeted_partitions must be sorted by domain"
            )

        budgets = {
            "train": self.train_budget,
            "val": self.val_budget,
            "test": self.test_budget,
        }
        for partition in self.budgeted_partitions:
            for name in PARTITION_NAMES:
                samples = getattr(partition, name)
                if samples != tuple(sorted(samples)):
                    raise SingleSourceManifestError(
                        f"{partition.domain.value} {name} must be sorted"
                    )
                budget = budgets[name]
                if budget is not None and len(samples) != budget:
                    raise SingleSourceManifestError(
                        f"{partition.domain.value} {name} holds {len(samples)} "
                        f"samples, not the budgeted {budget}"
                    )

        # Every domain must be budgeted, so that dropping any one of them still
        # leaves a complete set of equal-sized sources for the train-on-three arm.
        expected_folds = compose_all_single_source_folds(self.budgeted_partitions)
        if self.folds != expected_folds:
            raise SingleSourceManifestError(
                "Manifest folds do not match recomposed budgeted partitions"
            )

    @classmethod
    def build(
        cls,
        parent_manifest_sha256: str,
        domain_partitions: tuple[DomainPartitions, ...],
        train_budget: int,
        val_budget: int,
        test_budget: int | None,
        strata: Mapping[SampleKey, str],
        subsample_seed: int,
    ) -> "SingleSourceManifest":
        """Cap the locked partitions to the budget and derive every fold."""

        budgeted = budgeted_domain_partitions(
            domain_partitions,
            train_budget,
            val_budget,
            test_budget,
            strata,
            subsample_seed,
        )
        return cls(
            schema_version=SINGLE_SOURCE_MANIFEST_SCHEMA_VERSION,
            parent_manifest_sha256=parent_manifest_sha256,
            train_budget=train_budget,
            val_budget=val_budget,
            test_budget=test_budget,
            subsample_seed=subsample_seed,
            budgeted_partitions=budgeted,
            folds=compose_all_single_source_folds(budgeted),
        )


def _sample_key_payload(sample: SampleKey) -> dict[str, str]:
    return {"domain": sample.domain.value, "sample_id": sample.sample_id}


def single_source_manifest_payload(
    manifest: SingleSourceManifest,
) -> dict[str, object]:
    """Return the strict JSON-compatible representation of a manifest."""

    if not isinstance(manifest, SingleSourceManifest):
        raise TypeError("manifest must be a SingleSourceManifest")
    return {
        "schema_version": manifest.schema_version,
        "protocol": SINGLE_SOURCE_PROTOCOL_NAME,
        "parent_manifest_sha256": manifest.parent_manifest_sha256,
        "budget": {
            "train": manifest.train_budget,
            "val": manifest.val_budget,
            "test": manifest.test_budget,
            "subsample_seed": manifest.subsample_seed,
        },
        "budgeted_partitions": {
            partition.domain.value: {
                name: [sample.sample_id for sample in getattr(partition, name)]
                for name in PARTITION_NAMES
            }
            for partition in manifest.budgeted_partitions
        },
        "folds": [
            {
                "source_domain": fold.source_domain.value,
                "train": [_sample_key_payload(sample) for sample in fold.train],
                "val": [_sample_key_payload(sample) for sample in fold.val],
                "tests": {
                    domain.value: [
                        _sample_key_payload(sample) for sample in samples
                    ]
                    for domain, samples in fold.tests
                },
            }
            for fold in manifest.folds
        ],
    }


def write_single_source_manifest(
    manifest: SingleSourceManifest,
    output_path: str | Path,
) -> Path:
    """Atomically write a canonical single-source manifest as UTF-8 JSON."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            single_source_manifest_payload(manifest),
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
            raise SingleSourceManifestError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: object,
    expected: set[str],
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SingleSourceManifestError(f"{context} must be a JSON object")
    if set(value) != expected:
        raise SingleSourceManifestError(
            f"{context} keys must be exactly {sorted(expected)}, got {sorted(value)}"
        )
    return value


def _parse_domain(value: object, context: str) -> Domain:
    if not isinstance(value, str):
        raise SingleSourceManifestError(f"{context} must be a domain string")
    try:
        return Domain(value)
    except ValueError as error:
        raise SingleSourceManifestError(
            f"Unknown domain in {context}: {value!r}"
        ) from error


def _parse_sample_keys(value: object, context: str) -> tuple[SampleKey, ...]:
    if not isinstance(value, list):
        raise SingleSourceManifestError(f"{context} must be a JSON array")
    samples: list[SampleKey] = []
    for index, item in enumerate(value):
        row = _require_exact_keys(
            item, {"domain", "sample_id"}, f"{context}[{index}]"
        )
        domain = _parse_domain(row["domain"], f"{context}[{index}].domain")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str):
            raise SingleSourceManifestError(
                f"{context}[{index}].sample_id must be a string"
            )
        try:
            samples.append(SampleKey(domain, sample_id))
        except (TypeError, ValueError) as error:
            raise SingleSourceManifestError(
                f"Invalid sample in {context}: {error}"
            ) from error
    return tuple(samples)


def _parse_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SingleSourceManifestError(f"{context} must be an integer")
    return value


def _parse_optional_int(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _parse_int(value, context)


def load_single_source_manifest(manifest_path: str | Path) -> SingleSourceManifest:
    """Load a manifest and revalidate every budgeted partition and fold."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise SingleSourceManifestError(
            f"Cannot read single-source manifest {manifest_path}: {error}"
        ) from error
    root = _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol",
            "parent_manifest_sha256",
            "budget",
            "budgeted_partitions",
            "folds",
        },
        "manifest",
    )
    if root["protocol"] != SINGLE_SOURCE_PROTOCOL_NAME:
        raise SingleSourceManifestError(
            f"Unsupported single-source protocol: {root['protocol']!r}"
        )
    budget = _require_exact_keys(
        root["budget"], {"train", "val", "test", "subsample_seed"}, "budget"
    )

    partitions_payload = root["budgeted_partitions"]
    if not isinstance(partitions_payload, dict):
        raise SingleSourceManifestError("budgeted_partitions must be a JSON object")
    budgeted_partitions: list[DomainPartitions] = []
    for raw_domain, raw_partitions in partitions_payload.items():
        domain = _parse_domain(raw_domain, "budgeted_partitions key")
        rows = _require_exact_keys(
            raw_partitions,
            set(PARTITION_NAMES),
            f"budgeted_partitions.{domain.value}",
        )
        parsed: dict[str, tuple[SampleKey, ...]] = {}
        for name in PARTITION_NAMES:
            sample_ids = rows[name]
            if not isinstance(sample_ids, list) or not all(
                isinstance(sample_id, str) for sample_id in sample_ids
            ):
                raise SingleSourceManifestError(
                    f"budgeted_partitions.{domain.value}.{name} "
                    "must be an array of sample IDs"
                )
            try:
                parsed[name] = tuple(
                    SampleKey(domain, sample_id) for sample_id in sample_ids
                )
            except (TypeError, ValueError) as error:
                raise SingleSourceManifestError(
                    f"Invalid {domain.value} {name}: {error}"
                ) from error
        try:
            budgeted_partitions.append(
                DomainPartitions(
                    domain=domain,
                    train=parsed["train"],
                    val=parsed["val"],
                    test=parsed["test"],
                )
            )
        except (TypeError, ValueError) as error:
            raise SingleSourceManifestError(
                f"Invalid partitions for {domain.value}: {error}"
            ) from error

    folds_payload = root["folds"]
    if not isinstance(folds_payload, list):
        raise SingleSourceManifestError("folds must be a JSON array")
    folds: list[SingleSourceFold] = []
    for index, raw_fold in enumerate(folds_payload):
        row = _require_exact_keys(
            raw_fold, {"source_domain", "train", "val", "tests"}, f"folds[{index}]"
        )
        source_domain = _parse_domain(
            row["source_domain"], f"folds[{index}].source_domain"
        )
        tests_payload = row["tests"]
        if not isinstance(tests_payload, dict) or not tests_payload:
            raise SingleSourceManifestError(
                f"folds[{index}].tests must be a non-empty JSON object"
            )
        tests: list[tuple[Domain, tuple[SampleKey, ...]]] = []
        for raw_domain, raw_samples in tests_payload.items():
            domain = _parse_domain(raw_domain, f"folds[{index}].tests key")
            tests.append(
                (
                    domain,
                    _parse_sample_keys(
                        raw_samples, f"folds[{index}].tests.{domain.value}"
                    ),
                )
            )
        try:
            folds.append(
                SingleSourceFold(
                    source_domain=source_domain,
                    train=_parse_sample_keys(row["train"], f"folds[{index}].train"),
                    val=_parse_sample_keys(row["val"], f"folds[{index}].val"),
                    tests=tuple(sorted(tests, key=lambda item: item[0])),
                )
            )
        except (TypeError, ValueError) as error:
            raise SingleSourceManifestError(
                f"Invalid fold {index}: {error}"
            ) from error

    try:
        return SingleSourceManifest(
            schema_version=_parse_int(root["schema_version"], "schema_version"),
            parent_manifest_sha256=root["parent_manifest_sha256"],
            train_budget=_parse_int(budget["train"], "budget.train"),
            val_budget=_parse_int(budget["val"], "budget.val"),
            test_budget=_parse_optional_int(budget["test"], "budget.test"),
            subsample_seed=_parse_int(
                budget["subsample_seed"], "budget.subsample_seed"
            ),
            budgeted_partitions=tuple(
                sorted(budgeted_partitions, key=lambda item: item.domain)
            ),
            folds=tuple(sorted(folds, key=lambda item: item.source_domain)),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, SingleSourceManifestError):
            raise
        raise SingleSourceManifestError(
            f"Invalid single-source manifest: {error}"
        ) from error
