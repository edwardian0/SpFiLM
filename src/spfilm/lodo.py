from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
