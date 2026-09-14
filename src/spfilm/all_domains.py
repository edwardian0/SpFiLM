"""Train on every active domain at once, test on each domain separately.

This is the third Step 4 protocol, asked for by the supervisor on 2026-09-12:
one model is trained on the pooled budgeted training partitions of every active
domain with the true domain code (codes 0, 1, 2 for three domains), and is
scored on each domain's own budgeted test partition with that domain's own
code. Nothing is held out, so the test-time code is known, and the regime is
the "both" regime of the SpFiLM draft (train on T1w and T1ce together, test on
each). It answers a different question from leave-one-domain-out: not "does the
conditioning transfer to an unseen camera?" but "does it help when the camera
is known?" -- and, through the fixed-code sweep, "does the network use the code
at all?"

The fold is composed from the same locked budgeted partitions as the LODO
protocols, so each domain's 50 test images are the ones every other Step 4 arm
scores. Membership is immutable and every sample keeps one role: a domain's
test partition is only ever tested on, its train partition only trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .data import FundusRecord, validate_splits
from .lodo import Domain, DomainPartitions, SampleKey

ALL_DOMAINS_PROTOCOL_NAME = "all_domains_fixed_budget"


class AllDomainsFoldError(ValueError):
    """The fold cannot be composed or resolved as specified."""


@dataclass(frozen=True)
class AllDomainsFold:
    """Immutable membership: pooled train and val, one test set per domain."""

    domains: tuple[Domain, ...]
    train: tuple[SampleKey, ...]
    val: tuple[SampleKey, ...]
    tests: tuple[tuple[Domain, tuple[SampleKey, ...]], ...]

    def __post_init__(self) -> None:
        if len(self.domains) < 2:
            raise AllDomainsFoldError("At least two domains are required")
        if tuple(sorted(self.domains, key=lambda d: d.value)) != self.domains:
            raise AllDomainsFoldError("domains must be sorted by value")
        if tuple(domain for domain, _ in self.tests) != self.domains:
            raise AllDomainsFoldError("tests must cover exactly the fold's domains, in order")
        for name, partition in (("train", self.train), ("val", self.val)):
            if not partition:
                raise AllDomainsFoldError(f"{name} partition is empty")
            if len(partition) != len(set(partition)):
                raise AllDomainsFoldError(f"{name} contains duplicates")
            present = {sample.domain for sample in partition}
            if present != set(self.domains):
                raise AllDomainsFoldError(
                    f"{name} must contain every fold domain, got "
                    f"{sorted(d.value for d in present)}"
                )
        seen: set[SampleKey] = set(self.train) | set(self.val)
        if len(seen) != len(self.train) + len(self.val):
            raise AllDomainsFoldError("train and val overlap")
        for domain, test in self.tests:
            if not test:
                raise AllDomainsFoldError(f"{domain.value} test partition is empty")
            if any(sample.domain != domain for sample in test):
                raise AllDomainsFoldError(
                    f"{domain.value} test partition holds another domain's samples"
                )
            if not seen.isdisjoint(test):
                raise AllDomainsFoldError(
                    f"{domain.value} test partition overlaps train or val"
                )
            seen |= set(test)

    @property
    def test_by_domain(self) -> dict[Domain, tuple[SampleKey, ...]]:
        return dict(self.tests)


def compose_all_domains_fold(
    budgeted_partitions: Sequence[DomainPartitions],
    active_domains: Sequence[Domain],
) -> AllDomainsFold:
    """Pool the active domains' budgeted train/val; keep each test separate."""

    active = tuple(sorted(set(active_domains), key=lambda d: d.value))
    if len(active) < 2:
        raise AllDomainsFoldError(
            "Training on all domains needs at least two active domains, got "
            f"{[d.value for d in active]}"
        )
    by_domain = {partition.domain: partition for partition in budgeted_partitions}
    missing = [d for d in active if d not in by_domain]
    if missing:
        raise AllDomainsFoldError(
            "Active domains absent from the budgeted manifest: "
            f"{[d.value for d in missing]}"
        )
    return AllDomainsFold(
        domains=active,
        train=tuple(sorted(s for d in active for s in by_domain[d].train)),
        val=tuple(sorted(s for d in active for s in by_domain[d].val)),
        tests=tuple((d, tuple(sorted(by_domain[d].test))) for d in active),
    )


def all_domains_fold_splits(
    fold: AllDomainsFold,
    records_by_key: dict[SampleKey, FundusRecord],
) -> dict[str, list[FundusRecord]]:
    """Resolve keys to records: ``train``, ``val``, and one view per test domain."""

    views: dict[str, list[FundusRecord]] = {}
    keyed: tuple[tuple[str, tuple[SampleKey, ...]], ...] = (
        ("train", fold.train),
        ("val", fold.val),
        *((domain.value, samples) for domain, samples in fold.tests),
    )
    for name, keys in keyed:
        try:
            views[name] = [records_by_key[key] for key in keys]
        except KeyError as error:
            raise AllDomainsFoldError(
                f"{name}: sample {error.args[0]!r} is not among the discovered records"
            ) from None
    return views


def select_all_domains_smoke_views(
    views: dict[str, list[FundusRecord]],
) -> dict[str, list[FundusRecord]]:
    """One deterministic sample per domain in train and val, one per test view."""

    selected: dict[str, list[FundusRecord]] = {}
    for name, records in views.items():
        by_domain: dict[str, list[FundusRecord]] = {}
        for record in records:
            by_domain.setdefault(record.domain, []).append(record)
        selected[name] = [
            sorted(group, key=lambda record: record.sample_id)[0]
            for _, group in sorted(by_domain.items())
        ]
    return selected


def engine_splits(
    views: dict[str, list[FundusRecord]], domains: Sequence[Domain]
) -> tuple[dict[str, list[FundusRecord]], dict[str, list[FundusRecord]]]:
    """Adapt the views to the engine: pooled primary test plus named per-domain tests.

    The engine takes exactly one primary test set, so it receives the pooled
    union; the per-domain views go in as ``extra_test_sets`` and are the
    reportable result. The pooled number is renamed away by the runner.
    """

    pooled = sorted(
        (record for domain in domains for record in views[domain.value]),
        key=lambda record: (record.domain, record.sample_id),
    )
    splits = {"train": list(views["train"]), "val": list(views["val"]), "test": pooled}
    validate_splits(
        splits,
        [record for name in ("train", "val", "test") for record in splits[name]],
    )
    extra = {domain.value: list(views[domain.value]) for domain in domains}
    return splits, extra
