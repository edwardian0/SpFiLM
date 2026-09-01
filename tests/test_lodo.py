from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.lodo import (  # noqa: E402
    Domain,
    DomainPartitions,
    LodoFold,
    LodoManifest,
    LodoManifestError,
    SampleKey,
    compose_all_lodo_folds,
    compose_lodo_fold,
    load_lodo_manifest,
    lodo_manifest_payload,
    write_lodo_manifest,
)


class SampleKeyTests(unittest.TestCase):
    def test_valid_key_preserves_domain_and_sample_id(self) -> None:
        key = SampleKey(Domain.DRISHTI_GS, "sample-001")

        self.assertEqual(key.domain, Domain.DRISHTI_GS)
        self.assertEqual(key.sample_id, "sample-001")

    def test_domain_must_be_a_domain_enum_member(self) -> None:
        with self.assertRaisesRegex(TypeError, r"domain type must be valid"):
            SampleKey("drishti_gs", "sample-001")

    def test_sample_id_must_be_a_string(self) -> None:
        with self.assertRaisesRegex(TypeError, r"sample_id must be string"):
            SampleKey(Domain.DRISHTI_GS, 1)

    def test_sample_id_must_not_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, r"sample_id must not be empty"):
            SampleKey(Domain.DRISHTI_GS, "")

    def test_sample_id_must_not_have_surrounding_whitespace(self) -> None:
        for sample_id in (" sample-001", "sample-001 ", "\tsample-001\n", " "):
            with self.subTest(sample_id=sample_id):
                with self.assertRaisesRegex(
                    ValueError,
                    r"sample_id must not contain surrounding whitespace",
                ):
                    SampleKey(Domain.DRISHTI_GS, sample_id)

    def test_keys_are_hashable_and_sort_by_sample_id_within_a_domain(self) -> None:
        later = SampleKey(Domain.DRISHTI_GS, "sample-002")
        earlier = SampleKey(Domain.DRISHTI_GS, "sample-001")

        self.assertEqual(sorted((later, earlier)), [earlier, later])
        self.assertEqual({earlier, earlier}, {earlier})

    def test_key_is_immutable(self) -> None:
        key = SampleKey(Domain.DRISHTI_GS, "sample-001")

        with self.assertRaises(FrozenInstanceError):
            setattr(key, "sample_id", "changed")


class DomainPartitionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = Domain.DRISHTI_GS
        self.train_key = SampleKey(self.domain, "train-001")
        self.val_key = SampleKey(self.domain, "val-001")
        self.test_key = SampleKey(self.domain, "test-001")

    def test_valid_partitions_preserve_domain_and_membership(self) -> None:
        partitions = DomainPartitions(
            domain=self.domain,
            train=(self.train_key,),
            val=(self.val_key,),
            test=(self.test_key,),
        )

        self.assertEqual(partitions.domain, self.domain)
        self.assertEqual(partitions.train, (self.train_key,))
        self.assertEqual(partitions.val, (self.val_key,))
        self.assertEqual(partitions.test, (self.test_key,))

    def test_duplicate_within_any_partition_is_rejected(self) -> None:
        duplicate_cases = (
            (
                "train",
                (self.train_key, self.train_key),
                (self.val_key,),
                (self.test_key,),
            ),
            (
                "val",
                (self.train_key,),
                (self.val_key, self.val_key),
                (self.test_key,),
            ),
            (
                "test",
                (self.train_key,),
                (self.val_key,),
                (self.test_key, self.test_key),
            ),
        )

        for partition_name, train, val, test in duplicate_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{partition_name} contains duplicate SampleKey objects\.",
                ):
                    DomainPartitions(self.domain, train, val, test)

    def test_partitions_must_be_pairwise_disjoint(self) -> None:
        overlap_cases = (
            ("train/val", (self.train_key,), (self.train_key,), (self.test_key,)),
            ("train/test", (self.train_key,), (self.val_key,), (self.train_key,)),
            ("val/test", (self.train_key,), (self.val_key,), (self.val_key,)),
        )

        for overlap_name, train, val, test in overlap_cases:
            with self.subTest(overlap=overlap_name):
                with self.assertRaisesRegex(
                    ValueError,
                    r"Partitions are not pairwise disjoint\.",
                ):
                    DomainPartitions(self.domain, train, val, test)

    def test_sample_from_another_domain_is_rejected_in_any_partition(self) -> None:
        foreign_key = SampleKey(Domain.REFUGE_ZEISS, "foreign-001")
        foreign_domain_cases = (
            ("train", (foreign_key,), (self.val_key,), (self.test_key,)),
            ("val", (self.train_key,), (foreign_key,), (self.test_key,)),
            ("test", (self.train_key,), (self.val_key,), (foreign_key,)),
        )

        for partition_name, train, val, test in foreign_domain_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{partition_name} contains samples from another domain\.",
                ):
                    DomainPartitions(self.domain, train, val, test)

    def test_empty_partition_is_rejected(self) -> None:
        empty_cases = (
            ("train", (), (self.val_key,), (self.test_key,)),
            ("val", (self.train_key,), (), (self.test_key,)),
            ("test", (self.train_key,), (self.val_key,), ()),
        )

        for partition_name, train, val, test in empty_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{partition_name} partition is empty\.",
                ):
                    DomainPartitions(self.domain, train, val, test)

    def test_each_partition_must_be_a_tuple(self) -> None:
        non_tuple_cases = (
            ("train", [self.train_key], (self.val_key,), (self.test_key,)),
            ("val", (self.train_key,), [self.val_key], (self.test_key,)),
            ("test", (self.train_key,), (self.val_key,), [self.test_key]),
        )

        for partition_name, train, val, test in non_tuple_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(TypeError, r"Partitions must be tuples"):
                    DomainPartitions(self.domain, train, val, test)

    def test_each_partition_must_contain_only_sample_keys(self) -> None:
        invalid_member_cases = (
            ("train", ("train-001",), (self.val_key,), (self.test_key,)),
            ("val", (self.train_key,), ("val-001",), (self.test_key,)),
            ("test", (self.train_key,), (self.val_key,), ("test-001",)),
        )

        for partition_name, train, val, test in invalid_member_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"{partition_name} must contain only SampleKey objects\.",
                ):
                    DomainPartitions(self.domain, train, val, test)

    def test_domain_must_be_a_domain_enum_member(self) -> None:
        with self.assertRaisesRegex(TypeError, r"domain must be a valid domain"):
            DomainPartitions(
                "drishti_gs",
                (self.train_key,),
                (self.val_key,),
                (self.test_key,),
            )

    def test_partitions_are_immutable(self) -> None:
        partitions = DomainPartitions(
            self.domain,
            (self.train_key,),
            (self.val_key,),
            (self.test_key,),
        )

        with self.assertRaises(FrozenInstanceError):
            setattr(partitions, "train", (self.test_key,))


class LodoFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.held_out_domain = Domain.DRISHTI_GS
        self.train = (
            SampleKey(Domain.REFUGE_ZEISS, "zeiss-train-001"),
            SampleKey(Domain.RIM_ONE_DL, "rim-train-001"),
        )
        self.val = (
            SampleKey(Domain.REFUGE_ZEISS, "zeiss-val-001"),
            SampleKey(Domain.RIM_ONE_DL, "rim-val-001"),
        )
        self.test = (SampleKey(self.held_out_domain, "drishti-test-001"),)

    def test_valid_fold_preserves_flattened_membership(self) -> None:
        fold = LodoFold(
            self.held_out_domain,
            self.train,
            self.val,
            self.test,
        )

        self.assertEqual(fold.held_out_domain, self.held_out_domain)
        self.assertEqual(fold.train, self.train)
        self.assertEqual(fold.val, self.val)
        self.assertEqual(fold.test, self.test)

    def test_held_out_domain_must_be_a_domain_enum_member(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"held_out_domain must be a valid Domain",
        ):
            LodoFold("drishti_gs", self.train, self.val, self.test)

    def test_each_partition_must_be_a_tuple(self) -> None:
        non_tuple_cases = (
            ("train", list(self.train), self.val, self.test),
            ("val", self.train, list(self.val), self.test),
            ("test", self.train, self.val, list(self.test)),
        )

        for partition_name, train, val, test in non_tuple_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(TypeError, r"Partitions must be tuples"):
                    LodoFold(self.held_out_domain, train, val, test)

    def test_each_partition_must_be_non_empty(self) -> None:
        empty_cases = (
            ("train", (), self.val, self.test),
            ("val", self.train, (), self.test),
            ("test", self.train, self.val, ()),
        )

        for partition_name, train, val, test in empty_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{partition_name} partition is empty\.",
                ):
                    LodoFold(self.held_out_domain, train, val, test)

    def test_each_partition_must_contain_only_sample_keys(self) -> None:
        invalid_member_cases = (
            ("train", ("train-001",), self.val, self.test),
            ("val", self.train, ("val-001",), self.test),
            ("test", self.train, self.val, ("test-001",)),
        )

        for partition_name, train, val, test in invalid_member_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"{partition_name} must contain only SampleKey objects\.",
                ):
                    LodoFold(self.held_out_domain, train, val, test)

    def test_duplicate_within_any_partition_is_rejected(self) -> None:
        duplicate_cases = (
            ("train", self.train + (self.train[0],), self.val, self.test),
            ("val", self.train, self.val + (self.val[0],), self.test),
            ("test", self.train, self.val, self.test + (self.test[0],)),
        )

        for partition_name, train, val, test in duplicate_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{partition_name} contains duplicate SampleKey objects\.",
                ):
                    LodoFold(self.held_out_domain, train, val, test)

    def test_train_and_val_reject_held_out_domain_samples(self) -> None:
        target_train = SampleKey(self.held_out_domain, "target-train-001")
        target_val = SampleKey(self.held_out_domain, "target-val-001")
        leakage_cases = (
            ("train", self.train + (target_train,), self.val),
            ("val", self.train, self.val + (target_val,)),
        )

        for partition_name, train, val in leakage_cases:
            with self.subTest(partition=partition_name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{partition_name} contains samples from the held-out domain\.",
                ):
                    LodoFold(self.held_out_domain, train, val, self.test)

    def test_test_rejects_samples_from_source_domains(self) -> None:
        source_test = SampleKey(Domain.REFUGE_ZEISS, "source-test-001")

        for test in ((source_test,), self.test + (source_test,)):
            with self.subTest(test=test):
                with self.assertRaisesRegex(
                    ValueError,
                    r"test must contain only samples from the held-out domain\.",
                ):
                    LodoFold(self.held_out_domain, self.train, self.val, test)

    def test_train_and_val_must_be_disjoint(self) -> None:
        overlapping_val = self.val + (self.train[0],)

        with self.assertRaisesRegex(
            ValueError,
            r"Partitions are not pairwise disjoint\.",
        ):
            LodoFold(
                self.held_out_domain,
                self.train,
                overlapping_val,
                self.test,
            )

    def test_fold_is_immutable(self) -> None:
        fold = LodoFold(
            self.held_out_domain,
            self.train,
            self.val,
            self.test,
        )

        with self.assertRaises(FrozenInstanceError):
            setattr(fold, "train", self.val)


class LodoCompositionTests(unittest.TestCase):
    @staticmethod
    def _partitions(domain: Domain, prefix: str) -> DomainPartitions:
        return DomainPartitions(
            domain=domain,
            train=tuple(
                SampleKey(domain, f"{prefix}-train-{index}")
                for index in (4, 2, 1, 3)
            ),
            val=tuple(
                SampleKey(domain, f"{prefix}-val-{index}")
                for index in (2, 1)
            ),
            test=(SampleKey(domain, f"{prefix}-test-1"),),
        )

    def setUp(self) -> None:
        self.source_a = self._partitions(Domain.REFUGE_ZEISS, "zeiss")
        self.source_b = self._partitions(Domain.RIM_ONE_DL, "rim")
        self.target = self._partitions(Domain.DRISHTI_GS, "drishti")
        self.domain_partitions = (
            self.source_b,
            self.target,
            self.source_a,
        )

    def test_composes_source_train_val_and_target_test_only(self) -> None:
        fold = compose_lodo_fold(
            self.domain_partitions,
            self.target.domain,
        )

        self.assertIsInstance(fold, LodoFold)
        self.assertEqual(fold.held_out_domain, self.target.domain)
        self.assertEqual(
            fold.train,
            tuple(sorted(self.source_a.train + self.source_b.train)),
        )
        self.assertEqual(
            fold.val,
            tuple(sorted(self.source_a.val + self.source_b.val)),
        )
        self.assertEqual(fold.test, tuple(sorted(self.target.test)))

        excluded = set(
            self.source_a.test
            + self.source_b.test
            + self.target.train
            + self.target.val
        )
        included = set(fold.train + fold.val + fold.test)
        self.assertTrue(included.isdisjoint(excluded))

    def test_result_is_independent_of_input_domain_order(self) -> None:
        forward = compose_lodo_fold(
            self.domain_partitions,
            self.target.domain,
        )
        reverse = compose_lodo_fold(
            tuple(reversed(self.domain_partitions)),
            self.target.domain,
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.train, tuple(sorted(forward.train)))
        self.assertEqual(forward.val, tuple(sorted(forward.val)))
        self.assertEqual(forward.test, tuple(sorted(forward.test)))

    def test_each_supplied_domain_can_be_held_out(self) -> None:
        for target in self.domain_partitions:
            with self.subTest(held_out_domain=target.domain):
                fold = compose_lodo_fold(
                    self.domain_partitions,
                    target.domain,
                )
                self.assertEqual(fold.held_out_domain, target.domain)
                self.assertEqual(fold.test, tuple(sorted(target.test)))

    def test_held_out_domain_must_be_a_domain_enum_member(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"held_out_domain must be a valid Domain",
        ):
            compose_lodo_fold(self.domain_partitions, "drishti_gs")

    def test_domain_partitions_must_be_a_tuple(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"domain_partitions must be a tuple",
        ):
            compose_lodo_fold(list(self.domain_partitions), self.target.domain)

    def test_domain_partitions_must_contain_only_partition_objects(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"domain_partitions must contain only DomainPartitions",
        ):
            compose_lodo_fold(
                self.domain_partitions + ("not-a-partition",),
                self.target.domain,
            )

    def test_duplicate_domain_partitions_are_rejected(self) -> None:
        duplicate = self._partitions(self.source_a.domain, "duplicate-zeiss")

        with self.assertRaisesRegex(
            ValueError,
            r"domain_partitions contains duplicate domains",
        ):
            compose_lodo_fold(
                self.domain_partitions + (duplicate,),
                self.target.domain,
            )

    def test_missing_held_out_domain_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Expected exactly one held-out domain partition",
        ):
            compose_lodo_fold(
                (self.source_a, self.source_b),
                self.target.domain,
            )

    def test_fold_requires_at_least_one_source_domain(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"At least one source domain partition is required",
        ):
            compose_lodo_fold((self.target,), self.target.domain)

    def test_all_fold_composer_returns_one_sorted_fold_per_domain(self) -> None:
        folds = compose_all_lodo_folds(self.domain_partitions)
        expected_domains = tuple(
            sorted(partition.domain for partition in self.domain_partitions)
        )

        self.assertEqual(
            tuple(fold.held_out_domain for fold in folds),
            expected_domains,
        )
        self.assertEqual(
            folds,
            tuple(
                compose_lodo_fold(self.domain_partitions, domain)
                for domain in expected_domains
            ),
        )

    def test_all_fold_result_is_independent_of_input_order(self) -> None:
        forward = compose_all_lodo_folds(self.domain_partitions)
        reverse = compose_all_lodo_folds(
            tuple(reversed(self.domain_partitions))
        )

        self.assertEqual(forward, reverse)

    def test_all_fold_composer_requires_a_tuple(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"domain_partitions must be a tuple",
        ):
            compose_all_lodo_folds(list(self.domain_partitions))

    def test_all_fold_composer_rejects_invalid_members(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"domain_partitions must contain only DomainPartitions",
        ):
            compose_all_lodo_folds(
                self.domain_partitions + ("not-a-partition",)
            )

    def test_all_fold_composer_requires_at_least_two_domains(self) -> None:
        for domain_partitions in ((), (self.target,)):
            with self.subTest(domain_partitions=domain_partitions):
                with self.assertRaisesRegex(
                    ValueError,
                    r"At least two domain partitions are required",
                ):
                    compose_all_lodo_folds(domain_partitions)

    def test_all_fold_composer_rejects_duplicate_domains(self) -> None:
        duplicate = self._partitions(self.source_a.domain, "duplicate-zeiss")

        with self.assertRaisesRegex(
            ValueError,
            r"domain_partitions contains duplicate domains",
        ):
            compose_all_lodo_folds(self.domain_partitions + (duplicate,))


class LodoManifestTests(unittest.TestCase):
    @staticmethod
    def _partitions(domain: Domain, prefix: str) -> DomainPartitions:
        return DomainPartitions(
            domain=domain,
            train=(
                SampleKey(domain, f"{prefix}-train-2"),
                SampleKey(domain, f"{prefix}-train-1"),
            ),
            val=(SampleKey(domain, f"{prefix}-val-1"),),
            test=(SampleKey(domain, f"{prefix}-test-1"),),
        )

    def setUp(self) -> None:
        self.partitions = (
            self._partitions(Domain.RIM_ONE_DL, "rim"),
            self._partitions(Domain.DRISHTI_GS, "drishti"),
            self._partitions(Domain.REFUGE_ZEISS, "zeiss"),
        )
        self.split_seeds = {
            Domain.REFUGE_ZEISS: 42,
            Domain.DRISHTI_GS: 42,
            Domain.RIM_ONE_DL: None,
        }

    def test_manifest_round_trip_preserves_exact_protocol(self) -> None:
        manifest = LodoManifest.build(self.partitions, self.split_seeds)

        with tempfile.TemporaryDirectory() as directory:
            path = write_lodo_manifest(
                manifest,
                Path(directory) / "lodo.json",
            )
            loaded = load_lodo_manifest(path)

        self.assertEqual(loaded, manifest)
        self.assertEqual(
            loaded.folds,
            compose_all_lodo_folds(loaded.domain_partitions),
        )

    def test_manifest_serialization_is_canonical(self) -> None:
        forward = LodoManifest.build(self.partitions, self.split_seeds)
        reverse = LodoManifest.build(
            tuple(reversed(self.partitions)),
            dict(reversed(tuple(self.split_seeds.items()))),
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(
            json.dumps(lodo_manifest_payload(forward), sort_keys=True),
            json.dumps(lodo_manifest_payload(reverse), sort_keys=True),
        )

    def test_manifest_rejects_fold_tampering(self) -> None:
        manifest = LodoManifest.build(self.partitions, self.split_seeds)
        payload = lodo_manifest_payload(manifest)
        payload["folds"][0]["test"][0]["sample_id"] = "tampered-test"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                LodoManifestError,
                r"folds do not match recomposed domain partitions",
            ):
                load_lodo_manifest(path)

    def test_manifest_requires_one_split_seed_entry_per_domain(self) -> None:
        incomplete_seeds = dict(self.split_seeds)
        del incomplete_seeds[Domain.DRISHTI_GS]

        with self.assertRaisesRegex(
            LodoManifestError,
            r"split_seeds domains must exactly match domain_partitions",
        ):
            LodoManifest.build(self.partitions, incomplete_seeds)

    def test_manifest_rejects_unknown_root_fields(self) -> None:
        manifest = LodoManifest.build(self.partitions, self.split_seeds)
        payload = lodo_manifest_payload(manifest)
        payload["unexpected"] = True

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unexpected.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                LodoManifestError,
                r"manifest keys must be exactly",
            ):
                load_lodo_manifest(path)


if __name__ == "__main__":
    unittest.main()
