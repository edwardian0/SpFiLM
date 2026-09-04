from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import FundusRecord  # noqa: E402
from spfilm.lodo import Domain, DomainPartitions, SampleKey  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceFold,
    SingleSourceManifest,
    SingleSourceManifestError,
    compose_all_single_source_folds,
    load_single_source_manifest,
    stratified_subsample,
    write_single_source_manifest,
)
from spfilm.stage3_single_source import (  # noqa: E402
    Stage3SingleSourceConfig,
    select_single_source_smoke_splits,
    single_source_fold_splits,
    single_source_manifest_path,
)


class StratifiedSubsampleTests(unittest.TestCase):
    def setUp(self) -> None:
        domain = Domain.REFUGE_ZEISS
        self.keys = tuple(
            SampleKey(domain, f"sample-{index:02d}") for index in range(10)
        )
        # Exact shares for a four-sample draw are 2.0, 1.2 and 0.8. Largest
        # remainder therefore assigns 2/1/1, rather than dropping stratum C.
        labels = ("a",) * 5 + ("b",) * 3 + ("c",) * 2
        self.strata = dict(zip(self.keys, labels, strict=True))

    def test_is_deterministic_exact_and_uses_largest_remainder(self) -> None:
        first = stratified_subsample(
            self.keys,
            self.strata,
            budget=4,
            seed=42,
            label="refuge_zeiss.train",
        )
        second = stratified_subsample(
            tuple(reversed(self.keys)),
            self.strata,
            budget=4,
            seed=42,
            label="refuge_zeiss.train",
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(first, tuple(sorted(first)))
        self.assertEqual(
            {
                label: sum(self.strata[key] == label for key in first)
                for label in ("a", "b", "c")
            },
            {"a": 2, "b": 1, "c": 1},
        )

    def test_rejects_budget_larger_than_pool(self) -> None:
        with self.assertRaisesRegex(
            SingleSourceManifestError,
            r"budget 11 exceeds the 10 available samples",
        ):
            stratified_subsample(
                self.keys,
                self.strata,
                budget=11,
                seed=42,
                label="refuge_zeiss.train",
            )


class SingleSourceFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Domain.DRISHTI_GS
        self.target_a = Domain.REFUGE_CANON_VAL
        self.target_b = Domain.REFUGE_ZEISS
        self.train = tuple(
            SampleKey(self.source, f"train-{index}") for index in (1, 2)
        )
        self.val = tuple(
            SampleKey(self.source, f"val-{index}") for index in (1, 2)
        )
        self.target_a_test = tuple(
            SampleKey(self.target_a, f"test-{index}") for index in (1, 2)
        )
        self.target_b_test = tuple(
            SampleKey(self.target_b, f"test-{index}") for index in (1, 2)
        )
        self.tests = tuple(
            sorted(
                (
                    (self.target_a, self.target_a_test),
                    (self.target_b, self.target_b_test),
                ),
                key=lambda item: item[0],
            )
        )

    def _fold(
        self,
        train: tuple[SampleKey, ...] | None = None,
        val: tuple[SampleKey, ...] | None = None,
        tests: tuple[tuple[Domain, tuple[SampleKey, ...]], ...] | None = None,
    ) -> SingleSourceFold:
        return SingleSourceFold(
            source_domain=self.source,
            train=self.train if train is None else train,
            val=self.val if val is None else val,
            tests=self.tests if tests is None else tests,
        )

    def test_valid_fold_exposes_separate_targets_and_pooled_view(self) -> None:
        fold = self._fold()

        self.assertEqual(fold.target_domains, tuple(domain for domain, _ in self.tests))
        self.assertEqual(
            fold.test,
            tuple(sorted(self.target_a_test + self.target_b_test)),
        )
        self.assertEqual(fold.test_samples(self.target_a), self.target_a_test)

    def test_rejects_source_domain_as_target(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must not contain the source"):
            self._fold(
                tests=((self.source, (SampleKey(self.source, "source-test"),)),)
            )

    def test_rejects_unsorted_partitions(self) -> None:
        cases = (
            ("train", tuple(reversed(self.train)), self.val, self.tests),
            ("val", self.train, tuple(reversed(self.val)), self.tests),
            (
                "target test",
                self.train,
                self.val,
                ((self.target_a, tuple(reversed(self.target_a_test))),),
            ),
            (
                "target domains",
                self.train,
                self.val,
                tuple(reversed(self.tests)),
            ),
        )
        for name, train, val, tests in cases:
            with self.subTest(partition=name):
                with self.assertRaisesRegex(ValueError, r"must be sorted"):
                    self._fold(train=train, val=val, tests=tests)

    def test_rejects_duplicates(self) -> None:
        cases = (
            (self.train + (self.train[0],), self.val, self.tests),
            (self.train, self.val + (self.val[0],), self.tests),
            (
                self.train,
                self.val,
                ((self.target_a, self.target_a_test + (self.target_a_test[0],)),),
            ),
        )
        for train, val, tests in cases:
            with self.subTest(train=train, val=val, tests=tests):
                with self.assertRaisesRegex(ValueError, r"duplicate SampleKey"):
                    self._fold(train=train, val=val, tests=tests)

    def test_rejects_cross_domain_train_or_val(self) -> None:
        foreign = SampleKey(self.target_a, "foreign")
        for train, val in (
            ((foreign,), self.val),
            (self.train, (foreign,)),
        ):
            with self.subTest(train=train, val=val):
                with self.assertRaisesRegex(ValueError, r"another domain"):
                    self._fold(train=train, val=val)

    def test_rejects_empty_partitions(self) -> None:
        cases = (
            ((), self.val, self.tests),
            (self.train, (), self.tests),
            (self.train, self.val, ()),
            (self.train, self.val, ((self.target_a, ()),)),
        )
        for train, val, tests in cases:
            with self.subTest(train=train, val=val, tests=tests):
                with self.assertRaisesRegex(ValueError, r"empty|required"):
                    self._fold(train=train, val=val, tests=tests)


class SingleSourceManifestTests(unittest.TestCase):
    @staticmethod
    def _partition(domain: Domain) -> DomainPartitions:
        return DomainPartitions(
            domain=domain,
            train=tuple(
                SampleKey(domain, f"train-{index}") for index in range(4)
            ),
            val=tuple(SampleKey(domain, f"val-{index}") for index in range(2)),
            test=tuple(
                SampleKey(domain, f"test-{index}") for index in range(3)
            ),
        )

    def setUp(self) -> None:
        self.partitions = tuple(self._partition(domain) for domain in Domain)
        self.strata = {
            sample: ("even" if index % 2 == 0 else "odd")
            for partition in self.partitions
            for name in ("train", "val", "test")
            for index, sample in enumerate(getattr(partition, name))
        }
        self.manifest = SingleSourceManifest.build(
            "a" * 64,
            self.partitions,
            train_budget=2,
            val_budget=1,
            test_budget=2,
            strata=self.strata,
            subsample_seed=42,
        )

    def test_manifest_round_trip_preserves_exact_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_single_source_manifest(
                self.manifest,
                Path(directory) / "single_source.json",
            )
            loaded = load_single_source_manifest(path)

        self.assertEqual(loaded, self.manifest)
        self.assertEqual(
            loaded.folds,
            compose_all_single_source_folds(loaded.budgeted_partitions),
        )

    def test_every_fold_preserves_parent_roles_and_excludes_source_test(self) -> None:
        parent_by_domain = {
            partition.domain: partition for partition in self.partitions
        }
        for fold in self.manifest.folds:
            with self.subTest(source_domain=fold.source_domain):
                source = parent_by_domain[fold.source_domain]
                self.assertTrue(set(fold.train) <= set(source.train))
                self.assertTrue(set(fold.val) <= set(source.val))
                self.assertTrue(set(fold.train + fold.val).isdisjoint(source.test))
                for target_domain, test in fold.tests:
                    self.assertTrue(
                        set(test) <= set(parent_by_domain[target_domain].test)
                    )
                    self.assertTrue(
                        set(fold.train + fold.val).isdisjoint(test)
                    )


class SingleSourceStage3Tests(unittest.TestCase):
    @staticmethod
    def _record(key: SampleKey) -> FundusRecord:
        return FundusRecord(
            sample_id=key.sample_id,
            domain=key.domain.value,
            image_path=Path(f"/{key.domain.value}/{key.sample_id}.png"),
            combined_mask_path=Path(
                f"/{key.domain.value}/{key.sample_id}_mask.png"
            ),
            mask_encoding="unused_in_membership_test",
        )

    def setUp(self) -> None:
        self.local_config_path = (
            PROJECT_ROOT / "configs" / "stage3_lodo_single.json"
        )
        self.create_config_path = (
            PROJECT_ROOT / "configs" / "stage3_lodo_single_create.json"
        )

    def test_local_and_create_configs_define_the_fixed_budget_protocol(self) -> None:
        local = Stage3SingleSourceConfig.from_json(self.local_config_path)
        create = Stage3SingleSourceConfig.from_json(self.create_config_path)

        for config in (local, create):
            with self.subTest(config=config):
                self.assertEqual(set(config.source_domains), set(Domain))
                self.assertEqual(config.run_seeds, (42, 43, 44, 45, 46))
                self.assertEqual(
                    (config.train_budget, config.val_budget, config.test_budget),
                    (40, 10, 50),
                )
                self.assertEqual(config.subsample_seed, 42)
                json.dumps(asdict(config))

        self.assertTrue(
            all(
                config.data_root.startswith(
                    "/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets"
                )
                for config in create.domains
            )
        )
        self.assertTrue(
            all(not config.data_root.startswith("/scratch/") for config in local.domains)
        )
        shared_fields = (
            "image_size",
            "batch_size",
            "num_workers",
            "epochs",
            "patience",
            "min_epochs",
            "early_stopping_mode",
            "early_stopping_min_delta",
            "learning_rate",
            "weight_decay",
            "base_channels",
            "threshold",
            "horizontal_flip_probability",
            "rotation_degrees",
            "brightness_contrast",
            "requested_device",
        )
        for field in shared_fields:
            with self.subTest(field=field):
                self.assertEqual(getattr(local, field), getattr(create, field))

    def test_test_budget_can_be_null_but_not_missing(self) -> None:
        payload = json.loads(self.local_config_path.read_text(encoding="utf-8"))
        payload["protocol"]["budget"]["test"] = None
        with tempfile.TemporaryDirectory() as directory:
            nullable_path = Path(directory) / "nullable.json"
            nullable_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(
                Stage3SingleSourceConfig.from_json(nullable_path).test_budget
            )

            del payload["protocol"]["budget"]["test"]
            missing_path = Path(directory) / "missing.json"
            missing_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"budget.test must be a positive integer or null",
            ):
                Stage3SingleSourceConfig.from_json(missing_path)

    def test_training_config_uses_the_source_domain_metadata(self) -> None:
        config = Stage3SingleSourceConfig.from_json(self.local_config_path)
        training = config.training_config(
            Domain.RIM_ONE_DL,
            46,
            "artifacts/stage3_single_source/rim_one_dl/seed_46",
            requested_device="cpu",
        )

        self.assertEqual(training.seed, 46)
        self.assertEqual(training.dataset, "rim_one_dl")
        self.assertEqual(training.data_root, "../../datasets")
        self.assertEqual(training.rim_manifest, "splits/rim_one_dl.json")
        self.assertEqual(training.requested_device, "cpu")

    def test_manifest_path_uses_the_single_source_directory(self) -> None:
        config = Stage3SingleSourceConfig.from_json(self.local_config_path)
        with tempfile.TemporaryDirectory() as directory:
            path = single_source_manifest_path(config, directory)

        self.assertEqual(
            path,
            Path(directory).resolve()
            / "splits"
            / "single_source"
            / "single_source_manifest.json",
        )

    def test_fold_resolution_and_smoke_keep_targets_named_separately(self) -> None:
        source = Domain.DRISHTI_GS
        targets = tuple(sorted(set(Domain) - {source}))
        fold = SingleSourceFold(
            source_domain=source,
            train=tuple(
                SampleKey(source, f"train-{index}") for index in (1, 2)
            ),
            val=tuple(SampleKey(source, f"val-{index}") for index in (1, 2)),
            tests=tuple(
                (
                    domain,
                    tuple(
                        SampleKey(domain, f"test-{index}")
                        for index in (1, 2)
                    ),
                )
                for domain in targets
            ),
        )
        records_by_key = {
            key: self._record(key)
            for key in fold.train + fold.val + fold.test
        }

        splits = single_source_fold_splits(fold, records_by_key)
        smoke = select_single_source_smoke_splits(splits)

        self.assertEqual(set(splits), {"train", "val", *(d.value for d in targets)})
        self.assertEqual(len(splits["train"]), 2)
        self.assertEqual(len(splits["val"]), 2)
        self.assertTrue(all(len(splits[domain.value]) == 2 for domain in targets))
        self.assertTrue(all(len(records) == 1 for records in smoke.values()))


if __name__ == "__main__":
    unittest.main()
