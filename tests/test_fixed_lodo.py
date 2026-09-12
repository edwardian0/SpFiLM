"""Contract tests for the fixed-budget leave-one-domain-out arm.

The scientific claim this arm rests on is that it is *paired* with the
train-on-one arm: for each domain, both arms score the same test images, and the
only thing that differs is training volume. These tests pin that pairing, and
the fold shape it depends on.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_stage3_lodo_3_1_fixed import (  # noqa: E402
    CONFIG_DOMAINS_KEY,
    CONFIG_STAGE,
    fixed_lodo_folds,
)
from spfilm.lodo import Domain, DomainPartitions, SampleKey  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    load_single_source_manifest,
)
from spfilm.stage3 import Stage3ConfigError  # noqa: E402
from spfilm.stage3_single_source import Stage3SingleSourceConfig  # noqa: E402


TRAIN_BUDGET, VAL_BUDGET, TEST_BUDGET = 4, 2, 3
REAL_MANIFEST = PROJECT_ROOT / "splits" / "single_source" / "single_source_manifest.json"
REAL_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo_fixed.json"


def _synthetic_manifest() -> SingleSourceManifest:
    partitions = []
    strata = {}
    for domain in sorted(Domain, key=lambda item: item.value):
        prefix = domain.value[:3]
        train = tuple(SampleKey(domain, f"{prefix}_tr{n}") for n in range(TRAIN_BUDGET))
        val = tuple(SampleKey(domain, f"{prefix}_va{n}") for n in range(VAL_BUDGET))
        test = tuple(SampleKey(domain, f"{prefix}_te{n}") for n in range(TEST_BUDGET))
        partitions.append(
            DomainPartitions(domain=domain, train=train, val=val, test=test)
        )
        for key in train + val + test:
            strata[key] = "all"
    return SingleSourceManifest.build(
        "a" * 64, tuple(partitions), TRAIN_BUDGET, VAL_BUDGET, TEST_BUDGET, strata, 42
    )


class FoldShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _synthetic_manifest()
        self.folds = fixed_lodo_folds(self.manifest)

    def test_one_fold_per_domain(self) -> None:
        self.assertEqual(len(self.folds), len(Domain))
        self.assertEqual(
            [fold.held_out_domain for fold in self.folds],
            sorted(Domain, key=lambda item: item.value),
        )

    def test_every_fold_has_the_pooled_budget(self) -> None:
        sources = len(Domain) - 1
        for fold in self.folds:
            self.assertEqual(len(fold.train), sources * TRAIN_BUDGET)
            self.assertEqual(len(fold.val), sources * VAL_BUDGET)
            self.assertEqual(len(fold.test), TEST_BUDGET)

    def test_every_fold_is_the_same_size(self) -> None:
        """The whole point of the budget: no fold trains on more than another."""

        shapes = {
            (len(fold.train), len(fold.val), len(fold.test)) for fold in self.folds
        }
        self.assertEqual(len(shapes), 1)

    def test_held_out_domain_never_appears_in_train_or_val(self) -> None:
        for fold in self.folds:
            for partition in (fold.train, fold.val):
                self.assertFalse(
                    any(
                        sample.domain == fold.held_out_domain
                        for sample in partition
                    )
                )

    def test_test_is_only_the_held_out_domain(self) -> None:
        for fold in self.folds:
            self.assertTrue(
                all(
                    sample.domain == fold.held_out_domain for sample in fold.test
                )
            )

    def test_sources_are_every_other_domain(self) -> None:
        for fold in self.folds:
            sources = {sample.domain for sample in fold.train}
            self.assertEqual(
                sources, set(Domain) - {fold.held_out_domain}
            )


class ActiveDomainTests(unittest.TestCase):
    """Step 4 drops RIM-ONE-DL: folds are composed from the active subset only."""

    ACTIVE = (Domain.DRISHTI_GS, Domain.REFUGE_CANON_VAL, Domain.REFUGE_ZEISS)

    def setUp(self) -> None:
        self.manifest = _synthetic_manifest()
        self.folds = fixed_lodo_folds(self.manifest, self.ACTIVE)

    def test_one_fold_per_active_domain_and_none_for_the_inactive_one(self) -> None:
        self.assertEqual(
            [fold.held_out_domain for fold in self.folds],
            sorted(self.ACTIVE, key=lambda item: item.value),
        )
        self.assertNotIn(Domain.RIM_ONE_DL, {fold.held_out_domain for fold in self.folds})

    def test_folds_train_on_two_and_test_on_one(self) -> None:
        for fold in self.folds:
            sources = {sample.domain for sample in fold.train}
            self.assertEqual(sources, set(self.ACTIVE) - {fold.held_out_domain})
            self.assertEqual(len(sources), 2)
            self.assertEqual(len(fold.train), 2 * TRAIN_BUDGET)
            self.assertEqual(len(fold.val), 2 * VAL_BUDGET)
            self.assertEqual(len(fold.test), TEST_BUDGET)

    def test_inactive_domain_never_appears_anywhere(self) -> None:
        for fold in self.folds:
            for partition in (fold.train, fold.val, fold.test):
                self.assertFalse(any(s.domain == Domain.RIM_ONE_DL for s in partition))

    def test_held_out_test_sets_are_identical_to_the_four_domain_protocol(self) -> None:
        """Dropping a source must not move the 50 images a domain is scored on."""

        four = {fold.held_out_domain: fold.test for fold in fixed_lodo_folds(self.manifest)}
        for fold in self.folds:
            self.assertEqual(fold.test, four[fold.held_out_domain])

    def test_none_means_every_manifest_domain(self) -> None:
        self.assertEqual(fixed_lodo_folds(self.manifest), fixed_lodo_folds(self.manifest, None))
        self.assertEqual(len(fixed_lodo_folds(self.manifest)), len(Domain))

    def test_fewer_than_two_active_domains_is_refused(self) -> None:
        with self.assertRaises(Stage3ConfigError):
            fixed_lodo_folds(self.manifest, (Domain.DRISHTI_GS,))


class PairingTests(unittest.TestCase):
    """The two arms must score identical images, or the comparison is not paired."""

    def test_synthetic_test_sets_match_the_train_on_one_arm(self) -> None:
        manifest = _synthetic_manifest()
        for fold in fixed_lodo_folds(manifest):
            for single in manifest.folds:
                if fold.held_out_domain in single.target_domains:
                    self.assertEqual(
                        set(fold.test),
                        set(single.test_samples(fold.held_out_domain)),
                        f"{fold.held_out_domain.value} differs between arms",
                    )

    @unittest.skipUnless(REAL_MANIFEST.is_file(), "committed manifest not present")
    def test_committed_manifest_gives_120_30_50_and_matching_tests(self) -> None:
        manifest = load_single_source_manifest(REAL_MANIFEST)
        folds = fixed_lodo_folds(manifest)
        self.assertEqual(len(folds), 4)
        for fold in folds:
            self.assertEqual(len(fold.train), 120)
            self.assertEqual(len(fold.val), 30)
            self.assertEqual(len(fold.test), 50)
            single = next(
                s for s in manifest.folds
                if fold.held_out_domain in s.target_domains
            )
            self.assertEqual(
                set(fold.test), set(single.test_samples(fold.held_out_domain))
            )


class ConfigTests(unittest.TestCase):
    @unittest.skipUnless(REAL_CONFIG.is_file(), "fixed-budget config not present")
    def test_real_config_loads_under_the_fixed_budget_stage(self) -> None:
        config = Stage3SingleSourceConfig.from_json(
            REAL_CONFIG, expected_stage=CONFIG_STAGE, domains_key=CONFIG_DOMAINS_KEY
        )
        self.assertEqual(config.experiment_name, "stage3_lodo_fixed_budget_plain_unet")
        self.assertEqual((config.train_budget, config.val_budget, config.test_budget),
                         (40, 10, 50))
        self.assertEqual(len(config.held_out_domains), 4)
        # the alias must not diverge from the field it aliases
        self.assertEqual(config.held_out_domains, config.source_domains)

    @unittest.skipUnless(REAL_CONFIG.is_file(), "fixed-budget config not present")
    def test_wrong_stage_is_refused(self) -> None:
        with self.assertRaises(Stage3ConfigError) as caught:
            Stage3SingleSourceConfig.from_json(
                REAL_CONFIG,
                expected_stage="single_source",
                domains_key=CONFIG_DOMAINS_KEY,
            )
        self.assertIn("stage", str(caught.exception))

    def test_missing_domains_key_is_refused(self) -> None:
        if not REAL_CONFIG.is_file():
            self.skipTest("fixed-budget config not present")
        payload = json.loads(REAL_CONFIG.read_text())
        payload["protocol"].pop(CONFIG_DOMAINS_KEY)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as stream:
            json.dump(payload, stream)
            path = Path(stream.name)
        self.addCleanup(path.unlink)
        with self.assertRaises(Stage3ConfigError) as caught:
            Stage3SingleSourceConfig.from_json(
                path, expected_stage=CONFIG_STAGE, domains_key=CONFIG_DOMAINS_KEY
            )
        self.assertIn(CONFIG_DOMAINS_KEY, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
