"""Contract tests for the train-on-all, test-on-each protocol (Step 4, third regime).

What is pinned: the fold pools every active domain's budgeted train/val and
keeps each domain's test partition separate and identical to the LODO arms';
the inactive domain never appears; the FiLM arm must use oracle codes; the two
configs differ only in the arm; and the aggregator pairs FiLM against plain
per domain and reduces the fixed-code sweep to the wrong-code penalty.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aggregate_stage4_all_domains as agg  # noqa: E402
from run_stage3_lodo_3_1_fixed import fixed_lodo_folds  # noqa: E402
from run_stage4_all_domains import (  # noqa: E402
    CONFIG_DOMAINS_KEY,
    CONFIG_STAGE,
    _require_arm_policy,
)
from spfilm.all_domains import (  # noqa: E402
    AllDomainsFoldError,
    compose_all_domains_fold,
    engine_splits,
    select_all_domains_smoke_views,
)
from spfilm.data import FundusRecord  # noqa: E402
from spfilm.lodo import Domain, DomainPartitions, SampleKey  # noqa: E402
from spfilm.single_source import SingleSourceManifest  # noqa: E402
from spfilm.stage3 import Stage3ConfigError  # noqa: E402
from spfilm.stage3_single_source import Stage3SingleSourceConfig  # noqa: E402

ACTIVE = (Domain.DRISHTI_GS, Domain.REFUGE_CANON_VAL, Domain.REFUGE_ZEISS)
TRAIN, VAL, TEST = 4, 2, 3
PLAIN_CONFIG = PROJECT_ROOT / "configs" / "stage4_all_domains_plain_3dom.json"
FILM_CONFIG = PROJECT_ROOT / "configs" / "stage4_all_domains_global_film_3dom.json"
PLAIN_CREATE = PROJECT_ROOT / "configs" / "stage4_all_domains_plain_3dom_create.json"
FILM_CREATE = PROJECT_ROOT / "configs" / "stage4_all_domains_global_film_3dom_create.json"


def _manifest() -> SingleSourceManifest:
    partitions, strata = [], {}
    for domain in sorted(Domain, key=lambda d: d.value):
        prefix = domain.value[:3]
        train = tuple(SampleKey(domain, f"{prefix}_tr{n}") for n in range(TRAIN))
        val = tuple(SampleKey(domain, f"{prefix}_va{n}") for n in range(VAL))
        test = tuple(SampleKey(domain, f"{prefix}_te{n}") for n in range(TEST))
        partitions.append(DomainPartitions(domain=domain, train=train, val=val, test=test))
        for key in train + val + test:
            strata[key] = "all"
    return SingleSourceManifest.build("a" * 64, tuple(partitions), TRAIN, VAL, TEST, strata, 42)


class FoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.fold = compose_all_domains_fold(self.manifest.budgeted_partitions, ACTIVE)

    def test_pools_train_and_val_over_active_domains_only(self) -> None:
        self.assertEqual(len(self.fold.train), len(ACTIVE) * TRAIN)
        self.assertEqual(len(self.fold.val), len(ACTIVE) * VAL)
        self.assertEqual({s.domain for s in self.fold.train}, set(ACTIVE))
        self.assertNotIn(Domain.RIM_ONE_DL, {s.domain for s in self.fold.train + self.fold.val})

    def test_each_active_domain_has_its_own_test_set(self) -> None:
        self.assertEqual(tuple(d for d, _ in self.fold.tests), tuple(sorted(ACTIVE, key=lambda d: d.value)))
        for domain, test in self.fold.tests:
            self.assertEqual(len(test), TEST)
            self.assertTrue(all(s.domain == domain for s in test))

    def test_test_sets_are_the_lodo_test_sets(self) -> None:
        """The same 50 images every other Step 4 arm scores for that domain."""

        lodo = {f.held_out_domain: f.test for f in fixed_lodo_folds(self.manifest, ACTIVE)}
        for domain, test in self.fold.tests:
            self.assertEqual(test, lodo[domain])

    def test_roles_are_disjoint(self) -> None:
        seen = set(self.fold.train) | set(self.fold.val)
        self.assertEqual(len(seen), len(self.fold.train) + len(self.fold.val))
        for _, test in self.fold.tests:
            self.assertTrue(seen.isdisjoint(test))

    def test_needs_two_active_domains_present_in_the_manifest(self) -> None:
        with self.assertRaises(AllDomainsFoldError):
            compose_all_domains_fold(self.manifest.budgeted_partitions, (Domain.DRISHTI_GS,))
        with self.assertRaises(AllDomainsFoldError):
            compose_all_domains_fold(self.manifest.budgeted_partitions[:1], ACTIVE)


def _record(key: SampleKey) -> FundusRecord:
    return FundusRecord(
        sample_id=key.sample_id,
        domain=key.domain.value,
        image_path=Path(f"/img/{key.sample_id}.png"),
        combined_mask_path=Path(f"/img/{key.sample_id}_mask.png"),
        mask_encoding="unused_in_membership_test",
    )


class ViewTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest = _manifest()
        self.fold = compose_all_domains_fold(manifest.budgeted_partitions, ACTIVE)
        self.views = {
            "train": [_record(k) for k in self.fold.train],
            "val": [_record(k) for k in self.fold.val],
            **{d.value: [_record(k) for k in t] for d, t in self.fold.tests},
        }

    def test_engine_splits_pool_the_tests_and_name_each_domain(self) -> None:
        splits, extra = engine_splits(self.views, self.fold.domains)
        self.assertEqual(len(splits["test"]), len(ACTIVE) * TEST)
        self.assertEqual(set(extra), {d.value for d in ACTIVE})
        self.assertEqual({r.sample_id for r in splits["test"]}, {r.sample_id for v in extra.values() for r in v})

    def test_smoke_views_keep_one_per_domain_everywhere(self) -> None:
        smoke = select_all_domains_smoke_views(self.views)
        self.assertEqual(len(smoke["train"]), len(ACTIVE))
        self.assertEqual(len(smoke["val"]), len(ACTIVE))
        for domain in ACTIVE:
            self.assertEqual(len(smoke[domain.value]), 1)


def _load(path: Path) -> Stage3SingleSourceConfig:
    return Stage3SingleSourceConfig.from_json(path, expected_stage=CONFIG_STAGE, domains_key=CONFIG_DOMAINS_KEY)


def _temp(payload: dict) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
        return Path(stream.name)


@unittest.skipUnless(PLAIN_CONFIG.is_file() and FILM_CONFIG.is_file(), "configs not present")
class ConfigTests(unittest.TestCase):
    def test_both_arms_load_with_three_active_domains(self) -> None:
        for path, arm in ((PLAIN_CONFIG, "plain"), (FILM_CONFIG, "global_film"),
                          (PLAIN_CREATE, "plain"), (FILM_CREATE, "global_film")):
            config = _load(path)
            self.assertEqual(config.arm, arm, path.name)
            self.assertEqual(set(config.active_domains), set(ACTIVE))
            self.assertEqual({d.domain for d in config.domains}, set(Domain))
            _require_arm_policy(config)

    def test_film_arm_must_test_with_oracle_codes(self) -> None:
        payload = json.loads(FILM_CONFIG.read_text())
        payload["film"]["test_conditioning"] = "nearest_domain"
        with self.assertRaises(Stage3ConfigError):
            _require_arm_policy(_load(_temp(payload)))

    def test_configs_differ_only_in_the_arm(self) -> None:
        for plain_path, film_path in ((PLAIN_CONFIG, FILM_CONFIG), (PLAIN_CREATE, FILM_CREATE)):
            plain = json.loads(plain_path.read_text())
            film = json.loads(film_path.read_text())
            for key in ("arm", "film", "experiment_name", "output_dir"):
                plain.pop(key, None); film.pop(key, None)
            for key in ("policy", "paired_arm"):
                plain["protocol"].pop(key, None); film["protocol"].pop(key, None)
            self.assertEqual(plain, film, film_path.name)

    def test_shares_the_lodo_hyperparameters(self) -> None:
        lodo = json.loads((PROJECT_ROOT / "configs" / "stage4_plain_3dom.json").read_text())
        this = json.loads(PLAIN_CONFIG.read_text())
        for key in ("image_size", "batch_size", "epochs", "learning_rate", "base_channels",
                    "brightness_contrast", "rotation_degrees", "domains"):
            self.assertEqual(lodo[key], this[key], key)
        self.assertEqual(lodo["protocol"]["budget"], this["protocol"]["budget"])
        self.assertEqual(lodo["protocol"]["seeds"], this["protocol"]["seeds"])
        self.assertEqual(set(lodo["protocol"]["held_out_domains"]), set(this["protocol"]["active_domains"]))


def _write_run(base: Path, arm: str, conditioning_arm: str, seed: int, dice: dict[str, float]) -> Path:
    run = base / f"{arm}_seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    active = [d.value for d in sorted(ACTIVE, key=lambda d: d.value)]
    test_by_domain: dict = {}
    for domain in active:
        rows = [
            {"image_id": f"{domain[:3]}_te{i}", "structure": s, "dice": f"{dice[domain] + i * 0.001:.4f}",
             "iou": "0.6", "hd95": "10", "acc": "0.9", "tp": 1, "fp": 1, "fn": 1, "tn": 1}
            for i in range(TEST) for s in ("disc", "cup")
        ]
        with (run / f"test_{domain}_per_image_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        block: dict = {s: {"dice_mean": dice[domain] + 0.001} for s in ("disc", "cup")}
        block["evaluated_sample_count"] = TEST
        if conditioning_arm == "global_film":
            block["conditioning"] = {"fixed_code_sweep": {
                code: {s: {"dice_mean": dice[domain] + 0.001 - (0.0 if code == domain else 0.05)}
                       for s in ("disc", "cup")}
                for code in active
            }}
        test_by_domain[domain] = block
    (run / "test_metrics.json").write_text(json.dumps({
        "test_by_domain": test_by_domain,
        "all_domains": {
            "protocol": "all_domains_fixed_budget", "arm": arm, "conditioning_arm": conditioning_arm,
            "active_domains": active, "run_seed": seed,
            "budget": {"train": TRAIN, "val": VAL, "test": TEST, "subsample_seed": 42},
            "manifest_sha256": "a" * 64, "completed_at_utc": f"2026-09-13T00:0{seed - 42}:00+00:00",
            "smoke_rehearsal": False, "scientific_result": True,
        },
    }))
    return run


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        base = {"drishti_gs": 0.70, "refuge_canon_val": 0.80, "refuge_zeiss": 0.75}
        for seed in (42, 43):
            _write_run(self.directory, "plain", "plain", seed, base)
            _write_run(self.directory, "film", "global_film", seed, {k: v + 0.03 for k, v in base.items()})

    def test_select_one_run_per_seed_per_arm(self) -> None:
        runs = agg.discover_runs([self.directory])
        plain = agg.select_runs(runs, "plain", (42, 43))
        film = agg.select_runs(runs, "film", (42, 43))
        self.assertEqual([r.run_seed for r in plain], [42, 43])
        self.assertEqual({r.arm for r in film}, {"film"})
        with self.assertRaises(agg.FixedLodoReportError):
            agg.select_runs(runs, "film", (42, 43, 44))
        with self.assertRaises(agg.FixedLodoReportError):
            agg.select_runs(runs, "absent", (42,))

    def test_paired_tests_are_film_minus_plain_per_domain(self) -> None:
        runs = agg.discover_runs([self.directory])
        plain = agg.select_runs(runs, "plain", (42, 43))
        film = agg.select_runs(runs, "film", (42, 43))
        results = agg.paired_tests(agg.build_substrate(plain, film), reference_arm="film")
        self.assertEqual(len(results), len(ACTIVE) * 2)
        for r in results:
            self.assertEqual((r.arm_a, r.arm_b), ("plain", "film"))
            self.assertAlmostEqual(r.mean_difference, 0.03, places=6)

    def test_wrong_code_penalty_is_read_from_the_sweep(self) -> None:
        runs = agg.discover_runs([self.directory])
        film = agg.select_runs(runs, "film", (42, 43))
        cells = agg.build_penalty_cells(film)
        self.assertEqual(len(cells), len(ACTIVE) * 2)
        for c in cells:
            self.assertAlmostEqual(c.own_code_dice - c.worst_other_code_dice, 0.05, places=6)
            self.assertEqual(c.own_code_was_best, 2)

    def test_cells_summarise_each_domain(self) -> None:
        runs = agg.discover_runs([self.directory])
        cells = agg.build_cells(agg.select_runs(runs, "plain", (42, 43)))
        self.assertEqual({(c.domain.value, c.structure) for c in cells},
                         {(d.value, s) for d in ACTIVE for s in ("disc", "cup")})
        self.assertTrue(all(c.test_image_count == TEST for c in cells))


if __name__ == "__main__":
    unittest.main()
