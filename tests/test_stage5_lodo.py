"""Contract tests for the Step 5 leave-one-domain-out runner and its aggregator.

What is pinned: the Step 5 folds are the fixed-budget LODO folds over the three
active domains (train on two, test on the held-out third, on the same test
images every other arm scores); the two configs differ only in the arm, and
from the Step 4 LODO configs only in their names; a conditioned arm decides its
code from the held-out domain's unlabelled reference sample and cannot use the
oracle code; the runner writes its own metadata block, which the Step 5
aggregator reads and the Stage 3 / Step 4 tools ignore; and what the runner
writes is what the aggregator reads.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aggregate_stage4_film as step4_aggregator  # noqa: E402
import aggregate_stage5_lodo as step5_aggregator  # noqa: E402
import run_stage3_lodo_3_1_fixed as fixed_runner  # noqa: E402
import run_stage5_lodo as runner  # noqa: E402
from aggregate_stage3_fixed import discover_fixed_runs  # noqa: E402
from spfilm.all_domains import compose_all_domains_fold  # noqa: E402
from spfilm.lodo import Domain, DomainPartitions, SampleKey  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    load_single_source_manifest,
    write_single_source_manifest,
)
from spfilm.stage3 import Stage3ConfigError  # noqa: E402
from spfilm.stage3_single_source import Stage3SingleSourceConfig  # noqa: E402


ACTIVE = (Domain.DRISHTI_GS, Domain.REFUGE_CANON_VAL, Domain.REFUGE_ZEISS)
TRAIN, VAL, TEST = 4, 2, 3
CONFIGS = PROJECT_ROOT / "configs"
PLAIN_CONFIG = CONFIGS / "stage5_lodo_plain_3dom.json"
FILM_CONFIG = CONFIGS / "stage5_lodo_global_film_3dom.json"
PLAIN_CREATE = CONFIGS / "stage5_lodo_plain_3dom_create.json"
FILM_CREATE = CONFIGS / "stage5_lodo_global_film_3dom_create.json"
STEP4_TWINS = {
    PLAIN_CONFIG: CONFIGS / "stage4_plain_3dom.json",
    FILM_CONFIG: CONFIGS / "stage4_global_film_3dom.json",
    PLAIN_CREATE: CONFIGS / "stage4_plain_3dom_create.json",
    FILM_CREATE: CONFIGS / "stage4_global_film_3dom_create.json",
}
CONFIGS_PRESENT = all(path.is_file() for path in (*STEP4_TWINS, *STEP4_TWINS.values()))
REAL_MANIFEST = PROJECT_ROOT / "splits" / "single_source" / "single_source_manifest.json"
DATASETS_PRESENT = (PROJECT_ROOT.parents[1] / "datasets" / "REFUGE").is_dir()
PLAIN_ARM = step5_aggregator.DEFAULT_PLAIN_ARM
FILM_ARM = step5_aggregator.DEFAULT_FILM_ARM
STEP4_PLAIN_ARM = step4_aggregator.DEFAULT_PLAIN_ARM
STEP4_FILM_ARM = step4_aggregator.DEFAULT_FILM_ARM


def _load(path: Path) -> Stage3SingleSourceConfig:
    return Stage3SingleSourceConfig.from_json(
        path, expected_stage=runner.CONFIG_STAGE, domains_key=runner.CONFIG_DOMAINS_KEY
    )


def _temp_config(payload: dict) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
        return Path(stream.name)


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


def _run_main(module, argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = module.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


# --------------------------------------------------------------------------
# Folds and the reference sample
# --------------------------------------------------------------------------


@unittest.skipUnless(CONFIGS_PRESENT, "Step 4/5 LODO configs not present")
class FoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.config = _load(PLAIN_CONFIG)
        self.folds = runner.stage5_lodo_folds(self.manifest, self.config)

    def test_three_folds_train_on_two_and_test_on_the_third(self) -> None:
        self.assertEqual(
            [fold.held_out_domain for fold in self.folds],
            sorted(ACTIVE, key=lambda d: d.value),
        )
        for fold in self.folds:
            sources = {sample.domain for sample in fold.train}
            self.assertEqual(sources, set(ACTIVE) - {fold.held_out_domain})
            self.assertEqual({sample.domain for sample in fold.val}, sources)
            self.assertEqual({sample.domain for sample in fold.test}, {fold.held_out_domain})
            self.assertEqual((len(fold.train), len(fold.val), len(fold.test)),
                             (2 * TRAIN, 2 * VAL, TEST))

    def test_the_inactive_domain_never_enters_a_fold(self) -> None:
        for fold in self.folds:
            for partition in (fold.train, fold.val, fold.test):
                self.assertNotIn(Domain.RIM_ONE_DL, {sample.domain for sample in partition})

    def test_test_sets_are_the_ones_every_other_arm_scores(self) -> None:
        """Step 4 LODO, train-on-one and train-on-all score these same images."""

        step4 = {f.held_out_domain: f.test for f in fixed_runner.fixed_lodo_folds(self.manifest, ACTIVE)}
        train_on_all = compose_all_domains_fold(self.manifest.budgeted_partitions, ACTIVE).test_by_domain
        for fold in self.folds:
            self.assertEqual(fold.test, step4[fold.held_out_domain])
            self.assertEqual(fold.test, train_on_all[fold.held_out_domain])
            for single in self.manifest.folds:
                if fold.held_out_domain in single.target_domains:
                    self.assertEqual(fold.test, single.test_samples(fold.held_out_domain))

    def test_an_inactive_held_out_domain_is_refused(self) -> None:
        with self.assertRaises(Stage3ConfigError):
            runner.stage5_lodo_fold(self.manifest, self.config, Domain.RIM_ONE_DL)
        fold = runner.stage5_lodo_fold(self.manifest, self.config, Domain.DRISHTI_GS)
        self.assertEqual(fold.held_out_domain, Domain.DRISHTI_GS)

    @unittest.skipUnless(REAL_MANIFEST.is_file(), "committed manifest not present")
    def test_committed_manifest_gives_80_20_50(self) -> None:
        manifest = load_single_source_manifest(REAL_MANIFEST)
        folds = runner.stage5_lodo_folds(manifest, self.config)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual((len(fold.train), len(fold.val), len(fold.test)), (80, 20, 50))


@unittest.skipUnless(CONFIGS_PRESENT, "Step 4/5 LODO configs not present")
class ReferenceSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.plain = _load(PLAIN_CONFIG)
        self.film = _load(FILM_CONFIG)

    def test_plain_arm_reads_no_held_out_image(self) -> None:
        for domain in ACTIVE:
            self.assertEqual(runner.conditioning_reference_keys(self.plain, self.manifest, domain), ())

    def test_per_domain_rule_reads_the_held_out_train_partition(self) -> None:
        for fold in runner.stage5_lodo_folds(self.manifest, self.film):
            keys = runner.conditioning_reference_keys(self.film, self.manifest, fold.held_out_domain)
            partition = next(p for p in self.manifest.budgeted_partitions if p.domain == fold.held_out_domain)
            self.assertEqual(keys, partition.train)
            for used in (fold.train, fold.val, fold.test):
                self.assertTrue(set(keys).isdisjoint(used))

    def test_per_image_ablation_reads_no_held_out_image(self) -> None:
        ablation = replace(self.film, test_conditioning="nearest_image")
        for domain in ACTIVE:
            self.assertEqual(runner.conditioning_reference_keys(ablation, self.manifest, domain), ())


# --------------------------------------------------------------------------
# Configs and refusals (no data needed)
# --------------------------------------------------------------------------


@unittest.skipUnless(CONFIGS_PRESENT, "Step 4/5 LODO configs not present")
class ConfigTests(unittest.TestCase):
    ARM_ONLY_KEYS = ("arm", "film", "experiment_name", "output_dir")
    PROTOCOL_ARM_ONLY_KEYS = ("policy", "paired_arm")
    STEP5_IDENTITY_KEYS = ("experiment_name", "stage", "output_dir")
    STEP5_PROTOCOL_IDENTITY_KEYS = (
        "policy", "paired_arm", "inactive_domains_note", "pairing_rationale", "relation_to_step4",
    )

    def test_both_arms_load_under_the_step5_stage(self) -> None:
        for path, arm, name, partner in (
            (PLAIN_CONFIG, "plain", PLAIN_ARM, FILM_ARM),
            (PLAIN_CREATE, "plain", PLAIN_ARM, FILM_ARM),
            (FILM_CONFIG, "global_film", FILM_ARM, PLAIN_ARM),
            (FILM_CREATE, "global_film", FILM_ARM, PLAIN_ARM),
        ):
            config = _load(path)
            self.assertEqual(config.arm, arm, path.name)
            self.assertEqual(config.experiment_name, name, path.name)
            self.assertEqual(config.paired_arm, partner, path.name)
            self.assertEqual(set(config.active_domains), set(ACTIVE), path.name)
            self.assertNotIn(Domain.RIM_ONE_DL, config.held_out_domains)
            # still configured: the locked manifests cover all four domains
            self.assertEqual({d.domain for d in config.domains}, set(Domain))
            self.assertEqual((config.train_budget, config.val_budget, config.test_budget), (40, 10, 50))
            self.assertEqual(config.run_seeds, (42, 43, 44, 45, 46))
            runner._require_arm_policy(config)

    def test_film_arm_uses_the_agreed_per_domain_code(self) -> None:
        for path in (FILM_CONFIG, FILM_CREATE):
            config = _load(path)
            self.assertEqual(config.test_conditioning, "nearest_domain")
            self.assertEqual(config.training_config(Domain.DRISHTI_GS, 42, "artifacts/x").arm, "global_film")

    def test_configs_differ_only_in_the_arm(self) -> None:
        for plain_path, film_path in ((PLAIN_CONFIG, FILM_CONFIG), (PLAIN_CREATE, FILM_CREATE)):
            plain = json.loads(plain_path.read_text())
            film = json.loads(film_path.read_text())
            for key in self.ARM_ONLY_KEYS:
                plain.pop(key, None)
                film.pop(key, None)
            for key in self.PROTOCOL_ARM_ONLY_KEYS:
                plain["protocol"].pop(key, None)
                film["protocol"].pop(key, None)
            self.assertEqual(plain, film, f"{film_path.name} drifts from {plain_path.name}")

    def test_local_and_create_differ_only_in_data_roots(self) -> None:
        for local_path, create_path in ((PLAIN_CONFIG, PLAIN_CREATE), (FILM_CONFIG, FILM_CREATE)):
            local = json.loads(local_path.read_text())
            create = json.loads(create_path.read_text())
            for payload in (local, create):
                for domain in payload["domains"].values():
                    domain.pop("data_root")
            self.assertEqual(local, create, create_path.name)

    def test_step5_is_the_step4_lodo_protocol_under_new_names(self) -> None:
        """Same folds, budget, seeds, hyperparameters and test-time rule as Step 4's LODO."""

        for step5_path, step4_path in STEP4_TWINS.items():
            step5 = json.loads(step5_path.read_text())
            step4 = json.loads(step4_path.read_text())
            self.assertEqual(step5["stage"], runner.CONFIG_STAGE)
            for payload in (step5, step4):
                for key in self.STEP5_IDENTITY_KEYS:
                    payload.pop(key, None)
                for key in self.STEP5_PROTOCOL_IDENTITY_KEYS:
                    payload["protocol"].pop(key, None)
            self.assertEqual(step5, step4, f"{step5_path.name} drifts from {step4_path.name}")

    def test_oracle_code_is_refused_for_a_conditioned_arm(self) -> None:
        payload = json.loads(FILM_CONFIG.read_text())
        payload["film"]["test_conditioning"] = "oracle"
        config = _load(_temp_config(payload))
        with self.assertRaises(Stage3ConfigError) as caught:
            runner._require_arm_policy(config)
        self.assertIn("oracle", str(caught.exception))
        runner._require_arm_policy(replace(config, test_conditioning="nearest_image"))

    def test_each_runner_refuses_the_others_configs(self) -> None:
        code, _, stderr = _run_main(runner, ["--config", str(STEP4_TWINS[FILM_CONFIG]), "check", "--skip-mask-audit"])
        self.assertEqual(code, 2)
        self.assertIn(f"stage={runner.CONFIG_STAGE!r}", stderr)
        code, _, stderr = _run_main(fixed_runner, ["--config", str(FILM_CONFIG), "check", "--skip-mask-audit"])
        self.assertEqual(code, 2)
        self.assertIn(f"stage={fixed_runner.CONFIG_STAGE!r}", stderr)

    def test_an_inactive_held_out_domain_is_refused_on_the_command_line(self) -> None:
        code, _, stderr = _run_main(
            runner,
            ["--config", str(PLAIN_CONFIG), "run", "--held-out-domain", "rim_one_dl", "--seed", "42"],
        )
        self.assertEqual(code, 2)
        self.assertIn("not an active domain", stderr)
        code, _, stderr = _run_main(
            runner, ["--config", str(PLAIN_CONFIG), "run", "--held-out-domain", "drishti_gs", "--seed", "7"]
        )
        self.assertEqual(code, 2)
        self.assertIn("configured seeds", stderr)


# --------------------------------------------------------------------------
# A stand-in for run_experiment: writes what the engine writes, trains nothing
# --------------------------------------------------------------------------


def _conditioning_block(vocabulary: list[str], count: int, dice: float) -> dict:
    """The keys the Step 4/5 report reads, shaped as engine._conditioning_report writes them."""

    return {
        "vocabulary": vocabulary,
        "test_conditioning": "nearest_domain",
        "domain_decision": {"chosen_domain": vocabulary[0]},
        "selector_validation": {"accuracy": 1.0, "domain_level_accuracy": 1.0},
        "test": {
            "assignment_counts": {code: count if i == 0 else 0 for i, code in enumerate(vocabulary)},
            "nearest_image_counts": {code: count if i == 0 else 0 for i, code in enumerate(vocabulary)},
        },
        "fixed_code_sweep": {
            code: {s: {"dice_mean": dice - 0.05 * i} for s in ("disc", "cup")}
            for i, code in enumerate(vocabulary)
        },
        "best_fixed_code": {"disc": vocabulary[0], "cup": vocabulary[0]},
        "nearest_domain_minus_best_fixed_code_dice": {"disc": 0.0, "cup": 0.0},
    }


def _write_per_image_csv(path: Path, image_ids: list[str], dice: float) -> None:
    rows = [
        {"image_id": image_id, "structure": structure, "dice": f"{dice + 0.001 * index:.4f}",
         "iou": "0.6", "hd95": "10", "acc": "0.9", "tp": 1, "fp": 1, "fn": 1, "tn": 1}
        for index, image_id in enumerate(image_ids) for structure in ("disc", "cup")
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stub_engine(calls: list, plain_dice: float = 0.80, film_dice: float = 0.82):
    def fake_run_experiment(engine_config, project_root, **kwargs):
        calls.append((engine_config, kwargs))
        output = Path(project_root) / engine_config.output_dir
        if kwargs.get("smoke"):
            output = output.with_name(f"{output.name}_smoke")
        output.mkdir(parents=True, exist_ok=True)
        test = kwargs["split_records"]["test"]
        conditioned = engine_config.arm != "plain"
        dice = (film_dice if conditioned else plain_dice) + 0.001 * (engine_config.seed - 42)
        _write_per_image_csv(output / "test_per_image_metrics.csv", [r.sample_id for r in test], dice)
        vocabulary = sorted({record.domain for record in kwargs["split_records"]["train"]})
        return {
            "test": {s: {"dice_mean": dice} for s in ("disc", "cup")}
            | {"evaluated_sample_count": len(test)},
            "artifacts": {"per_image_metrics": str(output / "test_per_image_metrics.csv")},
            "resumed_from_epoch": None,
            "conditioning": _conditioning_block(vocabulary, len(test), dice) if conditioned else None,
        }

    return fake_run_experiment


@unittest.skipUnless(
    CONFIGS_PRESENT and REAL_MANIFEST.is_file() and DATASETS_PRESENT,
    "configs, committed manifest, or datasets not present",
)
class RunnerTests(unittest.TestCase):
    """_run_one against the real locked membership, with the engine stubbed out."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plain = _load(PLAIN_CONFIG)
        cls.film = _load(FILM_CONFIG)
        # Both configs share one domains block (ConfigTests), so one runtime serves both.
        cls.manifest, _, cls.records_by_key, cls.manifest_path = runner._load_locked_runtime(cls.plain)

    def setUp(self) -> None:
        (PROJECT_ROOT / "artifacts").mkdir(exist_ok=True)
        self.out = Path(tempfile.mkdtemp(dir=PROJECT_ROOT / "artifacts"))
        self.addCleanup(shutil.rmtree, self.out, True)

    def _run(self, config, config_path, held_out: Domain, smoke: bool = False):
        calls: list = []
        with mock.patch.object(runner, "run_experiment", _stub_engine(calls)), \
                contextlib.redirect_stdout(io.StringIO()):
            runner._run_one(
                config=config,
                config_path=config_path,
                manifest=self.manifest,
                records_by_key=self.records_by_key,
                manifest_path=self.manifest_path,
                held_out_domain=held_out,
                seed=42,
                smoke=smoke,
                requested_device="cpu",
                explicit_output=self.out / "run",
            )
        (engine_config, kwargs), = calls
        written = self.out / ("run_smoke" if smoke else "run")
        metrics = json.loads((written / "test_metrics.json").read_text())
        return engine_config, kwargs, written, metrics

    def test_plain_run_writes_the_step5_block_and_no_step4_block(self) -> None:
        engine_config, kwargs, written, metrics = self._run(self.plain, PLAIN_CONFIG, Domain.DRISHTI_GS)
        self.assertEqual(engine_config.arm, "plain")
        self.assertIsNone(kwargs["conditioning_reference"])
        self.assertEqual(kwargs["split_policy"], runner.STAGE5_LODO_SPLIT_POLICY)
        splits = kwargs["split_records"]
        self.assertEqual({r.domain for r in splits["train"]}, {"refuge_canon_val", "refuge_zeiss"})
        self.assertEqual({r.domain for r in splits["test"]}, {"drishti_gs"})
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 80, "val": 20, "test": 50})

        self.assertNotIn("fixed_lodo", metrics)
        block = metrics[runner.STAGE5_LODO_METADATA_KEY]
        self.assertEqual(block["protocol"], runner.STAGE5_LODO_PROTOCOL_NAME)
        self.assertEqual(block["arm"], PLAIN_ARM)
        self.assertEqual(block["conditioning_arm"], "plain")
        self.assertIsNone(block["test_conditioning"])
        self.assertEqual(block["paired_with"], FILM_ARM)
        self.assertEqual(block["held_out_domain"], "drishti_gs")
        self.assertEqual(block["source_domains"], ["refuge_canon_val", "refuge_zeiss"])
        self.assertEqual(block["active_domains"], [d.value for d in ACTIVE])
        self.assertEqual(block["inactive_domains"], ["rim_one_dl"])
        self.assertEqual(block["fold_shape"], "train on 2, test on 1")
        self.assertIsNone(block["conditioning_reference"])
        self.assertEqual(block["manifest_sha256"], hashlib.sha256(REAL_MANIFEST.read_bytes()).hexdigest())
        self.assertEqual(block["locked_split_counts"], {"train": 80, "val": 20, "test": 50})
        self.assertTrue(block["scientific_result"])
        self.assertFalse(block["smoke_rehearsal"])
        record = json.loads((written / runner.STAGE5_LODO_RUN_RECORD).read_text())
        self.assertEqual(record[runner.STAGE5_LODO_METADATA_KEY], block)
        self.assertTrue((written / runner.STAGE5_LODO_RESOLVED_CONFIG).is_file())
        self.assertFalse((written / "fixed_lodo_run.json").exists())

    def test_film_run_decides_its_code_from_the_held_out_reference_sample(self) -> None:
        engine_config, kwargs, _written, metrics = self._run(self.film, FILM_CONFIG, Domain.REFUGE_ZEISS)
        self.assertEqual((engine_config.arm, engine_config.test_conditioning), ("global_film", "nearest_domain"))
        reference = kwargs["conditioning_reference"]
        partition = next(p for p in self.manifest.budgeted_partitions if p.domain == Domain.REFUGE_ZEISS)
        self.assertEqual([r.sample_id for r in reference], [k.sample_id for k in partition.train])
        self.assertEqual({r.domain for r in reference}, {"refuge_zeiss"})
        tested = {r.sample_id for r in kwargs["split_records"]["test"]}
        self.assertTrue({r.sample_id for r in reference}.isdisjoint(tested))
        block = metrics[runner.STAGE5_LODO_METADATA_KEY]
        self.assertEqual(block["test_conditioning"], "nearest_domain")
        self.assertEqual(block["paired_with"], PLAIN_ARM)
        self.assertEqual(block["conditioning_reference"], {
            "partition": "held_out_budgeted_train", "domain": "refuge_zeiss",
            "image_count": 40, "labels_used": False, "disjoint_from_test": True,
        })

    def test_smoke_run_is_marked_and_decides_from_a_small_reference_sample(self) -> None:
        _engine_config, kwargs, written, metrics = self._run(
            self.film, FILM_CONFIG, Domain.DRISHTI_GS, smoke=True
        )
        self.assertEqual(written.name, "run_smoke")
        self.assertEqual(len(kwargs["conditioning_reference"]), runner.SMOKE_REFERENCE_IMAGES)
        block = metrics[runner.STAGE5_LODO_METADATA_KEY]
        self.assertTrue(block["smoke_rehearsal"])
        self.assertFalse(block["scientific_result"])
        self.assertEqual(block["locked_split_counts"], {"train": 80, "val": 20, "test": 50})
        self.assertEqual(block["executed_split_counts"], {"train": 2, "val": 2, "test": 1})

    def test_check_reports_three_folds_of_80_20_50(self) -> None:
        for path, reference in ((PLAIN_CONFIG, None), (FILM_CONFIG, 40)):
            code, stdout, stderr = _run_main(runner, ["--config", str(path), "check", "--skip-mask-audit"])
            self.assertEqual(code, 0, stderr)
            report = json.loads(stdout[stdout.index("{"):])
            self.assertEqual(report["protocol"], runner.STAGE5_LODO_PROTOCOL_NAME)
            self.assertEqual(report["inactive_domains"], ["rim_one_dl"])
            self.assertEqual(set(report["folds"]), {d.value for d in ACTIVE})
            for held_out, fold in report["folds"].items():
                self.assertEqual((fold["train"], fold["val"], fold["test"]), (80, 20, 50))
                self.assertNotIn(held_out, fold["source_domains"])
                self.assertEqual(fold["conditioning_reference"], reference)


# --------------------------------------------------------------------------
# Aggregation: Step 5 runs are read by the Step 5 aggregator and no one else
# --------------------------------------------------------------------------


def _write_run(
    base: Path,
    manifest_sha: str,
    fold,
    *,
    arm: str,
    seed: int,
    dice: float,
    conditioned: bool,
    metadata_key: str = runner.STAGE5_LODO_METADATA_KEY,
    protocol: str = runner.STAGE5_LODO_PROTOCOL_NAME,
    smoke: bool = False,
) -> Path:
    held_out = fold.held_out_domain
    sources = sorted({sample.domain.value for sample in fold.train})
    run = base / f"{arm}_{held_out.value}_seed_{seed}{'_smoke' if smoke else ''}"
    run.mkdir(parents=True)
    value = dice + 0.001 * (seed - 42)
    _write_per_image_csv(run / "test_per_image_metrics.csv", [s.sample_id for s in fold.test], value)
    payload: dict = {
        "test": {"evaluated_sample_count": len(fold.test)},
        metadata_key: {
            "protocol": protocol, "arm": arm, "held_out_domain": held_out.value,
            "source_domains": sources, "run_seed": seed,
            "budget": {"train": TRAIN, "val": VAL, "test": TEST, "subsample_seed": 42},
            "manifest_sha256": manifest_sha,
            "completed_at_utc": f"2026-09-23T00:00:{seed - 40:02d}+00:00",
            "smoke_rehearsal": smoke, "scientific_result": not smoke,
        },
    }
    if conditioned:
        payload["conditioning"] = _conditioning_block(sources, len(fold.test), value)
    (run / "test_metrics.json").write_text(json.dumps(payload))
    return run


class AggregatorTests(unittest.TestCase):
    """A run root holding both Step 4 and Step 5 grids, as artifacts/runs will."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        manifest = _manifest()
        self.manifest_path = self.directory / "single_source_manifest.json"
        write_single_source_manifest(manifest, self.manifest_path)
        self.manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.folds = fixed_runner.fixed_lodo_folds(manifest, ACTIVE)
        self.runs = self.directory / "runs"

    def write_step5(self, seeds=(42,), film_smoke: bool = False) -> None:
        for fold in self.folds:
            for seed in seeds:
                _write_run(self.runs, self.manifest_sha, fold, arm=PLAIN_ARM, seed=seed,
                           dice=0.80, conditioned=False)
                _write_run(self.runs, self.manifest_sha, fold, arm=FILM_ARM, seed=seed,
                           dice=0.83, conditioned=True, smoke=film_smoke)

    def write_step4(self, seeds=(42,)) -> None:
        for fold in self.folds:
            for seed in seeds:
                for arm, dice, conditioned in ((STEP4_PLAIN_ARM, 0.70, False), (STEP4_FILM_ARM, 0.71, True)):
                    _write_run(self.runs, self.manifest_sha, fold, arm=arm, seed=seed, dice=dice,
                               conditioned=conditioned, metadata_key="fixed_lodo",
                               protocol="leave_one_domain_out_fixed_budget")

    def report(self, module, *seeds: int, extra: tuple[str, ...] = ()) -> tuple[int, str, str, str]:
        report_path = self.directory / f"{module.__name__}.md"
        code, stdout, stderr = _run_main(module, [
            "--run-root", str(self.runs),
            "--manifest", str(self.manifest_path),
            "--expected-seeds", *map(str, seeds),
            "--report-out", str(report_path),
            "--csv-out", str(self.directory / f"{module.__name__}.csv"),
            *extra,
        ])
        return code, stdout, stderr, report_path.read_text() if report_path.is_file() else ""

    def test_each_discovery_sees_only_its_own_runner(self) -> None:
        self.write_step4()
        self.write_step5()
        step4 = discover_fixed_runs([self.runs])
        step5 = discover_fixed_runs(
            [self.runs], runner.STAGE5_LODO_METADATA_KEY, runner.STAGE5_LODO_PROTOCOL_NAME
        )
        self.assertEqual({r.arm for r in step4}, {STEP4_PLAIN_ARM, STEP4_FILM_ARM})
        self.assertEqual({r.arm for r in step5}, {PLAIN_ARM, FILM_ARM})
        self.assertEqual(len(step4), len(step5))

    def test_one_seed_per_arm_reports_without_a_seed_spread(self) -> None:
        self.write_step4()
        self.write_step5()
        code, stdout, stderr, report = self.report(step5_aggregator, 42)
        self.assertEqual(code, 0, stderr)
        self.assertIn("plain: 3 runs | film: 3 runs | paired tests: 6", stdout)
        self.assertTrue(report.startswith(f"# {step5_aggregator.REPORT_TITLE}\n"))
        self.assertIn("`aggregate_stage5_lodo.py`", report)
        self.assertIn(f"Plain arm: `{PLAIN_ARM}`. FiLM arm: `{FILM_ARM}`.", report)
        self.assertIn("(1 seed)", report)
        for fold in self.folds:
            self.assertIn(f"| `{fold.held_out_domain.value}` | disc | {TEST} |", report)
            self.assertIn("+0.0300", report)

    def test_two_seeds_report_the_seed_spread(self) -> None:
        self.write_step5(seeds=(42, 43))
        code, _stdout, stderr, report = self.report(step5_aggregator, 42, 43)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("(1 seed)", report)
        self.assertIn(" ± ", report)

    def test_the_step4_report_is_unchanged_and_never_sees_step5_runs(self) -> None:
        self.write_step4()
        self.write_step5()
        code, stdout, stderr, report = self.report(step4_aggregator, 42)
        self.assertEqual(code, 0, stderr)
        self.assertIn("plain: 3 runs | film: 3 runs", stdout)
        self.assertTrue(report.startswith(
            "# Step 4: Global FiLM against the plain U-Net, fixed-budget leave-one-domain-out\n"
        ))
        self.assertIn("`aggregate_stage4_film.py`", report)
        self.assertNotIn("stage5", report)
        # Even asked by name, the Step 4 tool cannot pick up a Step 5 run.
        code, _stdout, stderr, _report = self.report(
            step4_aggregator, 42, extra=("--plain-arm", PLAIN_ARM, "--film-arm", FILM_ARM)
        )
        self.assertEqual(code, 2)
        self.assertIn("No scientific runs", stderr)

    def test_smoke_runs_and_missing_arms_are_clean_failures(self) -> None:
        self.write_step5(film_smoke=True)
        code, _stdout, stderr, _report = self.report(step5_aggregator, 42)
        self.assertEqual(code, 2)
        self.assertIn(f"No scientific runs found for arm {FILM_ARM!r}", stderr)

    def test_a_run_from_another_manifest_is_refused(self) -> None:
        self.write_step5()
        other = self.directory / "other_manifest.json"
        other.write_text(self.manifest_path.read_text() + "\n")
        code, _stdout, stderr = _run_main(step5_aggregator, [
            "--run-root", str(self.runs), "--manifest", str(other), "--expected-seeds", "42",
        ])
        self.assertEqual(code, 2)
        self.assertIn("different manifest", stderr)


@unittest.skipUnless(
    CONFIGS_PRESENT and REAL_MANIFEST.is_file() and DATASETS_PRESENT,
    "configs, committed manifest, or datasets not present",
)
class RunnerToAggregatorContractTests(unittest.TestCase):
    """What run_stage5_lodo writes is exactly what aggregate_stage5_lodo reads.

    Every fold of both arms is run for two seeds on the real locked membership
    with the engine stubbed out, then aggregated from disk.
    """

    SEEDS = (42, 43)

    @classmethod
    def setUpClass(cls) -> None:
        plain, film = _load(PLAIN_CONFIG), _load(FILM_CONFIG)
        manifest, _, records_by_key, manifest_path = runner._load_locked_runtime(plain)
        (PROJECT_ROOT / "artifacts").mkdir(exist_ok=True)
        cls.out = Path(tempfile.mkdtemp(dir=PROJECT_ROOT / "artifacts"))
        calls: list = []
        with mock.patch.object(runner, "run_experiment", _stub_engine(calls)), \
                contextlib.redirect_stdout(io.StringIO()):
            for config, path in ((plain, PLAIN_CONFIG), (film, FILM_CONFIG)):
                for held_out in ACTIVE:
                    for seed in cls.SEEDS:
                        runner._run_one(
                            config=config, config_path=path, manifest=manifest,
                            records_by_key=records_by_key, manifest_path=manifest_path,
                            held_out_domain=held_out, seed=seed, smoke=False,
                            requested_device="cpu",
                            explicit_output=cls.out / f"{config.arm}_{held_out.value}_{seed}",
                        )
        cls.calls = len(calls)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_the_aggregator_reads_what_the_runner_writes(self) -> None:
        self.assertEqual(self.calls, 2 * len(ACTIVE) * len(self.SEEDS))
        report_path, csv_path = self.out / "report.md", self.out / "cells.csv"
        code, stdout, stderr = _run_main(step5_aggregator, [
            "--run-root", str(self.out),
            "--manifest", str(REAL_MANIFEST),
            "--expected-seeds", *map(str, self.SEEDS),
            "--report-out", str(report_path),
            "--csv-out", str(csv_path),
        ])
        self.assertEqual(code, 0, stderr)
        runs_per_arm = len(ACTIVE) * len(self.SEEDS)
        self.assertIn(
            f"plain: {runs_per_arm} runs | film: {runs_per_arm} runs | paired tests: 6", stdout
        )
        report = report_path.read_text()
        for domain in ACTIVE:
            self.assertIn(f"| `{domain.value}` | disc | 50 |", report)
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        tests = [r for r in rows if r["kind"] == "paired_test"]
        self.assertEqual(len(tests), 6)
        for row in tests:
            self.assertAlmostEqual(float(row["mean_difference"]), 0.02, places=6)
        conditioning = [r for r in rows if r["kind"] == "conditioning"]
        self.assertEqual({r["held_out_domain"] for r in conditioning}, {d.value for d in ACTIVE})

    def test_the_runs_are_invisible_to_the_stage3_and_step4_tools(self) -> None:
        self.assertEqual(discover_fixed_runs([self.out]), ())
        code, _stdout, stderr = _run_main(step4_aggregator, [
            "--run-root", str(self.out), "--manifest", str(REAL_MANIFEST),
            "--plain-arm", PLAIN_ARM, "--film-arm", FILM_ARM, "--expected-seeds", "42",
        ])
        self.assertEqual(code, 2)
        self.assertIn("No scientific runs", stderr)


if __name__ == "__main__":
    unittest.main()
