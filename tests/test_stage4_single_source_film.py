"""Contract tests for Global FiLM under the train-on-one, test-on-two protocol.

One source domain is one code, so this arm is a control: the FiLM generators see
a constant input and the modulation is a fixed per-channel affine. These tests
pin the plumbing that makes the control honest -- the runner says it is
degenerate, scores only the active targets on exactly the plain arm's images,
and the aggregator pairs FiLM with the *existing* plain grid per (source,
target) without ever mixing the two arms or a plain model trained elsewhere.
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
from pathlib import Path
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import aggregate_stage4_single_source_film as film_agg  # noqa: E402
import run_stage3_lodo_1_3 as runner  # noqa: E402
from aggregate_stage3_1_3 import (  # noqa: E402
    Stage3SingleSourceReportError,
    discover_runs,
    select_scientific_runs,
)
from spfilm.film.conditioning import (  # noqa: E402
    DESCRIPTOR_NAMES,
    DomainVocabulary,
    NearestDomainSelector,
)
from spfilm.lodo import Domain, DomainPartitions, SampleKey  # noqa: E402
from spfilm.metrics import summarise_per_image_csv  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    write_single_source_manifest,
)
from spfilm.stage3 import Stage3ConfigError  # noqa: E402
from spfilm.stage3_single_source import Stage3SingleSourceConfig  # noqa: E402


FILM_CONFIG = PROJECT_ROOT / "configs" / "stage4_single_source_global_film_3dom.json"
FILM_CREATE = PROJECT_ROOT / "configs" / "stage4_single_source_global_film_3dom_create.json"
STAGE3_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo_single.json"
STAGE3_CREATE = PROJECT_ROOT / "configs" / "stage3_lodo_single_create.json"
REAL_MANIFEST = PROJECT_ROOT / "splits" / "single_source" / "single_source_manifest.json"
DATASETS_PRESENT = (PROJECT_ROOT.parents[1] / "datasets" / "REFUGE").is_dir()

ACTIVE = (Domain.DRISHTI_GS, Domain.REFUGE_CANON_VAL, Domain.REFUGE_ZEISS)
PLAIN_ARM = "stage3_single_source_plain_unet"
FILM_ARM = "stage4_single_source_global_film_3dom"
SEEDS = (42, 43)
PARENT_SHA = "a" * 64
CONFIG_SHA = "c" * 64


def _write_temp_config(payload: dict) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
        return Path(stream.name)


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


@unittest.skipUnless(FILM_CONFIG.is_file() and FILM_CREATE.is_file(), "configs not present")
class ConfigTests(unittest.TestCase):
    ARM_ONLY_KEYS = {"arm", "film", "experiment_name", "output_dir"}
    PROTOCOL_ARM_ONLY_KEYS = {
        "policy",
        "paired_arm",
        "source_domains",
        "inactive_domains",
        "inactive_domains_note",
        "fold_shape",
        "degeneracy_note",
        "pairing_rationale",
        "test_conditioning_note",
        "target_test_rationale",
    }
    FOLD_MEMBERSHIP_ARM_ONLY_KEYS = {"tests"}

    def test_loads_with_the_global_film_arm_and_three_active_domains(self) -> None:
        for path in (FILM_CONFIG, FILM_CREATE):
            config = Stage3SingleSourceConfig.from_json(path)
            self.assertEqual(config.arm, "global_film")
            self.assertEqual(config.experiment_name, FILM_ARM)
            self.assertEqual(config.paired_arm, PLAIN_ARM)
            self.assertEqual(set(config.active_domains), set(ACTIVE), path.name)
            self.assertNotIn(Domain.RIM_ONE_DL, config.source_domains)
            # still discovered and validated: the locked manifests cover it
            self.assertEqual({d.domain for d in config.domains}, set(Domain))
            self.assertEqual(config.film_levels, 5)
            self.assertEqual(config.film_embedding_dim, 64)
            self.assertEqual(config.film_hidden_dim, 256)
            self.assertEqual(config.film_clamp, 5.0)

    def test_uses_the_per_image_rule_because_the_test_set_is_pooled(self) -> None:
        """The per-domain rule needs a single-domain test set; with one code it is moot."""

        for path in (FILM_CONFIG, FILM_CREATE):
            self.assertEqual(
                Stage3SingleSourceConfig.from_json(path).test_conditioning, "nearest_image"
            )

    def test_differs_from_the_stage3_plain_config_only_in_the_arm_and_domain_set(self) -> None:
        if not (STAGE3_CONFIG.is_file() and STAGE3_CREATE.is_file()):
            self.skipTest("stage 3 configs not present")
        for plain_path, film_path in ((STAGE3_CONFIG, FILM_CONFIG), (STAGE3_CREATE, FILM_CREATE)):
            plain = json.loads(plain_path.read_text())
            film = json.loads(film_path.read_text())
            self.assertEqual(
                plain["protocol"]["source_domains"],
                [*film["protocol"]["source_domains"], "rim_one_dl"],
            )
            for key in self.ARM_ONLY_KEYS:
                plain.pop(key, None)
                film.pop(key, None)
            for key in self.PROTOCOL_ARM_ONLY_KEYS:
                plain["protocol"].pop(key, None)
                film["protocol"].pop(key, None)
            for key in self.FOLD_MEMBERSHIP_ARM_ONLY_KEYS:
                plain["protocol"]["fold_membership"].pop(key, None)
                film["protocol"]["fold_membership"].pop(key, None)
            self.assertEqual(plain, film, f"{film_path.name} drifts from {plain_path.name}")

    def test_local_and_create_differ_only_in_data_roots(self) -> None:
        local = json.loads(FILM_CONFIG.read_text())
        create = json.loads(FILM_CREATE.read_text())
        for payload in (local, create):
            for domain in payload["domains"].values():
                domain.pop("data_root")
        self.assertEqual(local, create)

    def test_training_config_forwards_the_film_settings(self) -> None:
        engine_config = Stage3SingleSourceConfig.from_json(FILM_CONFIG).training_config(
            Domain.REFUGE_ZEISS, 42, "artifacts/x"
        )
        self.assertEqual(engine_config.arm, "global_film")
        self.assertEqual(engine_config.test_conditioning, "nearest_image")
        self.assertEqual(engine_config.film_levels, 5)


# --------------------------------------------------------------------------
# Runner: targets restricted to the active set, degeneracy stamped
# --------------------------------------------------------------------------


def _synthetic_manifest(train: int = 4, val: int = 2, test: int = 3) -> SingleSourceManifest:
    partitions = []
    strata = {}
    for domain in sorted(Domain, key=lambda item: item.value):
        prefix = domain.value[:3]
        keys = {
            "train": tuple(SampleKey(domain, f"{prefix}_tr{n}") for n in range(train)),
            "val": tuple(SampleKey(domain, f"{prefix}_va{n}") for n in range(val)),
            "test": tuple(SampleKey(domain, f"{prefix}_te{n}") for n in range(test)),
        }
        partitions.append(DomainPartitions(domain=domain, **keys))
        for partition in keys.values():
            for key in partition:
                strata[key] = "all"
    return SingleSourceManifest.build(
        PARENT_SHA, tuple(partitions), train, val, test, strata, 42
    )


class ActiveTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _synthetic_manifest()

    def _fold(self, source: Domain):
        return next(f for f in self.manifest.folds if f.source_domain == source)

    def test_targets_are_the_other_active_domains_only(self) -> None:
        for source in ACTIVE:
            targets = runner.active_target_domains(self._fold(source), ACTIVE)
            self.assertEqual(set(targets), set(ACTIVE) - {source})
            self.assertNotIn(Domain.RIM_ONE_DL, targets)
            self.assertEqual(len(targets), 2)

    def test_targets_keep_the_manifest_order(self) -> None:
        for source in ACTIVE:
            fold = self._fold(source)
            targets = runner.active_target_domains(fold, ACTIVE)
            self.assertEqual(
                targets, tuple(d for d in fold.target_domains if d != Domain.RIM_ONE_DL)
            )

    def test_active_targets_score_exactly_the_four_domain_protocol_images(self) -> None:
        """Dropping RIM-ONE-DL must not move the images a target is scored on."""

        for source in ACTIVE:
            fold = self._fold(source)
            four = {d: fold.test_samples(d) for d in fold.target_domains}
            for target in runner.active_target_domains(fold, ACTIVE):
                self.assertEqual(fold.test_samples(target), four[target])

    def test_none_means_every_manifest_target(self) -> None:
        fold = self._fold(Domain.REFUGE_ZEISS)
        self.assertEqual(runner.active_target_domains(fold, None), fold.target_domains)
        self.assertEqual(len(runner.active_target_domains(fold, None)), 3)

    def test_source_outside_the_active_set_is_refused(self) -> None:
        with self.assertRaises(Stage3ConfigError):
            runner.active_target_domains(self._fold(Domain.RIM_ONE_DL), ACTIVE)

    def test_no_active_target_is_refused(self) -> None:
        with self.assertRaises(Stage3ConfigError):
            runner.active_target_domains(self._fold(Domain.REFUGE_ZEISS), (Domain.REFUGE_ZEISS,))


class ProseTests(unittest.TestCase):
    def test_three_target_policy_is_the_historical_string(self) -> None:
        self.assertEqual(
            runner.single_source_split_policy(3),
            "locked single-source: source train/val only; three named target tests; "
            "source test and target train/val excluded",
        )
        self.assertEqual(runner.SINGLE_SOURCE_SPLIT_POLICY, runner.single_source_split_policy(3))

    def test_two_target_policy_says_two(self) -> None:
        self.assertIn("two named target tests", runner.single_source_split_policy(2))


@unittest.skipUnless(
    FILM_CONFIG.is_file() and REAL_MANIFEST.is_file() and DATASETS_PRESENT,
    "film config, locked manifest, or datasets not present",
)
class RunnerAcceptanceTests(unittest.TestCase):
    """The runner accepts the FiLM arm as a control and says so; run_experiment is stubbed."""

    def test_check_accepts_the_film_arm_and_prints_the_degeneracy_warning(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as stderr, contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout:
            code = runner.main(["--config", str(FILM_CONFIG), "check", "--skip-mask-audit"])
        self.assertEqual(code, 0)
        self.assertIn("one code", stderr.getvalue())
        text = stdout.getvalue()
        report = json.loads(text[text.index("{"):])
        self.assertEqual(report["inactive_domains"], ["rim_one_dl"])
        self.assertEqual(set(report["folds"]), {d.value for d in ACTIVE})
        for source, fold in report["folds"].items():
            self.assertEqual(set(fold["test_by_domain"]), {d.value for d in ACTIVE} - {source})
            self.assertEqual(fold["test"], 100)

    def test_per_domain_rule_is_refused_because_the_test_set_is_pooled(self) -> None:
        payload = json.loads(FILM_CONFIG.read_text())
        payload["film"]["test_conditioning"] = "nearest_domain"
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = runner.main(
                ["--config", str(_write_temp_config(payload)), "check", "--skip-mask-audit"]
            )
        self.assertEqual(code, 2)
        self.assertIn("nearest_image", stderr.getvalue())

    def _run_with_stubbed_engine(self, config_path: Path, source: Domain) -> dict:
        config = Stage3SingleSourceConfig.from_json(config_path)
        manifest, _, records_by_key, manifest_path = runner._load_locked_runtime(config)
        captured: dict = {}

        def fake_run_experiment(engine_config, project_root, **kwargs):
            captured["engine_config"] = engine_config
            captured["kwargs"] = kwargs
            names = list(kwargs["extra_test_sets"])
            block = {
                "disc": {"dice_mean": 0.9},
                "cup": {"dice_mean": 0.8},
                "evaluated_sample_count": 1,
            }
            return {
                "test": {"note": "pooled"},
                "test_by_name": {name: dict(block) for name in names},
                "artifacts": {},
                "resumed_from_epoch": None,
            }

        out_dir = Path(tempfile.mkdtemp(dir=PROJECT_ROOT / "artifacts"))
        self.addCleanup(shutil.rmtree, out_dir, True)
        with mock.patch.object(runner, "run_experiment", fake_run_experiment), \
                contextlib.redirect_stdout(io.StringIO()):
            report = runner._run_one(
                config=config,
                config_path=config_path,
                manifest=manifest,
                records_by_key=records_by_key,
                manifest_path=manifest_path,
                source_domain=source,
                seed=42,
                smoke=True,
                requested_device="cpu",
                explicit_output=out_dir / "run",
            )
        captured["report"] = report
        captured["written"] = json.loads(
            (out_dir / "run_smoke" / "test_metrics.json").read_text()
        )
        return captured

    def test_film_run_scores_two_targets_and_stamps_degenerate_conditioning(self) -> None:
        captured = self._run_with_stubbed_engine(FILM_CONFIG, Domain.REFUGE_ZEISS)
        self.assertEqual(captured["engine_config"].arm, "global_film")
        self.assertEqual(
            set(captured["kwargs"]["extra_test_sets"]), {"drishti_gs", "refuge_canon_val"}
        )
        self.assertIn("two named target tests", captured["kwargs"]["split_policy"])
        pooled_domains = {r.domain for r in captured["kwargs"]["split_records"]["test"]}
        self.assertEqual(pooled_domains, {"drishti_gs", "refuge_canon_val"})
        metadata = captured["written"]["single_source"]
        self.assertTrue(metadata["degenerate_conditioning"])
        self.assertEqual(metadata["conditioning_arm"], "global_film")
        self.assertEqual(metadata["arm"], FILM_ARM)
        self.assertEqual(metadata["paired_with"], PLAIN_ARM)
        self.assertEqual(metadata["target_domains"], ["drishti_gs", "refuge_canon_val"])
        self.assertEqual(metadata["active_domains"], [d.value for d in ACTIVE])
        self.assertEqual(metadata["inactive_domains"], ["rim_one_dl"])
        self.assertEqual(metadata["fold_shape"], "train on 1, test on 2")
        self.assertEqual(
            set(metadata["locked_split_counts"]["test_by_domain"]),
            {"drishti_gs", "refuge_canon_val"},
        )
        self.assertEqual(set(captured["report"]["test_by_domain"]), {"drishti_gs", "refuge_canon_val"})
        warning = captured["report"]["test_pooled"]["pooling"]["warning"]
        self.assertIn("two acquisition domains", warning)
        self.assertNotIn("RIM-ONE-DL", warning)

    def test_plain_four_domain_run_keeps_its_historical_metadata(self) -> None:
        if not STAGE3_CONFIG.is_file():
            self.skipTest("stage 3 config not present")
        captured = self._run_with_stubbed_engine(STAGE3_CONFIG, Domain.REFUGE_ZEISS)
        self.assertEqual(
            captured["kwargs"]["split_policy"],
            "locked single-source: source train/val only; three named target tests; "
            "source test and target train/val excluded",
        )
        metadata = captured["written"]["single_source"]
        self.assertFalse(metadata["degenerate_conditioning"])
        self.assertEqual(metadata["conditioning_arm"], "plain")
        self.assertEqual(metadata["paired_with"], "stage3_lodo_fixed_budget_plain_unet")
        self.assertEqual(metadata["target_domains"], ["drishti_gs", "refuge_canon_val", "rim_one_dl"])
        self.assertEqual(metadata["inactive_domains"], [])
        self.assertEqual(
            captured["report"]["test_pooled"]["pooling"]["warning"],
            "Pooled across three acquisition domains and therefore not a per-domain "
            "result. Its HD95 also mixes RIM-ONE-DL, whose native frame differs from "
            "the others. Report test_by_domain instead.",
        )
        self.assertIn("The three target domains", captured["report"]["reporting_rule"])


# --------------------------------------------------------------------------
# One-domain vocabulary and selector
# --------------------------------------------------------------------------


class OneCodeSelectorTests(unittest.TestCase):
    def test_one_domain_vocabulary_has_a_single_code(self) -> None:
        vocabulary = DomainVocabulary.from_domains(["refuge_zeiss"])
        self.assertEqual(len(vocabulary), 1)
        self.assertEqual(vocabulary.index_of("refuge_zeiss"), 0)

    def test_selector_fits_on_one_domain_and_always_picks_it(self) -> None:
        torch.manual_seed(0)
        vocabulary = DomainVocabulary.from_domains(["refuge_zeiss"])
        descriptors = torch.rand(12, len(DESCRIPTOR_NAMES))
        selector = NearestDomainSelector.fit(descriptors, ["refuge_zeiss"] * 12, vocabulary)
        self.assertTrue(bool((selector.scale > 0).all()))
        # within-domain spread is the only spread there is
        torch.testing.assert_close(
            selector.scale, descriptors.std(dim=0, unbiased=False), rtol=1e-5, atol=1e-6
        )
        probes = torch.rand(7, len(DESCRIPTOR_NAMES)) * 10
        self.assertEqual(selector.distances(probes).shape, (7, 1))
        self.assertTrue(bool((selector.distances(probes).argmin(dim=1) == 0).all()))

    def test_constant_descriptors_fall_back_to_unit_scale(self) -> None:
        vocabulary = DomainVocabulary.from_domains(["refuge_zeiss"])
        descriptors = torch.full((5, len(DESCRIPTOR_NAMES)), 0.3)
        selector = NearestDomainSelector.fit(descriptors, ["refuge_zeiss"] * 5, vocabulary)
        torch.testing.assert_close(selector.scale, torch.ones(len(DESCRIPTOR_NAMES)))


# --------------------------------------------------------------------------
# Aggregation: FiLM paired with the existing plain grid, per source
# --------------------------------------------------------------------------


def _small_manifest() -> SingleSourceManifest:
    return _synthetic_manifest(train=2, val=1, test=4)


def _rows(samples, dice_for):
    return [
        {
            "image_id": sample.sample_id,
            "structure": structure,
            "dice": f"{dice_for(sample.sample_id, structure):.12g}",
            "iou": "0.5",
            "hd95": "10",
            "acc": "0.95",
            "tp": 100,
            "fp": 10,
            "fn": 9,
            "tn": 900,
        }
        for sample in samples
        for structure in ("disc", "cup")
    ]


def _write_run(
    base: Path,
    manifest: SingleSourceManifest,
    manifest_sha: str,
    *,
    arm: str,
    source: Domain,
    seed: int,
    targets,
    dice_for,
    completed: str | None = None,
    film: bool = False,
    stamp_degenerate: bool = True,
) -> Path:
    fold = next(f for f in manifest.folds if f.source_domain == source)
    run = base / f"{arm}_{source.value}_seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    by_domain = {}
    for domain in targets:
        samples = fold.test_samples(domain)
        csv_path = run / f"test_{domain.value}_per_image_metrics.csv"
        rows = _rows(samples, dice_for)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        block = dict(summarise_per_image_csv(csv_path))
        block["evaluated_sample_count"] = len(samples)
        if domain == Domain.RIM_ONE_DL:
            block["hd95_unit"] = "native pixels"
        if film:
            block["conditioning"] = {
                "assignment_counts": {source.value: len(samples)},
                "true_domains_in_vocabulary": False,
            }
        by_domain[domain.value] = block
    metadata = {
        "protocol": "single_source_locked_multi_target_test",
        "arm": arm,
        "source_domain": source.value,
        "target_domains": [d.value for d in targets],
        "run_seed": seed,
        "budget": {"train": 2, "val": 1, "test": 4, "subsample_seed": 42},
        "manifest_sha256": manifest_sha,
        "parent_manifest_sha256": PARENT_SHA,
        "config_sha256": CONFIG_SHA,
        "git_revision": "test",
        "smoke_rehearsal": False,
        "scientific_result": True,
        "completed_at_utc": completed or f"2026-09-12T0{seed - 42}:00:00+00:00",
    }
    payload = {"test_by_domain": by_domain, "test_pooled": {"note": "ignored"}}
    if film:
        metadata["active_domains"] = [d.value for d in ACTIVE]
        metadata["inactive_domains"] = ["rim_one_dl"]
        metadata["conditioning_arm"] = "global_film"
        metadata["degenerate_conditioning"] = stamp_degenerate
        payload["conditioning"] = {
            "vocabulary": [source.value],
            "selector_validation": {"accuracy": 1.0},
            "fixed_code_sweep": {source.value: {}},
        }
    payload["single_source"] = metadata
    (run / "test_metrics.json").write_text(json.dumps(payload, indent=2))
    return run


class AggregationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.manifest = _small_manifest()
        self.manifest_path = self.directory / "single_source_manifest.json"
        write_single_source_manifest(self.manifest, self.manifest_path)
        self.manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.runs_root = self.directory / "runs"

    @staticmethod
    def _plain_dice(image_id: str, structure: str) -> float:
        return 0.80 + 0.01 * (hash((image_id, structure)) % 7)

    def write_plain_grid(self, seeds=SEEDS) -> None:
        for source in sorted(Domain, key=lambda d: d.value):
            fold = next(f for f in self.manifest.folds if f.source_domain == source)
            for seed in seeds:
                _write_run(
                    self.runs_root, self.manifest, self.manifest_sha,
                    arm=PLAIN_ARM, source=source, seed=seed,
                    targets=fold.target_domains, dice_for=self._plain_dice,
                )

    def write_film_grid(self, seeds=SEEDS, shift=None, **kwargs) -> None:
        for source in ACTIVE:
            fold = next(f for f in self.manifest.folds if f.source_domain == source)
            targets = tuple(d for d in fold.target_domains if d in ACTIVE)
            offset = 0.0 if shift is None or shift[0] != source else shift[1]

            def dice_for(image_id: str, structure: str, offset=offset) -> float:
                return self._plain_dice(image_id, structure) + offset

            for seed in seeds:
                _write_run(
                    self.runs_root, self.manifest, self.manifest_sha,
                    arm=FILM_ARM, source=source, seed=seed,
                    targets=targets, dice_for=dice_for, film=True, **kwargs,
                )

    def load_both(self, seeds=SEEDS):
        film = film_agg.load_arm(
            [self.runs_root], FILM_ARM, seeds, self.manifest, self.manifest_path
        )
        plain = film_agg.load_arm(
            [self.runs_root], PLAIN_ARM, seeds, self.manifest, self.manifest_path,
            sources=ACTIVE,
        )
        return plain, film


class SelectionTests(AggregationFixture):
    def test_mixed_arms_sharing_cells_are_refused_without_a_filter(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            select_scientific_runs(discover_runs([self.runs_root]), SEEDS)
        self.assertIn("mix experimental arms", str(caught.exception))

    def test_arm_filter_picks_one_arm_and_ignores_completion_order(self) -> None:
        self.write_plain_grid()
        # FiLM runs finish later; without the arm check the dedup would keep them.
        self.write_film_grid()
        for run in self.runs_root.glob(f"{FILM_ARM}_*"):
            payload = json.loads((run / "test_metrics.json").read_text())
            payload["single_source"]["completed_at_utc"] = "2026-12-31T00:00:00+00:00"
            (run / "test_metrics.json").write_text(json.dumps(payload))
        plain = select_scientific_runs(discover_runs([self.runs_root]), SEEDS, arm=PLAIN_ARM)
        self.assertEqual({r.identity.arm for r in plain}, {PLAIN_ARM})
        self.assertEqual(len(plain), len(Domain) * len(SEEDS))
        film = select_scientific_runs(discover_runs([self.runs_root]), SEEDS, arm=FILM_ARM)
        self.assertEqual({r.identity.arm for r in film}, {FILM_ARM})
        self.assertEqual(len(film), len(ACTIVE) * len(SEEDS))

    def test_unknown_arm_is_refused(self) -> None:
        self.write_plain_grid()
        with self.assertRaises(Stage3SingleSourceReportError):
            select_scientific_runs(discover_runs([self.runs_root]), SEEDS, arm="nope")


class PairingTests(AggregationFixture):
    def test_film_scores_a_subset_of_the_manifest_targets_and_passes_membership(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        plain, film = self.load_both()
        for run in film:
            self.assertEqual(len(run.target_domains), 2)
            self.assertNotIn(Domain.RIM_ONE_DL, run.target_domains)
            counts = film_agg.verify_target_membership(run, self.manifest)
            self.assertEqual(set(counts.values()), {4})
        self.assertEqual({r.identity.source_domain for r in plain}, set(ACTIVE))

    def test_a_wrong_image_set_is_refused(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        run_dir = self.runs_root / f"{FILM_ARM}_refuge_zeiss_seed_42"
        csv_path = run_dir / "test_drishti_gs_per_image_metrics.csv"
        text = csv_path.read_text().replace("dri_te0", "dri_te9")
        csv_path.write_text(text)
        with self.assertRaises(Stage3SingleSourceReportError):
            film_agg.load_arm([self.runs_root], FILM_ARM, SEEDS, self.manifest, self.manifest_path)

    def test_paired_targets_are_the_plain_targets_minus_the_inactive_domain(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        plain, film = self.load_both()
        pairs = film_agg.paired_targets(plain, film)
        self.assertEqual(set(pairs), set(ACTIVE))
        for source, targets in pairs.items():
            self.assertEqual(set(targets), set(ACTIVE) - {source})

    def test_target_sets_that_do_not_match_after_dropping_rim_one_are_refused(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        run_dir = self.runs_root / f"{FILM_ARM}_refuge_zeiss_seed_42"
        for seed in SEEDS:
            run_dir = self.runs_root / f"{FILM_ARM}_refuge_zeiss_seed_{seed}"
            payload = json.loads((run_dir / "test_metrics.json").read_text())
            payload["single_source"]["target_domains"] = ["drishti_gs"]
            payload["test_by_domain"].pop("refuge_canon_val")
            (run_dir / "test_metrics.json").write_text(json.dumps(payload))
        plain, film = self.load_both()
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            film_agg.paired_targets(plain, film)
        self.assertIn("must match to pair", str(caught.exception))

    def test_missing_plain_source_is_refused(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        plain, film = self.load_both()
        plain = tuple(r for r in plain if r.identity.source_domain != Domain.DRISHTI_GS)
        with self.assertRaises(Stage3SingleSourceReportError):
            film_agg.paired_targets(plain, film)


class PairedTestTests(AggregationFixture):
    def test_identical_arms_give_twelve_null_results(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        plain, film = self.load_both()
        pairs = film_agg.paired_targets(plain, film)
        results = film_agg.paired_tests_per_source(plain, film, pairs, FILM_ARM)
        self.assertEqual(len(results), 3 * 2 * 2)
        for item in results:
            self.assertEqual(item.result.arm_b, FILM_ARM)
            self.assertEqual(item.result.arm_a, PLAIN_ARM)
            self.assertEqual(item.result.image_count, 4)
            self.assertAlmostEqual(item.result.mean_difference, 0.0)
            self.assertEqual(item.result.p_value, 1.0)
            self.assertFalse(item.result.significant)

    def test_a_shift_on_one_source_shows_only_in_that_source(self) -> None:
        self.write_plain_grid()
        self.write_film_grid(shift=(Domain.REFUGE_ZEISS, 0.05))
        plain, film = self.load_both()
        pairs = film_agg.paired_targets(plain, film)
        results = film_agg.paired_tests_per_source(plain, film, pairs, FILM_ARM)
        for item in results:
            if item.source_domain == Domain.REFUGE_ZEISS:
                self.assertAlmostEqual(item.result.mean_difference, 0.05)
                self.assertLess(item.result.p_value, 1.0)
            else:
                self.assertAlmostEqual(item.result.mean_difference, 0.0)
                self.assertEqual(item.result.p_value, 1.0)
        # one Holm family over every cell, not one per source
        raw = [i.result.p_value for i in results]
        from aggregate_stage3_fixed import holm_adjust

        self.assertEqual([i.result.p_adjusted for i in results], list(holm_adjust(raw)))

    def test_one_seed_pairs_with_the_plain_grid_s_matching_seed(self) -> None:
        self.write_plain_grid()
        self.write_film_grid(seeds=(42,))
        plain, film = self.load_both(seeds=(42,))
        self.assertEqual({r.identity.run_seed for r in plain}, {42})
        pairs = film_agg.paired_targets(plain, film)
        cells = film_agg.build_arm_cells(film, pairs)
        self.assertTrue(all(c.std is None for c in cells))
        self.assertEqual(len(film_agg.paired_tests_per_source(plain, film, pairs, FILM_ARM)), 12)


class DegeneracyTests(AggregationFixture):
    def test_cells_confirm_one_code_per_source(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        _, film = self.load_both()
        cells = film_agg.build_degeneracy_cells(film)
        self.assertEqual([c.source_domain for c in cells], list(ACTIVE))
        for cell in cells:
            self.assertEqual(cell.vocabulary, (cell.source_domain.value,))
            self.assertEqual(cell.selector_val_accuracy, 1.0)
            self.assertEqual(cell.fixed_code_entries, 1)
            self.assertEqual(set(cell.targets_assigned_to_source.values()), {1.0})

    def test_an_unstamped_run_is_refused(self) -> None:
        self.write_plain_grid()
        self.write_film_grid(stamp_degenerate=False)
        _, film = self.load_both()
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            film_agg.build_degeneracy_cells(film)
        self.assertIn("degenerate_conditioning", str(caught.exception))

    def test_a_second_code_is_refused(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        run_dir = self.runs_root / f"{FILM_ARM}_drishti_gs_seed_42"
        payload = json.loads((run_dir / "test_metrics.json").read_text())
        payload["conditioning"]["vocabulary"] = ["drishti_gs", "refuge_zeiss"]
        (run_dir / "test_metrics.json").write_text(json.dumps(payload))
        _, film = self.load_both()
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            film_agg.build_degeneracy_cells(film)
        self.assertIn("exactly one code", str(caught.exception))


class CliTests(AggregationFixture):
    def test_end_to_end_writes_report_and_csv(self) -> None:
        self.write_plain_grid()
        self.write_film_grid()
        report = self.directory / "report.md"
        table = self.directory / "table.csv"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = film_agg.main([
                "--run-root", str(self.runs_root),
                "--manifest", str(self.manifest_path),
                "--expected-seeds", *map(str, SEEDS),
                "--report-out", str(report),
                "--csv-out", str(table),
            ])
        self.assertEqual(code, 0)
        self.assertIn("paired tests: 12", stdout.getvalue())
        text = report.read_text()
        self.assertIn("expected result is no difference", text)
        self.assertIn("| `drishti_gs` | `refuge_canon_val` | disc |", text)
        self.assertNotIn("rim_one_dl", text.split("## 3.")[1].split("## 4.")[0])
        with table.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(sum(r["kind"] == "paired_test" for r in rows), 12)
        self.assertEqual(sum(r["kind"] == "degeneracy" for r in rows), 3)
        self.assertTrue(all(r["significant"] == "False" for r in rows if r["kind"] == "paired_test"))

    def test_missing_film_arm_is_a_clean_failure(self) -> None:
        self.write_plain_grid()
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = film_agg.main([
                "--run-root", str(self.runs_root),
                "--manifest", str(self.manifest_path),
                "--expected-seeds", *map(str, SEEDS),
            ])
        self.assertEqual(code, 2)
        self.assertIn("FATAL", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
