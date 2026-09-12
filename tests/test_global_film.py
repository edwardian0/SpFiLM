"""Contract tests for Step 4: Global FiLM and nearest-source-domain conditioning.

Three things are pinned. The layer is the draft's channel-wise FiLM and nothing
more (identity at gamma = beta = 0, one scale and shift per channel, clamped).
The conditioned U-Net is the plain U-Net plus that layer, so the plain arm is
untouched and an old plain resume file still matches. And the test-time rule
is the one agreed with the supervisor: a held-out image gets the code of the
source domain whose training descriptor it is nearest to, chosen per image from
the image alone, and never a code the model was not trained with.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_stage3_fixed import (  # noqa: E402
    FixedLodoReportError,
    Substrate,
    discover_fixed_runs,
    paired_tests,
    select_fixed_runs,
)
from run_stage3_lodo_3_1_fixed import CONFIG_DOMAINS_KEY, CONFIG_STAGE  # noqa: E402
from spfilm.engine import (  # noqa: E402
    FILM_CONFIG_FIELDS,
    Stage2Config,
    _resume_fingerprint,
    evaluate,
    selector_confusion,
    summarise_conditioning,
)
from spfilm.film.conditioning import (  # noqa: E402
    DESCRIPTOR_NAMES,
    ConditioningError,
    DomainVocabulary,
    FixedCondition,
    NearestCondition,
    NearestDomainSelector,
    OracleCondition,
    fov_descriptor,
)
from spfilm.film.global_film import DomainOneHot, GlobalFiLM  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.losses import BCEDiceLoss  # noqa: E402
from spfilm.model import ConditionedUNet, PlainUNet, build_model  # noqa: E402
from spfilm.stage3 import Stage3ConfigError  # noqa: E402
from spfilm.stage3_single_source import Stage3SingleSourceConfig  # noqa: E402

import run_stage3_lodo_1_3  # noqa: E402

PLAIN_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo_fixed.json"
FILM_CONFIG = PROJECT_ROOT / "configs" / "stage4_global_film.json"
PLAIN_CREATE = PROJECT_ROOT / "configs" / "stage3_lodo_fixed_create.json"
FILM_CREATE = PROJECT_ROOT / "configs" / "stage4_global_film_create.json"


def _zero_generator_output(layer: GlobalFiLM) -> None:
    torch.nn.init.zeros_(layer.generator[-1].weight)
    torch.nn.init.zeros_(layer.generator[-1].bias)


# --------------------------------------------------------------------------
# The layer
# --------------------------------------------------------------------------


class GlobalFiLMTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.one_hot = DomainOneHot(64)
        self.layer = GlobalFiLM(8)
        self.features = torch.rand(2, 8, 5, 7)

    def test_output_keeps_the_feature_shape(self) -> None:
        out = self.layer(self.features, self.one_hot(torch.tensor([0, 1])))
        self.assertEqual(tuple(out.shape), (2, 8, 5, 7))

    def test_gamma_and_beta_zero_is_the_identity(self) -> None:
        _zero_generator_output(self.layer)
        out = self.layer(self.features, self.one_hot(torch.tensor([0, 1])))
        self.assertTrue(torch.allclose(out, self.features))

    def test_scale_and_shift_are_per_channel_and_uniform_over_pixels(self) -> None:
        gamma, beta = self.layer.gamma_beta(self.one_hot(torch.tensor([2, 2])))
        out = self.layer(self.features, self.one_hot(torch.tensor([2, 2])))
        expected = (1 + gamma)[:, :, None, None] * self.features + beta[:, :, None, None]
        self.assertTrue(torch.allclose(out, expected))
        self.assertEqual(tuple(gamma.shape), (2, 8))

    def test_gamma_and_beta_are_clamped(self) -> None:
        layer = GlobalFiLM(4, clamp=0.5)
        with torch.no_grad():
            layer.generator[-1].bias.fill_(100.0)
        gamma, beta = layer.gamma_beta(self.one_hot(torch.tensor([0])))
        self.assertTrue(torch.all(gamma <= 0.5) and torch.all(beta <= 0.5))

    def test_different_codes_give_different_modulation(self) -> None:
        gamma_a, _ = self.layer.gamma_beta(self.one_hot(torch.tensor([0])))
        gamma_b, _ = self.layer.gamma_beta(self.one_hot(torch.tensor([1])))
        self.assertFalse(torch.allclose(gamma_a, gamma_b))

    def test_one_hot_is_frozen_basis_vectors(self) -> None:
        codes = self.one_hot(torch.tensor([0, 3]))
        self.assertEqual(codes[0].argmax().item(), 0)
        self.assertEqual(codes[1].argmax().item(), 3)
        self.assertEqual(codes.sum().item(), 2.0)
        self.assertEqual(sum(p.numel() for p in self.one_hot.parameters()), 0)

    def test_one_hot_refuses_out_of_range_codes(self) -> None:
        with self.assertRaises(ValueError):
            DomainOneHot(4)(torch.tensor([4]))


# --------------------------------------------------------------------------
# The conditioned U-Net
# --------------------------------------------------------------------------


class ConditionedUNetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ConditionedUNet(num_domains=3, base_channels=8)
        self.inputs = torch.rand(2, 3, 64, 64)

    def test_preserves_spatial_shape_and_emits_two_channels(self) -> None:
        with torch.inference_mode():
            out = self.model(self.inputs, torch.tensor([0, 2]))
        self.assertEqual(tuple(out.shape), (2, 2, 64, 64))

    def test_with_zero_modulation_it_is_exactly_the_backbone(self) -> None:
        for film in self.model.films:
            _zero_generator_output(film)
        with torch.inference_mode():
            conditioned = self.model(self.inputs, torch.tensor([1, 1]))
            plain = self.model.backbone(self.inputs)
        self.assertTrue(torch.allclose(conditioned, plain, atol=1e-6))

    def test_parameter_count_is_backbone_plus_film_generators(self) -> None:
        backbone = sum(p.numel() for p in self.model.backbone.parameters())
        films = sum(p.numel() for p in self.model.films.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total, backbone + films)
        self.assertEqual(backbone, sum(p.numel() for p in PlainUNet(base_channels=8).parameters()))

    def test_backbone_state_dict_keys_are_the_plain_keys(self) -> None:
        plain_keys = set(PlainUNet(base_channels=8).state_dict())
        nested = {k.removeprefix("backbone.") for k in self.model.state_dict() if k.startswith("backbone.")}
        self.assertEqual(nested, plain_keys)

    def test_film_levels_limits_which_encoder_levels_are_conditioned(self) -> None:
        model = ConditionedUNet(num_domains=2, base_channels=8, film_levels=1)
        self.assertEqual(len(model.films), 1)
        self.assertEqual(model.films[0].num_channels, 8)
        with torch.inference_mode():
            self.assertEqual(tuple(model(self.inputs, torch.tensor([0, 1])).shape), (2, 2, 64, 64))

    def test_untrained_code_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.model(self.inputs, torch.tensor([0, 3]))

    def test_build_model_maps_arms(self) -> None:
        self.assertIsInstance(build_model("plain", 8), PlainUNet)
        self.assertIsInstance(build_model("global_film", 8, num_domains=2), ConditionedUNet)
        with self.assertRaises(ValueError):
            build_model("global_film", 8)
        with self.assertRaises(ValueError):
            build_model("spatial", 8)


# --------------------------------------------------------------------------
# Descriptor, selector, and condition providers
# --------------------------------------------------------------------------


def _images(level: float, count: int = 4, size: int = 16) -> torch.Tensor:
    torch.manual_seed(int(level * 1000))
    return (torch.rand(count, 3, size, size) * 0.2 + level).clamp(0, 1)


class DescriptorTests(unittest.TestCase):
    def test_shape_and_order(self) -> None:
        self.assertEqual(tuple(fov_descriptor(_images(0.5)).shape), (4, len(DESCRIPTOR_NAMES)))

    def test_black_letterbox_border_does_not_change_the_descriptor(self) -> None:
        images = _images(0.5, count=1)
        bordered = torch.zeros(1, 3, 32, 32)
        bordered[:, :, 8:24, 8:24] = images
        self.assertTrue(torch.allclose(fov_descriptor(images), fov_descriptor(bordered), atol=1e-6))

    def test_constant_image_has_zero_spread(self) -> None:
        images = torch.full((1, 3, 8, 8), 0.4)
        descriptor = fov_descriptor(images)
        self.assertTrue(torch.allclose(descriptor[0, :3], torch.full((3,), 0.4)))
        self.assertTrue(torch.allclose(descriptor[0, 3:], torch.zeros(3), atol=1e-6))

    def test_all_black_image_falls_back_to_every_pixel_instead_of_nan(self) -> None:
        descriptor = fov_descriptor(torch.zeros(1, 3, 8, 8))
        self.assertTrue(torch.isfinite(descriptor).all())


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = DomainVocabulary.from_domains(["dark", "bright"])
        descriptors = torch.cat([fov_descriptor(_images(0.2)), fov_descriptor(_images(0.7))])
        self.selector = NearestDomainSelector.fit(
            descriptors, ["dark"] * 4 + ["bright"] * 4, self.vocabulary
        )

    def test_vocabulary_is_sorted_and_codes_are_stable(self) -> None:
        self.assertEqual(self.vocabulary.domains, ("bright", "dark"))
        self.assertEqual(self.vocabulary.index_of("dark"), 1)
        with self.assertRaises(ValueError):
            DomainVocabulary(("dark", "bright"))

    def test_nearest_domain_recovers_the_synthetic_domains(self) -> None:
        indices, distances, descriptors = self.selector.select(
            torch.cat([_images(0.25, count=2), _images(0.65, count=2)])
        )
        self.assertEqual(indices.tolist(), [1, 1, 0, 0])
        self.assertEqual(tuple(distances.shape), (4, 2))
        self.assertEqual(tuple(descriptors.shape), (4, len(DESCRIPTOR_NAMES)))

    def test_distances_are_standardised_by_the_within_domain_spread(self) -> None:
        within = (self.selector.spreads ** 2).mean(dim=0).sqrt()  # equal counts
        self.assertTrue(torch.allclose(self.selector.scale, within, atol=1e-6))
        standardised = (self.selector.centroids - self.selector.shift) / self.selector.scale
        distances = self.selector.distances(self.selector.centroids)
        self.assertTrue(torch.allclose(distances, torch.cdist(standardised, standardised), atol=1e-4))
        self.assertTrue(torch.allclose(distances.diagonal(), torch.zeros(2), atol=1e-4))

    def test_a_noisy_uninformative_dimension_does_not_outvote_the_signal(self) -> None:
        """The failure mode total-spread z-scoring has: the descriptors' std
        dimensions barely differ between these synthetic domains, so scaling
        them to unit variance would let their noise decide the argmin."""

        indices, _, _ = self.selector.select(
            torch.cat([_images(0.3, count=3), _images(0.6, count=3)])
        )
        self.assertEqual(indices.tolist(), [1, 1, 1, 0, 0, 0])

    def test_json_round_trip(self) -> None:
        restored = NearestDomainSelector.from_json(json.loads(json.dumps(self.selector.to_json())))
        self.assertEqual(restored.vocabulary, self.vocabulary)
        self.assertTrue(torch.allclose(restored.centroids, self.selector.centroids))
        self.assertTrue(torch.allclose(restored.scale, self.selector.scale))
        self.assertEqual(restored.fitted_counts, self.selector.fitted_counts)

    def test_every_vocabulary_domain_must_have_training_images(self) -> None:
        with self.assertRaises(ValueError):
            NearestDomainSelector.fit(
                fov_descriptor(_images(0.2)), ["dark"] * 4, self.vocabulary
            )

    def test_fit_from_loader_uses_the_batch_metadata_domains(self) -> None:
        class _Loader:
            def __iter__(self):
                yield _images(0.2), None, {"domain": ["dark"] * 4}
                yield _images(0.7), None, {"domain": ["bright"] * 4}

        selector = NearestDomainSelector.fit_from_loader(_Loader(), self.vocabulary)
        self.assertEqual(selector.fitted_counts, (4, 4))


class ConditionProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = DomainVocabulary.from_domains(["a", "b"])
        self.images = torch.rand(3, 3, 8, 8)

    def test_oracle_uses_the_true_domain(self) -> None:
        result = OracleCondition(self.vocabulary)(self.images, {"domain": ["b", "a", "b"]})
        self.assertEqual(result.indices.tolist(), [1, 0, 1])
        self.assertEqual(result.source, "oracle")
        self.assertIsNone(result.distances)

    def test_oracle_refuses_a_domain_outside_the_vocabulary(self) -> None:
        with self.assertRaises(ConditioningError):
            OracleCondition(self.vocabulary)(self.images, {"domain": ["a", "held_out", "b"]})

    def test_fixed_code_fills_the_batch(self) -> None:
        result = FixedCondition(self.vocabulary, "b")(self.images, {"domain": ["x"] * 3})
        self.assertEqual(result.indices.tolist(), [1, 1, 1])
        self.assertEqual(result.source, "fixed:b")
        with self.assertRaises(ConditioningError):
            FixedCondition(self.vocabulary, "held_out")

    def test_nearest_ignores_the_metadata_domain(self) -> None:
        descriptors = torch.cat([fov_descriptor(_images(0.2)), fov_descriptor(_images(0.7))])
        selector = NearestDomainSelector.fit(descriptors, ["a"] * 4 + ["b"] * 4, self.vocabulary)
        result = NearestCondition(selector)(_images(0.7, count=2), {"domain": ["unseen", "unseen"]})
        self.assertEqual(result.indices.tolist(), [1, 1])
        self.assertEqual(result.source, "nearest_domain")
        self.assertIsNotNone(result.distances)


# --------------------------------------------------------------------------
# Engine: fingerprint, evaluation records, and summaries
# --------------------------------------------------------------------------


def _config(**overrides) -> Stage2Config:
    values = {
        "experiment_name": "film_test",
        "dataset": "refuge",
        "data_root": "datasets/REFUGE",
        "output_dir": "artifacts/film_test",
        "seed": 42,
        "epochs": 10,
    }
    values.update(overrides)
    return Stage2Config(**values)


class FingerprintTests(unittest.TestCase):
    counts = {"train": 120, "val": 30, "test": 50}

    def test_plain_fingerprint_is_the_pre_conditioning_hash(self) -> None:
        """A plain run preempted under the old code must still resume."""

        legacy = asdict(_config())
        for field in FILM_CONFIG_FIELDS:
            legacy.pop(field)
        expected = hashlib.sha256(
            json.dumps({"config": legacy, "split_counts": self.counts}, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(_resume_fingerprint(_config(), self.counts), expected)

    def test_film_settings_do_not_touch_the_plain_fingerprint(self) -> None:
        self.assertEqual(
            _resume_fingerprint(_config(), self.counts),
            _resume_fingerprint(_config(film_levels=1, film_clamp=2.0), self.counts),
        )

    def test_the_arm_and_its_settings_are_part_of_a_film_fingerprint(self) -> None:
        plain = _resume_fingerprint(_config(), self.counts)
        film = _resume_fingerprint(_config(arm="global_film"), self.counts)
        shallower = _resume_fingerprint(_config(arm="global_film", film_levels=1), self.counts)
        self.assertNotEqual(plain, film)
        self.assertNotEqual(film, shallower)


class _TwoDomainDataset(Dataset):
    def __init__(self, domains: list[str]) -> None:
        self.domains = domains

    def __len__(self) -> int:
        return len(self.domains)

    def __getitem__(self, index: int):
        level = 0.2 if self.domains[index] == "dark" else 0.7
        image = _images(level, count=1, size=8)[0]
        return image, torch.zeros(2, 8, 8), {
            "sample_id": f"img{index}",
            "domain": self.domains[index],
            "letterbox_scale": 1.0,
        }


class _CodeEchoModel(torch.nn.Module):
    """Predicts nothing; records the codes it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[int]] = []

    def forward(self, images: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor:
        self.seen.append(domain_index.tolist())
        return torch.full((images.shape[0], 2, 8, 8), -10.0)


class EvaluateConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = DomainVocabulary.from_domains(["dark", "bright"])
        descriptors = torch.cat([fov_descriptor(_images(0.2)), fov_descriptor(_images(0.7))])
        self.selector = NearestDomainSelector.fit(
            descriptors, ["dark"] * 4 + ["bright"] * 4, self.vocabulary
        )
        self.loader = DataLoader(_TwoDomainDataset(["dark", "bright", "bright"]), batch_size=2)
        self.model = _CodeEchoModel()

    def test_records_one_row_per_image_with_the_code_used(self) -> None:
        metrics = evaluate(
            self.model,
            self.loader,
            BCEDiceLoss(),
            torch.device("cpu"),
            threshold=0.5,
            condition_fn=NearestCondition(self.selector),
        )
        rows = metrics["conditioning"]["rows"]
        self.assertEqual([r["image_id"] for r in rows], ["img0", "img1", "img2"])
        self.assertEqual([r["selected_domain"] for r in rows], ["dark", "bright", "bright"])
        self.assertEqual({r["condition_source"] for r in rows}, {"nearest_domain"})
        self.assertEqual(sum(self.model.seen, []), [1, 0, 0])
        for row in rows:
            self.assertIsInstance(row["distance_dark"], float)
            self.assertIsInstance(row["red_mean"], float)

    def test_summary_counts_assignments_and_flags_unseen_true_domains(self) -> None:
        rows = [
            {"image_id": "a", "true_domain": "held_out", "selected_domain": "dark",
             "distance_bright": 3.0, "distance_dark": 1.0},
            {"image_id": "b", "true_domain": "held_out", "selected_domain": "bright",
             "distance_bright": 0.5, "distance_dark": 2.0},
        ]
        summary = summarise_conditioning(rows, self.vocabulary)
        self.assertEqual(summary["assignment_counts"], {"bright": 1, "dark": 1})
        self.assertFalse(summary["true_domains_in_vocabulary"])
        self.assertAlmostEqual(summary["mean_nearest_distance"], 0.75)

    def test_selector_confusion_is_only_defined_on_vocabulary_domains(self) -> None:
        rows = [
            {"true_domain": "dark", "selected_domain": "dark"},
            {"true_domain": "bright", "selected_domain": "dark"},
        ]
        result = selector_confusion(rows, self.vocabulary)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["confusion"]["bright"]["dark"], 1)
        with self.assertRaises(ConditioningError):
            selector_confusion([{"true_domain": "held_out", "selected_domain": "dark"}], self.vocabulary)


# --------------------------------------------------------------------------
# Configs and runners
# --------------------------------------------------------------------------


def _load(path: Path) -> Stage3SingleSourceConfig:
    return Stage3SingleSourceConfig.from_json(
        path, expected_stage=CONFIG_STAGE, domains_key=CONFIG_DOMAINS_KEY
    )


def _write_temp_config(payload: dict) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
        return Path(stream.name)


@unittest.skipUnless(PLAIN_CONFIG.is_file() and FILM_CONFIG.is_file(), "configs not present")
class Stage4ConfigTests(unittest.TestCase):
    FILM_ONLY_KEYS = {"arm", "film", "experiment_name", "output_dir"}
    PROTOCOL_ONLY_KEYS = {"policy", "paired_arm", "pairing_rationale"}

    def test_film_configs_load_with_the_global_film_arm(self) -> None:
        for path in (FILM_CONFIG, FILM_CREATE):
            config = _load(path)
            self.assertEqual(config.arm, "global_film")
            self.assertEqual(config.experiment_name, "stage4_lodo_fixed_budget_global_film")
            self.assertEqual(config.paired_arm, "stage3_lodo_fixed_budget_plain_unet")
            self.assertEqual(config.film_levels, 5)
            self.assertEqual(config.test_conditioning, "nearest_domain")

    def test_film_configs_differ_from_plain_only_in_the_conditioning(self) -> None:
        """Change one thing: everything but the arm must be byte-for-byte the plain protocol."""

        for plain_path, film_path in ((PLAIN_CONFIG, FILM_CONFIG), (PLAIN_CREATE, FILM_CREATE)):
            plain = json.loads(plain_path.read_text())
            film = json.loads(film_path.read_text())
            for key in self.FILM_ONLY_KEYS:
                plain.pop(key, None)
                film.pop(key, None)
            for key in self.PROTOCOL_ONLY_KEYS:
                plain["protocol"].pop(key, None)
                film["protocol"].pop(key, None)
            self.assertEqual(plain, film, f"{film_path.name} drifts from {plain_path.name}")

    def test_training_config_forwards_the_film_settings(self) -> None:
        engine_config = _load(FILM_CONFIG).training_config(Domain.REFUGE_ZEISS, 42, "artifacts/x")
        self.assertEqual(engine_config.arm, "global_film")
        self.assertEqual(engine_config.film_levels, 5)
        self.assertEqual(engine_config.film_clamp, 5.0)
        plain_config = _load(PLAIN_CONFIG).training_config(Domain.REFUGE_ZEISS, 42, "artifacts/x")
        self.assertEqual(plain_config.arm, "plain")

    def test_unknown_arm_is_refused(self) -> None:
        payload = json.loads(FILM_CONFIG.read_text())
        payload["arm"] = "spatial_film"
        with self.assertRaises(Stage3ConfigError):
            _load(_write_temp_config(payload))

    def test_plain_arm_must_not_carry_a_film_block(self) -> None:
        payload = json.loads(PLAIN_CONFIG.read_text())
        payload["film"] = {"levels": 1}
        with self.assertRaises(Stage3ConfigError):
            _load(_write_temp_config(payload))

    def test_film_block_is_validated(self) -> None:
        for bad in ({"levels": 6}, {"clamp": 0}, {"test_conditioning": "guess"}, {"rank": 8}):
            payload = json.loads(FILM_CONFIG.read_text())
            payload["film"] = bad
            with self.assertRaises(Stage3ConfigError, msg=str(bad)):
                _load(_write_temp_config(payload))

    def test_train_on_one_runner_refuses_a_conditioned_arm(self) -> None:
        payload = json.loads(FILM_CONFIG.read_text())
        payload["stage"] = "single_source"
        payload["protocol"]["source_domains"] = payload["protocol"].pop(CONFIG_DOMAINS_KEY)
        path = _write_temp_config(payload)
        self.assertEqual(
            run_stage3_lodo_1_3.main(["--config", str(path), "check", "--skip-mask-audit"]),
            2,
        )


# --------------------------------------------------------------------------
# Aggregation: two arms in one run root, FiLM as the reference
# --------------------------------------------------------------------------


def _write_run(base: Path, arm: str, domain: Domain, seed: int, dice: float) -> Path:
    run = base / f"{arm}_{domain.value}_seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    rows = [
        {"image_id": f"img{i}", "structure": s, "dice": f"{dice + i * 0.001:.4f}",
         "iou": "0.6", "hd95": "10", "acc": "0.9", "tp": 1, "fp": 1, "fn": 1, "tn": 1}
        for i in range(5) for s in ("disc", "cup")
    ]
    with (run / "test_per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run / "test_metrics.json").write_text(json.dumps({
        "test": {"evaluated_sample_count": 5},
        "fixed_lodo": {
            "protocol": "leave_one_domain_out_fixed_budget", "arm": arm,
            "held_out_domain": domain.value, "source_domains": [], "run_seed": seed,
            "budget": {"train": 40, "val": 10, "test": 50, "subsample_seed": 42},
            "manifest_sha256": "a" * 64, "completed_at_utc": f"2026-09-12T00:0{seed - 42}:00+00:00",
            "smoke_rehearsal": False, "scientific_result": True,
        },
    }))
    return run


class TwoArmAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        for seed in (42, 43):
            _write_run(self.directory, "plain", Domain.DRISHTI_GS, seed, 0.70)
            _write_run(self.directory, "film", Domain.DRISHTI_GS, seed, 0.75)

    def test_mixed_arms_are_refused_without_an_arm_filter(self) -> None:
        with self.assertRaises(FixedLodoReportError) as caught:
            select_fixed_runs(discover_fixed_runs([self.directory]), (42, 43))
        self.assertIn("--arm", str(caught.exception))

    def test_arm_filter_selects_one_arm(self) -> None:
        runs = select_fixed_runs(discover_fixed_runs([self.directory]), (42, 43), arm="film")
        self.assertEqual({r.arm for r in runs}, {"film"})
        self.assertEqual(len(runs), 2)
        with self.assertRaises(FixedLodoReportError):
            select_fixed_runs(discover_fixed_runs([self.directory]), (42, 43), arm="absent")

    def test_paired_test_uses_the_requested_reference_arm(self) -> None:
        values = {}
        for i in range(30):
            values[("plain", Domain.DRISHTI_GS, "disc", f"img{i}")] = {"dice": 0.5 + i * 0.001, "iou": 0.5}
            values[("film", Domain.DRISHTI_GS, "disc", f"img{i}")] = {"dice": 0.52 + i * 0.001, "iou": 0.5}
        substrate = Substrate(
            seed_counts={("plain", Domain.DRISHTI_GS): 2, ("film", Domain.DRISHTI_GS): 2},
            values=values,
        )
        (result,) = paired_tests(substrate, reference_arm="film")
        self.assertEqual((result.arm_a, result.arm_b), ("plain", "film"))
        self.assertAlmostEqual(result.mean_difference, 0.02)
        self.assertTrue(result.significant)


if __name__ == "__main__":
    unittest.main()
