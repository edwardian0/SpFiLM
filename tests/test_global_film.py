"""Contract tests for Step 4: Global FiLM and nearest-source-domain conditioning.

Three things are pinned. The layer is the draft's channel-wise FiLM and nothing
more (identity at gamma = beta = 0, one scale and shift per channel, clamped).
The conditioned U-Net is the plain U-Net plus that layer, so the plain arm is
untouched and an old plain resume file still matches. And the test-time rule
is the one agreed with the supervisor: the whole held-out domain gets one code,
that of the source domain whose training descriptor centroid is nearest to its
unlabelled reference sample (the per-image rule is kept as an ablation), and
never a code the model was not trained with.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
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

import aggregate_stage4_film  # noqa: E402
from aggregate_stage3_fixed import (  # noqa: E402
    FixedLodoReportError,
    Substrate,
    build_domain_cells,
    discover_fixed_runs,
    paired_tests,
    select_fixed_runs,
)
from run_stage3_lodo_3_1_fixed import (  # noqa: E402
    CONFIG_DOMAINS_KEY,
    CONFIG_STAGE,
    fixed_lodo_folds,
    held_out_reference_keys,
)
from spfilm.lodo import DomainPartitions, SampleKey  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    write_single_source_manifest,
)
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
    DomainCondition,
    DomainDecision,
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


STAGE3_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo_fixed.json"
PLAIN_CONFIG = PROJECT_ROOT / "configs" / "stage4_plain_3dom.json"
FILM_CONFIG = PROJECT_ROOT / "configs" / "stage4_global_film_3dom.json"
PLAIN_CREATE = PROJECT_ROOT / "configs" / "stage4_plain_3dom_create.json"
FILM_CREATE = PROJECT_ROOT / "configs" / "stage4_global_film_3dom_create.json"
STEP4_ACTIVE = {Domain.REFUGE_ZEISS, Domain.REFUGE_CANON_VAL, Domain.DRISHTI_GS}


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

    def test_modulation_runs_in_float32_under_autocast(self) -> None:
        """Matches the reference implementation: half precision must not reach the affine."""

        embedding = self.one_hot(torch.tensor([0, 1]))
        reference = self.layer(self.features, embedding)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            under_autocast = self.layer(self.features, embedding)
        self.assertEqual(under_autocast.dtype, self.features.dtype)
        self.assertTrue(torch.allclose(under_autocast, reference, atol=1e-6))

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

    def test_domain_decision_averages_before_comparing(self) -> None:
        """Option B: one decision for the whole domain from its mean descriptor."""

        mixed = torch.cat([fov_descriptor(_images(0.6, count=5)), fov_descriptor(_images(0.25, count=1))])
        index, distances, centroid = self.selector.select_domain(mixed)
        self.assertEqual(self.vocabulary.domains[index], "bright")
        self.assertEqual(tuple(distances.shape), (2,))
        self.assertTrue(torch.allclose(centroid, mixed.mean(dim=0)))
        with self.assertRaises(ValueError):
            self.selector.select_domain(torch.zeros(0, len(DESCRIPTOR_NAMES)))

    def test_domain_decision_from_loader_records_its_evidence(self) -> None:
        class _Loader:
            def __iter__(self):
                yield _images(0.65, count=3), None, {"domain": ["unseen"] * 3, "sample_id": ["a", "b", "c"]}
                yield _images(0.7, count=2), None, {"domain": ["unseen"] * 2, "sample_id": ["d", "e"]}

        decision = self.selector.select_domain_from_loader(_Loader())
        self.assertEqual(decision.held_out_domain, "unseen")
        self.assertEqual(decision.chosen_domain, "bright")
        self.assertEqual(decision.chosen_index, self.vocabulary.index_of("bright"))
        self.assertEqual(decision.reference_sample_ids, ("a", "b", "c", "d", "e"))
        self.assertLess(decision.distances["bright"], decision.distances["dark"])
        payload = json.loads(json.dumps(decision.to_json()))
        self.assertEqual(payload["reference_image_count"], 5)

    def test_domain_decision_refuses_a_mixed_reference_sample(self) -> None:
        class _Loader:
            def __iter__(self):
                yield _images(0.65, count=2), None, {"domain": ["x", "y"], "sample_id": ["a", "b"]}

        with self.assertRaises(ConditioningError):
            self.selector.select_domain_from_loader(_Loader())

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

    def test_nearest_image_ignores_the_metadata_domain(self) -> None:
        descriptors = torch.cat([fov_descriptor(_images(0.2)), fov_descriptor(_images(0.7))])
        selector = NearestDomainSelector.fit(descriptors, ["a"] * 4 + ["b"] * 4, self.vocabulary)
        result = NearestCondition(selector)(_images(0.7, count=2), {"domain": ["unseen", "unseen"]})
        self.assertEqual(result.indices.tolist(), [1, 1])
        self.assertEqual(result.source, "nearest_image")
        self.assertIsNotNone(result.distances)

    def test_domain_condition_uses_one_code_but_logs_the_per_image_view(self) -> None:
        descriptors = torch.cat([fov_descriptor(_images(0.2)), fov_descriptor(_images(0.7))])
        selector = NearestDomainSelector.fit(descriptors, ["a"] * 4 + ["b"] * 4, self.vocabulary)
        decision = DomainDecision(
            held_out_domain="unseen", chosen_domain="a", chosen_index=0,
            distances={"a": 0.1, "b": 2.0}, reference_centroid={}, reference_sample_ids=("r",),
        )
        condition = DomainCondition(selector, decision)
        # bright images would individually map to "b", but the domain code is "a"
        result = condition(_images(0.7, count=2), {"domain": ["unseen", "unseen"]})
        self.assertEqual(result.indices.tolist(), [0, 0])
        self.assertEqual(result.source, "nearest_domain")
        self.assertEqual(result.distances.argmin(dim=1).tolist(), [1, 1])
        with self.assertRaises(ConditioningError):
            DomainCondition(selector, DomainDecision(
                held_out_domain="unseen", chosen_domain="b", chosen_index=0,
                distances={}, reference_centroid={}, reference_sample_ids=(),
            ))


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
        self.assertEqual([r["nearest_image_domain"] for r in rows], ["dark", "bright", "bright"])
        self.assertEqual({r["condition_source"] for r in rows}, {"nearest_image"})
        self.assertEqual(sum(self.model.seen, []), [1, 0, 0])
        for row in rows:
            self.assertIsInstance(row["distance_dark"], float)
            self.assertIsInstance(row["red_mean"], float)

    def test_per_domain_code_is_used_for_every_image_and_per_image_view_is_kept(self) -> None:
        decision = DomainDecision(
            held_out_domain="held_out", chosen_domain="dark", chosen_index=1,
            distances={"bright": 2.0, "dark": 0.5}, reference_centroid={}, reference_sample_ids=("r",),
        )
        metrics = evaluate(
            self.model,
            self.loader,
            BCEDiceLoss(),
            torch.device("cpu"),
            threshold=0.5,
            condition_fn=DomainCondition(self.selector, decision),
        )
        rows = metrics["conditioning"]["rows"]
        self.assertEqual({r["selected_domain"] for r in rows}, {"dark"})
        self.assertEqual([r["nearest_image_domain"] for r in rows], ["dark", "bright", "bright"])
        self.assertEqual(sum(self.model.seen, []), [1, 1, 1])
        summary = summarise_conditioning(rows, self.vocabulary)
        self.assertEqual(summary["assignment_counts"], {"bright": 0, "dark": 3})
        self.assertEqual(summary["nearest_image_counts"], {"bright": 2, "dark": 1})

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
    ARM_ONLY_KEYS = {"arm", "film", "experiment_name", "output_dir"}
    PROTOCOL_ARM_ONLY_KEYS = {"policy", "paired_arm"}

    def test_film_configs_load_with_the_global_film_arm(self) -> None:
        for path in (FILM_CONFIG, FILM_CREATE):
            config = _load(path)
            self.assertEqual(config.arm, "global_film")
            self.assertEqual(config.experiment_name, "stage4_lodo_fixed_budget_global_film_3dom")
            self.assertEqual(config.paired_arm, "stage4_lodo_fixed_budget_plain_unet_3dom")
            self.assertEqual(config.film_levels, 5)
            self.assertEqual(config.test_conditioning, "nearest_domain")

    def test_plain_configs_load_and_point_back_at_the_film_arm(self) -> None:
        for path in (PLAIN_CONFIG, PLAIN_CREATE):
            config = _load(path)
            self.assertEqual(config.arm, "plain")
            self.assertEqual(config.experiment_name, "stage4_lodo_fixed_budget_plain_unet_3dom")
            self.assertEqual(config.paired_arm, "stage4_lodo_fixed_budget_global_film_3dom")

    def test_step4_drops_rim_one_from_the_folds_but_keeps_it_configured(self) -> None:
        for path in (PLAIN_CONFIG, FILM_CONFIG, PLAIN_CREATE, FILM_CREATE):
            config = _load(path)
            self.assertEqual(set(config.active_domains), STEP4_ACTIVE, path.name)
            self.assertNotIn(Domain.RIM_ONE_DL, config.held_out_domains)
            # still discovered and validated: the locked manifests cover it
            self.assertEqual({d.domain for d in config.domains}, set(Domain))

    def test_film_configs_differ_from_plain_only_in_the_conditioning(self) -> None:
        """Change one thing: everything but the arm must be byte-for-byte the plain protocol."""

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

    def test_step4_configs_share_the_stage3_hyperparameters(self) -> None:
        """Only the domain set and the arm change relative to the Stage 3 plain arm."""

        if not STAGE3_CONFIG.is_file():
            self.skipTest("stage 3 config not present")
        stage3 = json.loads(STAGE3_CONFIG.read_text())
        step4 = json.loads(PLAIN_CONFIG.read_text())
        for key in ("image_size", "batch_size", "epochs", "patience", "min_epochs",
                    "early_stopping_mode", "learning_rate", "weight_decay", "base_channels",
                    "threshold", "horizontal_flip_probability", "rotation_degrees",
                    "brightness_contrast", "domains"):
            self.assertEqual(stage3[key], step4[key], key)
        self.assertEqual(stage3["protocol"]["budget"], step4["protocol"]["budget"])
        self.assertEqual(stage3["protocol"]["seeds"], step4["protocol"]["seeds"])

    def test_training_config_forwards_the_film_settings(self) -> None:
        engine_config = _load(FILM_CONFIG).training_config(Domain.REFUGE_ZEISS, 42, "artifacts/x")
        self.assertEqual(engine_config.arm, "global_film")
        self.assertEqual(engine_config.film_levels, 5)
        self.assertEqual(engine_config.film_clamp, 5.0)
        plain_config = _load(PLAIN_CONFIG).training_config(Domain.REFUGE_ZEISS, 42, "artifacts/x")
        self.assertEqual(plain_config.arm, "plain")

    def test_legacy_prose_paired_arm_reads_as_unstated(self) -> None:
        if not STAGE3_CONFIG.is_file():
            self.skipTest("stage 3 config not present")
        self.assertIsNone(_load(STAGE3_CONFIG).paired_arm)

    def test_protocol_domain_list_must_be_configured_and_at_least_two(self) -> None:
        payload = json.loads(PLAIN_CONFIG.read_text())
        payload["protocol"][CONFIG_DOMAINS_KEY] = ["refuge_zeiss"]
        with self.assertRaises(Stage3ConfigError):
            _load(_write_temp_config(payload))
        payload = json.loads(PLAIN_CONFIG.read_text())
        payload["domains"].pop("rim_one_dl")
        with self.assertRaises(Stage3ConfigError):
            _load(_write_temp_config(payload))

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

    def test_test_conditioning_policies_are_the_three_named_ones(self) -> None:
        for policy in ("nearest_domain", "nearest_image", "oracle"):
            payload = json.loads(FILM_CONFIG.read_text())
            payload["film"]["test_conditioning"] = policy
            self.assertEqual(_load(_write_temp_config(payload)).test_conditioning, policy)

    def test_film_block_is_validated(self) -> None:
        for bad in ({"levels": 6}, {"clamp": 0}, {"test_conditioning": "guess"}, {"rank": 8}):
            payload = json.loads(FILM_CONFIG.read_text())
            payload["film"] = bad
            with self.assertRaises(Stage3ConfigError, msg=str(bad)):
                _load(_write_temp_config(payload))


def _synthetic_manifest() -> SingleSourceManifest:
    """Four domains with a 4/2/3 budget, shaped like the committed manifest."""

    partitions = []
    strata = {}
    for domain in sorted(Domain, key=lambda item: item.value):
        prefix = domain.value[:3]
        train = tuple(SampleKey(domain, f"{prefix}_tr{n}") for n in range(4))
        val = tuple(SampleKey(domain, f"{prefix}_va{n}") for n in range(2))
        test = tuple(SampleKey(domain, f"{prefix}_te{n}") for n in range(3))
        partitions.append(DomainPartitions(domain=domain, train=train, val=val, test=test))
        for key in train + val + test:
            strata[key] = "all"
    return SingleSourceManifest.build("a" * 64, tuple(partitions), 4, 2, 3, strata, 42)


class ReferenceSampleTests(unittest.TestCase):
    """The held-out domain's reference sample is its budgeted train partition."""

    def setUp(self) -> None:
        self.manifest = _synthetic_manifest()

    def test_reference_is_the_held_out_train_partition_and_disjoint_from_the_fold(self) -> None:
        active = (Domain.DRISHTI_GS, Domain.REFUGE_CANON_VAL, Domain.REFUGE_ZEISS)
        for fold in fixed_lodo_folds(self.manifest, active):
            reference = held_out_reference_keys(self.manifest, fold.held_out_domain)
            self.assertEqual(len(reference), 4)
            self.assertTrue(all(key.domain == fold.held_out_domain for key in reference))
            self.assertTrue(set(reference).isdisjoint(fold.test))
            self.assertTrue(set(reference).isdisjoint(fold.train))
            self.assertTrue(set(reference).isdisjoint(fold.val))


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


PLAIN_ARM = "stage4_lodo_fixed_budget_plain_unet_3dom"
FILM_ARM = "stage4_lodo_fixed_budget_global_film_3dom"


def _write_lodo_run(
    base: Path,
    manifest_sha: str,
    fold,
    arm: str,
    seed: int,
    dice: float,
    source_domains: list[str] | None = None,
) -> Path:
    """One fixed-budget LODO run as the runner writes it, scored on the fold's test IDs.

    A FiLM run also carries the ``conditioning`` keys the report reads, shaped
    as ``engine._conditioning_report`` writes them.
    """

    held_out = fold.held_out_domain
    sources = source_domains or sorted({sample.domain.value for sample in fold.train})
    run = base / f"{arm}_{held_out.value}_seed_{seed}"
    run.mkdir(parents=True)
    rows = [
        {"image_id": sample.sample_id, "structure": structure,
         "dice": f"{dice + 0.01 * index + 0.001 * (seed - 42):.4f}", "iou": "0.6",
         "hd95": "10", "acc": "0.9", "tp": 1, "fp": 1, "fn": 1, "tn": 1}
        for index, sample in enumerate(fold.test) for structure in ("disc", "cup")
    ]
    with (run / "test_per_image_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload: dict = {
        "test": {"evaluated_sample_count": len(fold.test)},
        "fixed_lodo": {
            "protocol": "leave_one_domain_out_fixed_budget", "arm": arm,
            "held_out_domain": held_out.value, "source_domains": sources, "run_seed": seed,
            "budget": {"train": 4, "val": 2, "test": 3, "subsample_seed": 42},
            "manifest_sha256": manifest_sha,
            "completed_at_utc": f"2026-09-23T00:00:{seed - 40:02d}+00:00",
            "smoke_rehearsal": False, "scientific_result": True,
        },
    }
    if arm == FILM_ARM:
        vocabulary = sorted(sources)
        count = len(fold.test)
        payload["conditioning"] = {
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
    (run / "test_metrics.json").write_text(json.dumps(payload))
    return run


class LodoReportTests(unittest.TestCase):
    """aggregate_stage4_film end to end, on a synthetic three-domain grid.

    The first look after launching is seed 42 of each arm, so one seed per arm
    must produce a report (no seed spread yet) rather than stop.
    """

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.manifest = _synthetic_manifest()
        self.manifest_path = self.directory / "single_source_manifest.json"
        write_single_source_manifest(self.manifest, self.manifest_path)
        self.manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.folds = fixed_lodo_folds(self.manifest, tuple(sorted(STEP4_ACTIVE, key=lambda d: d.value)))
        self.runs = self.directory / "runs"

    def write_grid(self, seeds, film_extra_source: str | None = None) -> None:
        for fold in self.folds:
            sources = sorted({sample.domain.value for sample in fold.train})
            film_sources = sorted([*sources, film_extra_source]) if film_extra_source else None
            for seed in seeds:
                _write_lodo_run(self.runs, self.manifest_sha, fold, PLAIN_ARM, seed, 0.80)
                _write_lodo_run(self.runs, self.manifest_sha, fold, FILM_ARM, seed, 0.82,
                                source_domains=film_sources)

    def report(self, *seeds: int) -> tuple[int, str, str, str]:
        report_path = self.directory / "report.md"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = aggregate_stage4_film.main([
                "--run-root", str(self.runs),
                "--manifest", str(self.manifest_path),
                "--expected-seeds", *map(str, seeds),
                "--report-out", str(report_path),
                "--csv-out", str(self.directory / "cells.csv"),
            ])
        report = report_path.read_text() if report_path.is_file() else ""
        return code, stdout.getvalue(), stderr.getvalue(), report

    def test_one_seed_per_arm_reports_without_a_seed_spread(self) -> None:
        self.write_grid((42,))
        code, stdout, stderr, report = self.report(42)
        self.assertEqual(code, 0, stderr)
        self.assertIn("paired tests: 6", stdout)
        self.assertIn("(1 seed)", report)
        self.assertNotIn("± nan", report)
        for fold in self.folds:
            self.assertIn(f"| `{fold.held_out_domain.value}` | disc | 3 |", report)

    def test_two_seeds_report_the_seed_spread(self) -> None:
        self.write_grid((42, 43))
        code, _stdout, stderr, report = self.report(42, 43)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("(1 seed)", report)
        self.assertIn(" ± ", report)

    def test_film_arm_trained_on_other_sources_is_not_paired(self) -> None:
        """A FiLM grid whose folds included RIM-ONE-DL is not the plain 3-domain arm's pair."""

        self.write_grid((42,), film_extra_source="rim_one_dl")
        code, _stdout, stderr, _report = self.report(42)
        self.assertEqual(code, 2)
        self.assertIn("trained on", stderr)

    def test_the_stage3_report_still_refuses_a_single_seed(self) -> None:
        self.write_grid((42,))
        runs = select_fixed_runs(discover_fixed_runs([self.runs]), (42,), arm=PLAIN_ARM)
        with self.assertRaises(ValueError):
            build_domain_cells(runs)
        cells = build_domain_cells(runs, allow_single_seed=True)
        dice = next(c for c in cells if c.structure == "disc").intervals["dice"]
        self.assertEqual(dice.seeds, (42,))
        self.assertAlmostEqual(dice.mean, 0.81)  # 0.80, 0.81, 0.82 over the fold's 3 images
        self.assertTrue(math.isnan(dice.std))

    def test_the_single_seed_flag_is_inert_with_two_seeds(self) -> None:
        self.write_grid((42, 43))
        runs = select_fixed_runs(discover_fixed_runs([self.runs]), (42, 43), arm=PLAIN_ARM)
        self.assertEqual(build_domain_cells(runs), build_domain_cells(runs, allow_single_seed=True))


if __name__ == "__main__":
    unittest.main()
