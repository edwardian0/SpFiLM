"""Contract tests for Step 5: the Spatial FiLM (SpFiLM) layer, network and arm.

What is pinned. The layer is the draft's rank-K spatial FiLM (Eq. 2-3): dense
scale and shift fields assembled from per-channel bars plus K image-derived
basis maps weighted by coefficients generated from the conditioning signal,
clamped after assembly and computed in float32 under autocast. The basis maps
see the image and never the signal, so every signal shares them. With K = 0 the
layer is numerically the Global FiLM layer, which is what makes SpFiLM against
Global FiLM a comparison of the spatial term alone. FOV gating, when switched
on, leaves pixels outside the field of view untouched. The spatial arm must
state its rank, the global arm refuses one, and neither the plain nor the
Global FiLM resume fingerprint moves, so every run launched before Step 5
still resumes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.engine import (  # noqa: E402
    SPATIAL_FILM_CONFIG_FIELDS,
    Stage2Config,
    _resume_fingerprint,
)
from spfilm.film.conditioning import fov_descriptor, fov_mask  # noqa: E402
from spfilm.film.global_film import DomainOneHot, GlobalFiLM  # noqa: E402
from spfilm.film.spfilm import SpatialBasis, SpatialFiLM  # noqa: E402
from spfilm.lodo import Domain  # noqa: E402
from spfilm.model import (  # noqa: E402
    ARMS,
    ConditionedUNet,
    PlainUNet,
    SpatialFiLMUNet,
    build_model,
)
from spfilm.stage3 import Stage3ConfigError  # noqa: E402
from spfilm.stage3_single_source import Stage3SingleSourceConfig  # noqa: E402

import run_stage4_all_domains  # noqa: E402
import run_stage5_lodo  # noqa: E402

CONFIGS = PROJECT_ROOT / "configs"
GLOBAL_LODO_CONFIG = CONFIGS / "stage5_lodo_global_film_3dom.json"
LODO_STAGE, LODO_DOMAINS_KEY = run_stage5_lodo.CONFIG_STAGE, run_stage5_lodo.CONFIG_DOMAINS_KEY
# Each SpFiLM config, the Global FiLM config it was derived from, and the plain
# config of the same protocol (the secondary comparison).
SPATIAL_CONFIGS = {
    f"{spatial}{suffix}": (f"{twin}{suffix}", f"{plain}{suffix}", runner)
    for spatial, twin, plain, runner in (
        ("stage4_all_domains_spatial_film_k8_3dom", "stage4_all_domains_global_film_3dom",
         "stage4_all_domains_plain_3dom", run_stage4_all_domains),
        ("stage5_lodo_spatial_film_k8_3dom", "stage5_lodo_global_film_3dom",
         "stage5_lodo_plain_3dom", run_stage5_lodo),
    )
    for suffix in ("", "_create")
}
SPATIAL_CONFIGS_PRESENT = all(
    (CONFIGS / f"{name}.json").is_file()
    for spatial, (twin, plain, _runner) in SPATIAL_CONFIGS.items()
    for name in (spatial, twin, plain)
)


def _signals(*indices: int) -> torch.Tensor:
    """Frozen one-hot conditioning signals, as the network builds them."""

    return DomainOneHot(64)(torch.tensor(indices))


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _zero_generator_output(layer: nn.Module) -> None:
    torch.nn.init.zeros_(layer.generator[-1].weight)
    torch.nn.init.zeros_(layer.generator[-1].bias)


def _coefficient_slices(channels: int, rank: int) -> dict[str, slice]:
    """Where each coefficient lives in the generator's output: [γ̄ | A | β̄ | B]."""

    return {
        "gamma_bar": slice(0, channels),
        "A": slice(channels, channels + channels * rank),
        "beta_bar": slice(channels + channels * rank, 2 * channels + channels * rank),
        "B": slice(2 * channels + channels * rank, 2 * channels * (1 + rank)),
    }


# --------------------------------------------------------------------------
# The basis generators
# --------------------------------------------------------------------------


class SpatialBasisTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.basis = SpatialBasis(rank=5)
        self.images = torch.rand(2, 3, 12, 10)

    def test_maps_are_rank_channels_at_the_image_resolution(self) -> None:
        self.assertEqual(tuple(self.basis(self.images).shape), (2, 5, 12, 10))

    def test_maps_lie_in_minus_one_to_one_even_when_saturated(self) -> None:
        """tanh bounds the coarse maps and bilinear upsampling cannot leave the bound."""

        with torch.no_grad():
            self.basis.layer[-2].weight.fill_(50.0)  # the last InstanceNorm's scale
        maps = self.basis(self.images * 100.0)
        self.assertGreaterEqual(maps.min().item(), -1.0)
        self.assertLessEqual(maps.max().item(), 1.0)
        self.assertGreater(maps.abs().max().item(), 0.99)

    def test_different_images_give_different_maps(self) -> None:
        self.assertFalse(
            torch.allclose(self.basis(self.images[:1]), self.basis(self.images[1:]))
        )

    def test_maps_are_shared_by_every_conditioning_signal(self) -> None:
        """φ and ψ computed once, with no signal, rebuild every signal's fields (Eq. 2)."""

        layer = SpatialFiLM(4, rank=3)
        image = torch.rand(1, 3, 8, 8)
        phi = layer.basis_gamma(image)
        psi = layer.basis_beta(image)
        for index in (0, 1, 2):
            signal = _signals(index)
            gamma_bar, A, beta_bar, B = layer.coefficients(signal)
            gamma, beta = layer.fields(image, signal, size=(8, 8))
            expected_gamma = gamma_bar[:, :, None, None] + torch.einsum("nck,nkhw->nchw", A, phi)
            expected_beta = beta_bar[:, :, None, None] + torch.einsum("nck,nkhw->nchw", B, psi)
            self.assertTrue(torch.allclose(gamma, expected_gamma.clamp(-5, 5), atol=1e-6))
            self.assertTrue(torch.allclose(beta, expected_beta.clamp(-5, 5), atol=1e-6))

    def test_rank_below_one_is_refused(self) -> None:
        for rank in (0, -1):
            with self.assertRaises(ValueError, msg=str(rank)):
                SpatialBasis(rank)

    def test_convolutions_carry_no_bias(self) -> None:
        """InstanceNorm follows each convolution and would remove a bias anyway."""

        convolutions = [m for m in self.basis.modules() if isinstance(m, nn.Conv2d)]
        self.assertEqual(len(convolutions), 2)
        self.assertTrue(all(conv.bias is None for conv in convolutions))
        # The first is strided: the maps are learned one level coarser than
        # the features they modulate, a deliberate smoothness constraint.
        self.assertEqual(convolutions[0].stride, (2, 2))


# --------------------------------------------------------------------------
# The layer
# --------------------------------------------------------------------------


class SpatialFiLMTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.features = torch.rand(2, 8, 6, 5) * 2 - 1
        self.image = torch.rand(2, 3, 6, 5)
        self.signal = _signals(0, 1)

    def test_output_keeps_the_feature_shape(self) -> None:
        out = SpatialFiLM(8, rank=4)(self.features, self.image, self.signal)
        self.assertEqual(tuple(out.shape), (2, 8, 6, 5))

    def test_rank_zero_reproduces_global_film(self) -> None:
        """Eq. 2 collapses to Eq. 1: same generator weights, same output."""

        for channels in (1, 8, 16):
            global_layer = GlobalFiLM(channels)
            spatial = SpatialFiLM(channels, rank=0)
            spatial.generator.load_state_dict(global_layer.generator.state_dict())
            features = torch.randn(3, channels, 7, 9) * 3
            image = torch.rand(3, 3, 7, 9)
            signal = _signals(0, 1, 2)
            self.assertTrue(
                torch.allclose(spatial(features, image, signal), global_layer(features, signal), atol=1e-6),
                f"C={channels}",
            )
            # Also where both clamps are active.
            with torch.no_grad():
                global_layer.generator[-1].bias.copy_(
                    8.0 * torch.tensor([(-1.0) ** i for i in range(2 * channels)])
                )
            spatial.generator.load_state_dict(global_layer.generator.state_dict())
            self.assertTrue(
                torch.allclose(spatial(features, image, signal), global_layer(features, signal), atol=1e-6),
                f"C={channels}, clamped",
            )

    def test_rank_zero_has_no_basis_modules_and_the_global_parameter_count(self) -> None:
        spatial = SpatialFiLM(16, rank=0)
        global_layer = GlobalFiLM(16)
        self.assertIsNone(spatial.basis_gamma)
        self.assertIsNone(spatial.basis_beta)
        self.assertFalse(any("basis" in key for key in spatial.state_dict()))
        self.assertEqual(set(spatial.state_dict()), set(global_layer.state_dict()))
        self.assertEqual(_parameter_count(spatial), _parameter_count(global_layer))
        self.assertEqual(_parameter_count(spatial), 90_656)

    def test_zero_generator_output_is_the_identity(self) -> None:
        layer = SpatialFiLM(8, rank=4)
        _zero_generator_output(layer)
        self.assertTrue(torch.equal(layer(self.features, self.image, self.signal), self.features))

    def test_the_field_varies_over_pixels_only_through_the_spatial_coefficients(self) -> None:
        """The contrast with Global FiLM's test that its modulation is uniform over pixels."""

        layer = SpatialFiLM(8, rank=4)
        gamma, beta = layer.fields(self.image, self.signal, size=(6, 5))
        self.assertGreater(gamma.flatten(2).std(dim=2).min().item(), 0.0)
        self.assertGreater(beta.flatten(2).std(dim=2).min().item(), 0.0)

        slices = _coefficient_slices(8, 4)
        with torch.no_grad():
            for name in ("A", "B"):
                layer.generator[-1].weight[slices[name]] = 0.0
                layer.generator[-1].bias[slices[name]] = 0.0
        gamma, beta = layer.fields(self.image, self.signal, size=(6, 5))
        self.assertTrue(torch.equal(gamma, gamma[:, :, :1, :1].expand_as(gamma)))
        self.assertTrue(torch.equal(beta, beta[:, :, :1, :1].expand_as(beta)))

    def test_the_assembled_field_is_clamped(self) -> None:
        """Bars inside the range, spatial coefficients large: only the sum leaves it."""

        layer = SpatialFiLM(4, rank=2, clamp=1.0)
        slices = _coefficient_slices(4, 2)
        with torch.no_grad():
            last = layer.generator[-1]
            last.weight.zero_()
            last.bias.fill_(3.0)
            last.bias[slices["gamma_bar"]] = 0.9
            last.bias[slices["beta_bar"]] = -0.9
        features = torch.rand(2, 4, 6, 5)
        gamma, beta = layer.fields(self.image, self.signal, size=(6, 5))
        for field in (gamma, beta):
            self.assertLessEqual(field.abs().max().item(), 1.0)
            self.assertEqual(field.abs().max().item(), 1.0)  # the clamp bit
        self.assertTrue(
            torch.allclose(layer(features, self.image, self.signal), (1 + gamma) * features + beta)
        )

    def test_modulation_matches_float32_under_autocast(self) -> None:
        """Basis convolutions, bmm and affine all stay out of half precision."""

        layer = SpatialFiLM(8, rank=4)
        reference = layer(self.features, self.image, self.signal)
        for dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cpu", dtype=dtype):
                under_autocast = layer(self.features, self.image, self.signal)
            self.assertEqual(under_autocast.dtype, self.features.dtype, str(dtype))
            self.assertTrue(torch.equal(under_autocast, reference), str(dtype))

    def test_output_keeps_the_feature_dtype_and_stays_finite_for_large_inputs(self) -> None:
        layer = SpatialFiLM(8, rank=4)
        with torch.no_grad():
            last = layer.generator[-1]
            last.weight.zero_()
            last.bias.copy_(4e4 * torch.tensor([(-1.0) ** i for i in range(last.bias.numel())]))
        features = (self.features * 1e4).half()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out = layer(features, self.image * 100.0, self.signal)
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.isfinite(out).all())

    def test_fov_gating_leaves_pixels_outside_the_field_of_view_untouched(self) -> None:
        layer = SpatialFiLM(8, rank=4, fov_gating=True)
        mask = torch.ones(2, 1, 6, 5)
        mask[:, :, :, :2] = 0.0
        out = layer(self.features, self.image, self.signal, fov_mask=mask)
        outside = (mask == 0).expand_as(out)
        self.assertTrue(torch.equal(out[outside], self.features[outside]))
        self.assertFalse(torch.allclose(out[~outside], self.features[~outside]))

    def test_the_mask_is_nearest_resampled_to_the_feature_grid(self) -> None:
        """The network passes one full-resolution mask to every level."""

        layer = SpatialFiLM(8, rank=4, fov_gating=True)
        mask = torch.ones(2, 1, 12, 10)
        # Nearest reads input column 2j for feature column j, so columns 0 and 1
        # are fully gated. The edge sits mid-pair on purpose: bilinear would
        # blend columns 2 and 3 into a half-gated feature column 1.
        mask[:, :, :, :3] = 0.0
        out = layer(self.features, self.image, self.signal, fov_mask=mask)
        self.assertTrue(torch.equal(out[..., :2], self.features[..., :2]))
        self.assertFalse(torch.allclose(out[..., 2:], self.features[..., 2:]))

    def test_without_gating_the_mask_is_ignored(self) -> None:
        layer = SpatialFiLM(8, rank=4)
        mask = torch.zeros(2, 1, 6, 5)
        self.assertTrue(
            torch.equal(
                layer(self.features, self.image, self.signal, fov_mask=mask),
                layer(self.features, self.image, self.signal),
            )
        )

    def test_parameter_count_matches_the_closed_form(self) -> None:
        embedding, hidden, image_channels, basis_hidden = 64, 256, 3, 16
        for channels, rank in ((1, 0), (16, 0), (16, 1), (16, 8), (32, 16)):
            outputs = 2 * channels * (1 + rank)
            generator = (embedding * hidden + hidden) + (hidden * hidden + hidden) + (hidden * outputs + outputs)
            basis = (
                0
                if rank == 0
                else 2 * (9 * image_channels * basis_hidden + 2 * basis_hidden + 9 * basis_hidden * rank + 2 * rank)
            )
            self.assertEqual(
                _parameter_count(SpatialFiLM(channels, rank=rank)), generator + basis, f"C={channels}, K={rank}"
            )

    def test_coefficients_are_laid_out_gamma_bar_a_beta_bar_b(self) -> None:
        channels, rank = 3, 2
        layer = SpatialFiLM(channels, rank=rank)
        signal = _signals(1)
        before = layer.coefficients(signal)
        self.assertEqual(
            [tuple(t.shape) for t in before], [(1, 3), (1, 3, 2), (1, 3), (1, 3, 2)]
        )
        slices = _coefficient_slices(channels, rank)
        for position, name in enumerate(("gamma_bar", "A", "beta_bar", "B")):
            perturbed = copy.deepcopy(layer)
            with torch.no_grad():
                perturbed.generator[-1].bias[slices[name]] += 1.0
            after = perturbed.coefficients(signal)
            for other, (old, new) in enumerate(zip(before, after)):
                if other == position:
                    self.assertTrue(torch.allclose(new - old, torch.ones_like(old)), name)
                else:
                    self.assertTrue(torch.equal(new, old), f"{name} moved slot {other}")
        # A is channel-major: output C + c * K + k is A[c, k].
        perturbed = copy.deepcopy(layer)
        with torch.no_grad():
            perturbed.generator[-1].bias[channels + 1 * rank + 0] += 1.0
        moved = (perturbed.coefficients(signal)[1] - before[1]).abs() > 0.5
        self.assertEqual(moved.nonzero().tolist(), [[0, 1, 0]])

    def test_spatial_mismatch_between_features_and_image_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            SpatialFiLM(8, rank=4)(self.features, torch.rand(2, 3, 6, 4), self.signal)

    def test_batch_mismatches_are_refused(self) -> None:
        layer = SpatialFiLM(8, rank=4, fov_gating=True)
        with self.assertRaises(ValueError):
            layer(self.features, self.image[:1], self.signal)
        with self.assertRaises(ValueError):
            layer(self.features, self.image, self.signal[:1])
        with self.assertRaises(ValueError):
            layer(self.features, self.image, self.signal, fov_mask=torch.ones(1, 1, 6, 5))

    def test_malformed_inputs_and_settings_are_refused(self) -> None:
        layer = SpatialFiLM(8, rank=4)
        with self.assertRaises(ValueError):
            layer(torch.rand(2, 7, 6, 5), self.image, self.signal)
        with self.assertRaises(ValueError):
            layer(self.features, self.image[0], self.signal)
        with self.assertRaises(ValueError):
            layer(self.features, self.image, self.signal[0])
        for kwargs in ({"num_channels": 0}, {"num_channels": 8, "rank": -1}, {"num_channels": 8, "clamp": 0.0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                SpatialFiLM(**kwargs)


# --------------------------------------------------------------------------
# The field-of-view mask used by FOV gating
# --------------------------------------------------------------------------


def _letterboxed_fundus(count: int = 2, size: int = 64) -> torch.Tensor:
    """A bright retinal disc on black, inside a black letterbox band (like Drishti)."""

    torch.manual_seed(1)
    images = torch.zeros(count, 3, size, size)
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    disc = ((ys - size / 2) ** 2 + (xs - size / 2) ** 2) < (size / 3) ** 2
    tissue = torch.rand(count, 3, size, size) * 0.5 + 0.3
    images[:, :, disc] = tissue[:, :, disc]
    images[:, :, : size // 8] = 0.0  # letterbox band
    return images


class FovMaskTests(unittest.TestCase):
    def test_mask_is_one_on_tissue_and_zero_on_the_black_surround(self) -> None:
        images = _letterboxed_fundus()
        mask = fov_mask(images)
        self.assertEqual(tuple(mask.shape), (2, 1, 64, 64))
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
        self.assertTrue(torch.all(mask[:, :, :8] == 0))  # letterbox band
        self.assertTrue(torch.all(mask[:, :, 32, 32] == 1))  # disc centre

    def test_mask_is_the_rule_the_conditioning_descriptor_uses(self) -> None:
        images = _letterboxed_fundus()
        mask = fov_mask(images)
        means = (images * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
        self.assertTrue(torch.allclose(means, fov_descriptor(images)[:, :3], atol=1e-6))

    def test_non_rgb_input_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fov_mask(torch.rand(1, 1, 8, 8))


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------


class SpatialFiLMUNetTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = SpatialFiLMUNet(num_domains=3, base_channels=8, rank=4)
        # 64 px, not 32: at 32 px the bottleneck is 2x2 and the stride-2 basis
        # convolution leaves one pixel, which InstanceNorm cannot normalise.
        self.inputs = torch.rand(2, 3, 64, 64)
        self.signal = torch.tensor([0, 2])

    def test_preserves_spatial_shape_and_emits_two_channels(self) -> None:
        with torch.inference_mode():
            out = self.model(self.inputs, self.signal)
        self.assertEqual(tuple(out.shape), (2, 2, 64, 64))

    def test_backbone_state_dict_keys_are_the_plain_keys(self) -> None:
        plain_keys = set(PlainUNet(base_channels=8).state_dict())
        nested = {
            k.removeprefix("backbone.") for k in self.model.state_dict() if k.startswith("backbone.")
        }
        self.assertEqual(nested, plain_keys)

    def test_parameter_count_is_backbone_plus_the_conditioning_layers(self) -> None:
        backbone = _parameter_count(self.model.backbone)
        films = _parameter_count(self.model.films)
        self.assertEqual(_parameter_count(self.model), backbone + films)
        self.assertEqual(backbone, _parameter_count(PlainUNet(base_channels=8)))
        self.assertEqual(
            films, sum(_parameter_count(SpatialFiLM(width, rank=4)) for width in (8, 16, 32, 64, 128))
        )

    def test_full_size_parameter_counts_are_the_ones_reported(self) -> None:
        """Base 16, five conditioned levels: the counts quoted next to every result."""

        expected = {
            "plain": 1_944_066,
            "global_film": 2_611_170,
            2: 3_128_618,
            8: 4_667_042,
            16: 6_718_274,
        }
        self.assertEqual(_parameter_count(build_model("plain", 16)), expected["plain"])
        self.assertEqual(
            _parameter_count(build_model("global_film", 16, num_domains=3)), expected["global_film"]
        )
        for rank in (2, 8, 16):
            self.assertEqual(
                _parameter_count(build_model("spatial_film", 16, num_domains=3, rank=rank)),
                expected[rank],
                f"K={rank}",
            )

    def test_film_levels_limits_which_encoder_levels_are_conditioned(self) -> None:
        model = SpatialFiLMUNet(num_domains=2, base_channels=8, film_levels=2, rank=4)
        self.assertEqual([film.num_channels for film in model.films], [8, 16])
        with torch.inference_mode():
            self.assertEqual(tuple(model(self.inputs, torch.tensor([0, 1])).shape), (2, 2, 64, 64))

    def test_with_zero_generator_output_it_is_exactly_the_plain_unet(self) -> None:
        for film in self.model.films:
            _zero_generator_output(film)
        plain = PlainUNet(base_channels=8)
        plain.load_state_dict(self.model.backbone.state_dict())
        with torch.inference_mode():
            self.assertTrue(torch.equal(self.model(self.inputs, self.signal), plain(self.inputs)))

    def test_rank_zero_is_the_global_film_network(self) -> None:
        """Same state dict, same output: the K = 0 identity at network level."""

        conditioned = ConditionedUNet(num_domains=3, base_channels=8)
        spatial = SpatialFiLMUNet(num_domains=3, base_channels=8, rank=0)
        spatial.load_state_dict(conditioned.state_dict())  # strict: the keys are identical
        self.assertEqual(_parameter_count(spatial), _parameter_count(conditioned))
        with torch.inference_mode():
            self.assertTrue(
                torch.allclose(spatial(self.inputs, self.signal), conditioned(self.inputs, self.signal), atol=1e-6)
            )

    def test_basis_generators_see_the_image_resampled_to_each_level_never_the_features(self) -> None:
        seen: dict[int, torch.Tensor] = {}
        for level, film in enumerate(self.model.films):
            film.basis_gamma.register_forward_hook(
                lambda _module, args, _out, level=level: seen.__setitem__(level, args[0])
            )
        with torch.inference_mode():
            self.model(self.inputs, self.signal)
        self.assertEqual(sorted(seen), [0, 1, 2, 3, 4])
        self.assertTrue(torch.equal(seen[0], self.inputs))
        for level in range(1, 5):
            size = 64 >> level
            expected = torch.nn.functional.interpolate(
                self.inputs, size=(size, size), mode="bilinear", align_corners=False
            )
            self.assertTrue(torch.equal(seen[level], expected), f"level {level}")

    def test_fov_gating_leaves_the_black_surround_unmodulated_at_every_level(self) -> None:
        model = SpatialFiLMUNet(num_domains=2, base_channels=8, rank=4, fov_gating=True)
        inputs = _letterboxed_fundus()
        captured: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for level, film in enumerate(model.films):
            film.register_forward_hook(
                lambda _module, args, out, level=level: captured.__setitem__(level, (args[0], args[3], out))
            )
        with torch.inference_mode():
            model(inputs, torch.tensor([0, 1]))
        full_mask = fov_mask(inputs)
        for level, (features, mask, out) in captured.items():
            self.assertTrue(torch.equal(mask, full_mask), f"level {level}")
            outside = (
                torch.nn.functional.interpolate(full_mask, size=features.shape[-2:], mode="nearest") == 0
            ).expand_as(out)
            self.assertTrue(outside.any(), f"level {level}")
            self.assertTrue(torch.equal(out[outside], features[outside]), f"level {level}")
            self.assertFalse(torch.equal(out[~outside], features[~outside]), f"level {level}")

    def test_fov_gating_needs_rgb_input(self) -> None:
        with self.assertRaises(ValueError):
            SpatialFiLMUNet(num_domains=2, in_channels=1, base_channels=8, rank=4, fov_gating=True)

    def test_untrained_signal_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.model(self.inputs, torch.tensor([0, 3]))

    def test_build_model_maps_the_spatial_arm(self) -> None:
        self.assertEqual(ARMS, ("plain", "global_film", "spatial_film"))
        model = build_model("spatial_film", 8, num_domains=2, rank=4)
        self.assertIsInstance(model, SpatialFiLMUNet)
        self.assertEqual({film.rank for film in model.films}, {4})
        self.assertFalse(any(film.fov_gating for film in model.films))
        gated = build_model("spatial_film", 8, num_domains=2, rank=4, fov_gating=True)
        self.assertTrue(all(film.fov_gating for film in gated.films))
        with self.assertRaises(ValueError):
            build_model("spatial_film", 8, rank=4)  # no number of source domains
        with self.assertRaises(ValueError):
            build_model("spatial_film", 8, num_domains=2)  # rank 0 is global_film
        for kwargs in ({"rank": 4}, {"fov_gating": True}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                build_model("global_film", 8, num_domains=2, **kwargs)
        self.assertIsInstance(build_model("global_film", 8, num_domains=2), ConditionedUNet)


# --------------------------------------------------------------------------
# Config plumbing and the resume fingerprint
# --------------------------------------------------------------------------


def _write_config(payload: dict) -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(payload, stream)
    return Path(stream.name)


def _load_lodo(payload: dict) -> Stage3SingleSourceConfig:
    path = _write_config(payload)
    try:
        return Stage3SingleSourceConfig.from_json(path, expected_stage=LODO_STAGE, domains_key=LODO_DOMAINS_KEY)
    finally:
        path.unlink()


def _spatial_payload(**film: object) -> dict:
    """The Step 5 Global FiLM config turned into a spatial one with the given film keys."""

    payload = json.loads(GLOBAL_LODO_CONFIG.read_text())
    payload["arm"] = "spatial_film"
    payload["film"].update(film)
    return payload


def _stage2(**overrides: object) -> Stage2Config:
    values: dict[str, object] = {
        "experiment_name": "spfilm_test",
        "dataset": "refuge",
        "data_root": "datasets/REFUGE",
        "output_dir": "artifacts/spfilm_test",
        "seed": 42,
        "epochs": 10,
    }
    values.update(overrides)
    return Stage2Config(**values)


@unittest.skipUnless(GLOBAL_LODO_CONFIG.is_file(), "Step 5 Global FiLM config not present")
class SpatialFiLMConfigTests(unittest.TestCase):
    counts = {"train": 80, "val": 20, "test": 50}

    def test_rank_and_gating_are_parsed_and_forwarded_to_the_engine(self) -> None:
        config = _load_lodo(_spatial_payload(rank=8))
        self.assertEqual((config.arm, config.film_rank, config.film_fov_gating), ("spatial_film", 8, False))
        engine = config.training_config(Domain.DRISHTI_GS, 42, "artifacts/x")
        self.assertEqual((engine.arm, engine.film_rank, engine.film_fov_gating), ("spatial_film", 8, False))
        gated = _load_lodo(_spatial_payload(rank=2, fov_gating=True))
        engine = gated.training_config(Domain.DRISHTI_GS, 42, "artifacts/x")
        self.assertEqual((engine.film_rank, engine.film_fov_gating), (2, True))

    def test_a_missing_rank_is_refused_for_the_spatial_arm(self) -> None:
        with self.assertRaisesRegex(Stage3ConfigError, "film.rank is required"):
            _load_lodo(_spatial_payload())
        payload = _spatial_payload()
        payload.pop("film")
        with self.assertRaisesRegex(Stage3ConfigError, "film.rank"):
            _load_lodo(payload)

    def test_rank_zero_is_refused_as_global_film(self) -> None:
        with self.assertRaisesRegex(Stage3ConfigError, "use the global_film arm"):
            _load_lodo(_spatial_payload(rank=0))

    def test_rank_must_be_a_positive_integer(self) -> None:
        for rank in (-1, 2.5, True, "8", None):
            with self.assertRaises(Stage3ConfigError, msg=repr(rank)):
                _load_lodo(_spatial_payload(rank=rank))

    def test_fov_gating_must_be_a_bool(self) -> None:
        for value in (1, 0, "false", None):
            with self.assertRaisesRegex(Stage3ConfigError, "fov_gating", msg=repr(value)):
                _load_lodo(_spatial_payload(rank=8, fov_gating=value))

    def test_the_global_arm_refuses_the_spatial_settings(self) -> None:
        for film in ({"rank": 8}, {"rank": 0}, {"fov_gating": False}):
            payload = json.loads(GLOBAL_LODO_CONFIG.read_text())
            payload["film"].update(film)
            with self.assertRaisesRegex(Stage3ConfigError, "no spatial term", msg=str(film)):
                _load_lodo(payload)
        config = _load_lodo(json.loads(GLOBAL_LODO_CONFIG.read_text()))
        self.assertEqual((config.film_rank, config.film_fov_gating), (0, False))

    def test_the_plain_arm_still_carries_no_film_block(self) -> None:
        payload = json.loads((CONFIGS / "stage5_lodo_plain_3dom.json").read_text())
        self.assertNotIn("film", payload)
        payload["film"] = {"rank": 8}
        with self.assertRaises(Stage3ConfigError):
            _load_lodo(payload)

    def test_the_plain_fingerprint_ignores_the_spatial_fields(self) -> None:
        self.assertEqual(
            _resume_fingerprint(_stage2(), self.counts),
            _resume_fingerprint(_stage2(film_rank=8, film_fov_gating=True), self.counts),
        )

    def test_the_global_film_fingerprint_is_the_pre_spatial_hash(self) -> None:
        """A Global FiLM run preempted before Step 5 must still resume after it."""

        config = _stage2(arm="global_film")
        legacy = asdict(config)
        for field in SPATIAL_FILM_CONFIG_FIELDS:
            legacy.pop(field)
        expected = hashlib.sha256(
            json.dumps({"config": legacy, "split_counts": self.counts}, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(_resume_fingerprint(config, self.counts), expected)

    def test_the_spatial_fingerprint_hashes_rank_and_gating(self) -> None:
        k8 = _resume_fingerprint(_stage2(arm="spatial_film", film_rank=8), self.counts)
        self.assertNotEqual(k8, _resume_fingerprint(_stage2(arm="spatial_film", film_rank=2), self.counts))
        self.assertNotEqual(
            k8, _resume_fingerprint(_stage2(arm="spatial_film", film_rank=8, film_fov_gating=True), self.counts)
        )
        self.assertNotEqual(k8, _resume_fingerprint(_stage2(arm="global_film"), self.counts))


# --------------------------------------------------------------------------
# The run configs: each is its Global FiLM twin plus the spatial term
# --------------------------------------------------------------------------


@unittest.skipUnless(SPATIAL_CONFIGS_PRESENT, "SpFiLM configs or their twins not present")
class SpatialFiLMConfigFileTests(unittest.TestCase):
    ARM_KEYS = ("arm", "experiment_name", "output_dir")
    PROTOCOL_ARM_KEYS = ("policy", "paired_arm", "secondary_comparison")

    @staticmethod
    def _read(name: str) -> dict:
        return json.loads((CONFIGS / f"{name}.json").read_text())

    def test_each_config_is_its_global_film_twin_plus_the_spatial_term(self) -> None:
        """Change one thing: SpFiLM against Global FiLM must compare the spatial term alone."""

        for spatial_name, (twin_name, _plain, _runner) in SPATIAL_CONFIGS.items():
            spatial, twin = self._read(spatial_name), self._read(twin_name)
            self.assertEqual(spatial["arm"], "spatial_film", spatial_name)
            self.assertEqual(spatial["film"]["rank"], 8, spatial_name)
            self.assertIs(spatial["film"]["fov_gating"], False, spatial_name)
            for payload in (spatial, twin):
                for key in self.ARM_KEYS:
                    payload.pop(key)
                for key in self.PROTOCOL_ARM_KEYS:
                    payload["protocol"].pop(key, None)
                for key in ("rank", "fov_gating"):
                    payload["film"].pop(key, None)
            self.assertEqual(spatial, twin, f"{spatial_name} drifts from {twin_name}")

    def test_local_and_create_differ_only_in_data_roots(self) -> None:
        for name in SPATIAL_CONFIGS:
            if name.endswith("_create"):
                continue
            local, create = self._read(name), self._read(f"{name}_create")
            self.assertEqual(len(create["domains"]), 4, name)
            for payload in (local, create):
                for domain in payload["domains"].values():
                    domain.pop("data_root")
            self.assertEqual(local, create, name)

    def test_the_arm_is_paired_with_global_film_and_compared_with_plain(self) -> None:
        for spatial_name, (twin_name, plain_name, _runner) in SPATIAL_CONFIGS.items():
            protocol = self._read(spatial_name)["protocol"]
            self.assertEqual(protocol["paired_arm"], self._read(twin_name)["experiment_name"], spatial_name)
            self.assertEqual(
                protocol["secondary_comparison"], self._read(plain_name)["experiment_name"], spatial_name
            )

    def test_each_runner_loads_its_config_and_accepts_the_arm(self) -> None:
        for spatial_name, (twin_name, _plain, runner) in SPATIAL_CONFIGS.items():
            config = Stage3SingleSourceConfig.from_json(
                CONFIGS / f"{spatial_name}.json",
                expected_stage=runner.CONFIG_STAGE,
                domains_key=runner.CONFIG_DOMAINS_KEY,
            )
            runner._require_arm_policy(config)
            twin = Stage3SingleSourceConfig.from_json(
                CONFIGS / f"{twin_name}.json",
                expected_stage=runner.CONFIG_STAGE,
                domains_key=runner.CONFIG_DOMAINS_KEY,
            )
            self.assertEqual(config.arm, "spatial_film")
            self.assertEqual((config.film_rank, config.film_fov_gating), (8, False))
            self.assertEqual(config.test_conditioning, twin.test_conditioning, spatial_name)
            self.assertEqual(config.paired_arm, twin.experiment_name)
            self.assertEqual(set(config.active_domains), set(twin.active_domains))
            engine = config.training_config(config.active_domains[0], 42, "artifacts/x")
            self.assertEqual((engine.arm, engine.film_rank, engine.film_fov_gating), ("spatial_film", 8, False))


if __name__ == "__main__":
    unittest.main()
