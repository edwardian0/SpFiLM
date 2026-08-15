from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.metrics import OverlapAccumulator, per_image_overlap  # noqa: E402
from spfilm.model import PlainUNet  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_plain_unet_preserves_spatial_shape_and_emits_two_channels(self) -> None:
        model = PlainUNet(base_channels=8)
        inputs = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (1, 2, 64, 64))


class MetricTests(unittest.TestCase):
    def test_disc_and_cup_dice_remain_separate(self) -> None:
        targets = torch.zeros(1, 2, 4, 4)
        targets[:, 0, 1:3, 1:3] = 1
        targets[:, 1, 1, 1] = 1
        predictions = targets.clone()
        predictions[:, 1] = 0
        dice, _ = per_image_overlap(predictions, targets)
        self.assertAlmostEqual(float(dice[0, 0]), 1.0)
        self.assertLess(float(dice[0, 1]), 1e-6)

        logits = torch.where(predictions > 0, 20.0, -20.0)
        accumulator = OverlapAccumulator()
        accumulator.update(logits, targets)
        metrics = accumulator.compute()
        self.assertEqual(set(metrics), {"disc", "cup"})
        self.assertNotIn("average", metrics)


if __name__ == "__main__":
    unittest.main()

