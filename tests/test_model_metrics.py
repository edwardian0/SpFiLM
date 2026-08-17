from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.metrics import (  # noqa: E402
    OverlapAccumulator,
    hd95,
    per_image_confusion,
    per_image_overlap,
    summarise_per_image_csv,
)
from spfilm.model import PlainUNet  # noqa: E402


CANVAS = 24


def _square(top: int, left: int, size: int = 8) -> np.ndarray:
    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


def _brute_force_hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    """Definition of HD95 written out directly, for cross-checking the fast path."""

    def surface(mask: np.ndarray) -> np.ndarray:
        points = []
        for row, column in zip(*np.nonzero(mask)):
            neighbours = [
                mask[row - 1, column] if row > 0 else False,
                mask[row + 1, column] if row + 1 < mask.shape[0] else False,
                mask[row, column - 1] if column > 0 else False,
                mask[row, column + 1] if column + 1 < mask.shape[1] else False,
            ]
            if not all(neighbours):
                points.append((row, column))
        return np.array(points, dtype=float)

    prediction_surface = surface(prediction)
    target_surface = surface(target)
    pairwise = np.linalg.norm(
        prediction_surface[:, None, :] - target_surface[None, :, :], axis=2
    )
    return max(
        float(np.percentile(pairwise.min(axis=1), 95)),
        float(np.percentile(pairwise.min(axis=0), 95)),
    )


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


class OffsetSquareTests(unittest.TestCase):
    """Two 8x8 squares sharing four columns; every number below is hand-checkable."""

    def setUp(self) -> None:
        self.target = _square(top=4, left=4)
        self.prediction = _square(top=4, left=8)

    def test_dice_and_iou_match_hand_computed_overlap(self) -> None:
        # 64 predicted, 64 actual, 32 shared, union 96.
        prediction = torch.from_numpy(self.prediction)[None, None]
        target = torch.from_numpy(self.target)[None, None]
        dice, iou = per_image_overlap(prediction, target)
        self.assertAlmostEqual(float(dice[0, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(iou[0, 0]), 1 / 3, places=6)

    def test_confusion_counts_and_pixel_accuracy(self) -> None:
        prediction = torch.from_numpy(self.prediction)[None, None]
        target = torch.from_numpy(self.target)[None, None]
        tp, fp, fn, tn = per_image_confusion(prediction, target)
        self.assertEqual(
            (int(tp[0, 0]), int(fp[0, 0]), int(fn[0, 0]), int(tn[0, 0])),
            (32, 32, 32, CANVAS * CANVAS - 96),
        )
        accuracy = (int(tp[0, 0]) + int(tn[0, 0])) / (CANVAS * CANVAS)
        self.assertAlmostEqual(accuracy, 512 / 576, places=6)

    def test_hd95_is_the_four_pixel_offset(self) -> None:
        # The far edge of the prediction sits 4px outside the target's far edge,
        # and 8 of the 28 surface pixels are at that distance, so p95 is 4.0.
        self.assertAlmostEqual(hd95(self.prediction, self.target), 4.0, places=6)
        self.assertAlmostEqual(
            hd95(self.prediction, self.target),
            _brute_force_hd95(self.prediction, self.target),
            places=6,
        )

    def test_hd95_is_symmetric_and_zero_on_identical_masks(self) -> None:
        self.assertAlmostEqual(hd95(self.target, self.target), 0.0, places=6)
        self.assertAlmostEqual(
            hd95(self.prediction, self.target),
            hd95(self.target, self.prediction),
            places=6,
        )


class DegenerateCaseTests(unittest.TestCase):
    def test_empty_prediction_against_non_empty_target(self) -> None:
        prediction = np.zeros((CANVAS, CANVAS), dtype=bool)
        self.assertTrue(math.isnan(hd95(prediction, _square(4, 4))))

    def test_empty_target_against_non_empty_prediction(self) -> None:
        target = np.zeros((CANVAS, CANVAS), dtype=bool)
        self.assertTrue(math.isnan(hd95(_square(4, 4), target)))

    def test_both_empty_scores_perfect_overlap_but_undefined_hd95(self) -> None:
        empty = np.zeros((CANVAS, CANVAS), dtype=bool)
        dice, iou = per_image_overlap(
            torch.from_numpy(empty)[None, None], torch.from_numpy(empty)[None, None]
        )
        self.assertAlmostEqual(float(dice[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=6)
        self.assertTrue(math.isnan(hd95(empty, empty)))


class PerImageCsvTests(unittest.TestCase):
    """Three images: an offset square, an empty cup prediction, and an empty pair."""

    def _accumulate(self) -> OverlapAccumulator:
        empty = np.zeros((CANVAS, CANVAS), dtype=bool)
        targets = np.stack(
            [
                np.stack([_square(4, 4), _square(6, 6, size=4)]),
                np.stack([_square(4, 4), _square(6, 6, size=4)]),
                np.stack([empty, empty]),
            ]
        )
        predictions = np.stack(
            [
                np.stack([_square(4, 8), _square(6, 6, size=4)]),
                np.stack([_square(4, 4), empty]),
                np.stack([empty, empty]),
            ]
        )
        logits = torch.where(torch.from_numpy(predictions), 20.0, -20.0)
        accumulator = OverlapAccumulator()
        accumulator.update(
            logits, torch.from_numpy(targets).float(), image_ids=["a", "b", "c"]
        )
        return accumulator

    def test_csv_rows_and_summary_agree_and_exclusions_are_counted(self) -> None:
        accumulator = self._accumulate()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = accumulator.write_per_image_csv(
                Path(directory) / "per_image.csv"
            )
            text = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                text[0], "image_id,structure,dice,iou,hd95,acc,tp,fp,fn,tn"
            )
            self.assertEqual(len(text), 1 + 6)  # three images x two structures
            summary = summarise_per_image_csv(csv_path)

        self.assertEqual(set(summary), {"disc", "cup"})
        # Disc: a is the offset square (4.0), b is exact (0.0), c is both-empty.
        self.assertEqual(summary["disc"]["hd95_sample_count"], 2)
        self.assertEqual(summary["disc"]["hd95_excluded_count"], 1)
        self.assertEqual(summary["disc"]["hd95_excluded_both_empty"], 1)
        self.assertAlmostEqual(summary["disc"]["hd95_mean"], 2.0, places=6)
        self.assertAlmostEqual(summary["disc"]["dice_mean"], (0.5 + 1.0 + 1.0) / 3)

        # Cup: image b has an empty prediction, image c is empty on both sides.
        self.assertEqual(summary["cup"]["hd95_sample_count"], 1)
        self.assertEqual(summary["cup"]["hd95_excluded_count"], 2)
        self.assertEqual(summary["cup"]["hd95_excluded_empty_prediction"], 1)
        self.assertEqual(summary["cup"]["hd95_excluded_both_empty"], 1)
        self.assertEqual(summary["cup"]["hd95_excluded_empty_target"], 0)
        self.assertAlmostEqual(summary["cup"]["hd95_mean"], 0.0, places=6)
        self.assertEqual(summary["cup"]["sample_count"], 3)

    def test_summary_from_csv_matches_in_memory_summary(self) -> None:
        accumulator = self._accumulate()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = accumulator.write_per_image_csv(
                Path(directory) / "per_image.csv"
            )
            self.assertEqual(summarise_per_image_csv(csv_path), accumulator.compute())

    def test_all_hd95_excluded_reports_none_rather_than_zero(self) -> None:
        empty = np.zeros((1, 2, CANVAS, CANVAS), dtype=bool)
        accumulator = OverlapAccumulator()
        accumulator.update(
            torch.full((1, 2, CANVAS, CANVAS), -20.0),
            torch.from_numpy(empty).float(),
            image_ids=["a"],
        )
        summary = accumulator.compute()
        self.assertIsNone(summary["cup"]["hd95_mean"])
        self.assertIsNone(summary["cup"]["hd95_std"])
        self.assertIsNone(summary["cup"]["hd95_median"])
        self.assertEqual(summary["cup"]["hd95_excluded_count"], 1)


if __name__ == "__main__":
    unittest.main()

