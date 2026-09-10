"""Tests for the native-resolution overlay path.

The letterbox inverse is the one piece of this path that can be wrong in a way
that still produces a plausible-looking figure: a contour offset by a few pixels
reads as a slightly inaccurate model rather than as a bug. These tests pin the
geometry, and pin it hardest on the non-square case, which is the only one where
padding exists to get wrong.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import _resize_and_pad  # noqa: E402
from spfilm.visualization import (  # noqa: E402
    binary_mask_image,
    invert_letterbox,
    letterbox_geometry,
    overlay_contours,
    overlay_regions,
    thicken,
)


def _disc(size: tuple[int, int], centre, radius: float) -> np.ndarray:
    width, height = size
    rows, columns = np.ogrid[:height, :width]
    return ((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2) <= radius**2


class LetterboxGeometryTests(unittest.TestCase):
    def test_matches_resize_and_pad(self) -> None:
        """The inverse must be derived from the same arithmetic as the forward."""

        for size in ((2047, 1759), (1634, 1634), (452, 452), (800, 1200)):
            with self.subTest(size=size):
                masks = np.zeros((2, size[1], size[0]), dtype=np.uint8)
                image = Image.new("RGB", size)
                canvas, _ = _resize_and_pad(image, masks, 512)
                self.assertEqual(canvas.size, (512, 512))
                (resized, offset) = letterbox_geometry(size, 512)
                # Reproduce the paste the dataset performs and check the pasted
                # region lands where letterbox_geometry says it does.
                probe = Image.new("L", (512, 512), color=0)
                probe.paste(Image.new("L", resized, color=255), offset)
                array = np.asarray(probe)
                rows, columns = np.nonzero(array)
                self.assertEqual((columns.min(), rows.min()), offset)
                self.assertEqual(
                    (columns.max() - columns.min() + 1, rows.max() - rows.min() + 1),
                    resized,
                )

    def test_rejects_degenerate_size(self) -> None:
        with self.assertRaises(ValueError):
            letterbox_geometry((0, 100), 512)


class InvertLetterboxTests(unittest.TestCase):
    def test_round_trip_recovers_the_mask(self) -> None:
        """Letterbox a mask, invert it, and expect the original back."""

        for size in ((2047, 1759), (1634, 1634), (452, 452), (1200, 800)):
            with self.subTest(size=size):
                width, height = size
                disc = _disc(size, (height * 0.45, width * 0.4), min(size) * 0.18)
                cup = _disc(size, (height * 0.45, width * 0.4), min(size) * 0.09)
                masks = np.stack([disc, cup]).astype(np.uint8)
                _, letterboxed = _resize_and_pad(Image.new("RGB", size), masks, 512)
                grid = np.stack(
                    [np.asarray(mask, dtype=np.uint8) >= 128 for mask in letterboxed]
                )
                recovered = invert_letterbox(grid, size, 512)
                for channel, original in enumerate((disc, cup)):
                    intersection = np.count_nonzero(recovered[channel] & original)
                    total = np.count_nonzero(recovered[channel]) + np.count_nonzero(
                        original
                    )
                    dice = 2 * intersection / total
                    # Nearest-neighbour down-and-up loses a boundary pixel or two;
                    # a systematic offset would land far below this.
                    self.assertGreater(dice, 0.98)

    def test_offset_is_not_transposed(self) -> None:
        """A non-square mask must come back the right way up.

        Swapping the PIL (x, y) offset for numpy [y, x] is the easiest mistake to
        make here, and on a square image it is undetectable because the padding is
        symmetric. This asserts on a tall image, where it is not.
        """

        size = (800, 1600)
        width, height = size
        # Deliberately off-centre so a transpose cannot coincidentally agree.
        disc = _disc(size, (height * 0.25, width * 0.6), 90)
        masks = np.stack([disc, disc]).astype(np.uint8)
        _, letterboxed = _resize_and_pad(Image.new("RGB", size), masks, 512)
        grid = np.stack(
            [np.asarray(mask, dtype=np.uint8) >= 128 for mask in letterboxed]
        )
        recovered = invert_letterbox(grid, size, 512)
        rows, columns = np.nonzero(recovered[0])
        expected_rows, expected_columns = np.nonzero(disc)
        self.assertAlmostEqual(rows.mean(), expected_rows.mean(), delta=2.0)
        self.assertAlmostEqual(columns.mean(), expected_columns.mean(), delta=2.0)

    def test_rejects_wrong_grid_size(self) -> None:
        with self.assertRaises(ValueError):
            invert_letterbox(np.zeros((2, 256, 256), dtype=bool), (800, 600), 512)


class ContourDrawingTests(unittest.TestCase):
    def test_thicken_grows_by_one_ring_per_step(self) -> None:
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 5] = True
        self.assertEqual(np.count_nonzero(thicken(mask, 1)), 1)
        self.assertEqual(np.count_nonzero(thicken(mask, 2)), 5)
        self.assertEqual(np.count_nonzero(thicken(mask, 3)), 13)

    def test_overlay_paints_only_the_boundary(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        painted = overlay_contours(image, [mask], [(1.0, 0.0, 0.0)], contour_width=1)
        self.assertTrue(np.allclose(painted[10, 10], (1.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(painted[15, 15], (0.0, 0.0, 0.0)))  # interior
        self.assertTrue(np.allclose(painted[0, 0], (0.0, 0.0, 0.0)))  # background

    def test_rejects_palette_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            overlay_contours(
                np.zeros((8, 8, 3), dtype=np.float32),
                np.zeros((2, 8, 8), dtype=bool),
                [(1.0, 0.0, 0.0)],
            )


class FilledRegionTests(unittest.TestCase):
    def test_blends_interior_and_leaves_background(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        painted = overlay_regions(image, [mask], [(1.0, 0.0, 0.0)], alpha=0.25)
        self.assertAlmostEqual(float(painted[8, 8, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(painted[0, 0, 0]), 0.0, places=6)

    def test_later_masks_paint_over_earlier_ones(self) -> None:
        """A cup lies inside its disc, so it must win where they overlap."""

        image = np.zeros((16, 16, 3), dtype=np.float32)
        disc = np.zeros((16, 16), dtype=bool)
        disc[2:14, 2:14] = True
        cup = np.zeros((16, 16), dtype=bool)
        cup[6:10, 6:10] = True
        painted = overlay_regions(
            image, [disc, cup], [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)], alpha=1.0
        )
        self.assertTrue(np.allclose(painted[8, 8], (0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(painted[3, 3], (1.0, 0.0, 0.0)))

    def test_rejects_alpha_outside_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            overlay_regions(
                np.zeros((4, 4, 3), dtype=np.float32),
                [np.zeros((4, 4), dtype=bool)],
                [(1.0, 0.0, 0.0)],
                alpha=1.5,
            )


class BinaryMaskImageTests(unittest.TestCase):
    def test_renders_flat_colour_on_black(self) -> None:
        disc = np.zeros((10, 10), dtype=bool)
        disc[2:8, 2:8] = True
        cup = np.zeros((10, 10), dtype=bool)
        cup[4:6, 4:6] = True
        canvas = binary_mask_image([disc, cup], [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        self.assertEqual(canvas.shape, (10, 10, 3))
        self.assertTrue(np.allclose(canvas[0, 0], (0.0, 0.0, 0.0)))
        self.assertTrue(np.allclose(canvas[2, 2], (0.0, 1.0, 0.0)))
        self.assertTrue(np.allclose(canvas[5, 5], (0.0, 0.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
