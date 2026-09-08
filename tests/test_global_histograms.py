from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.global_histograms import (  # noqa: E402
    BIN_COUNT,
    CHANNELS,
    POPULATIONS,
    DomainHistograms,
    HistogramError,
    accumulate_paths,
    bin_centres,
    bin_edges,
    image_histograms,
    plot_domain_overlay,
    plot_global_histogram,
    sample_images,
)


def _expected_bin(value_0_255: int) -> int:
    """Index of the bin a uint8 intensity falls in, matching ``np.histogram``."""

    return min(int((value_0_255 / 255.0) * BIN_COUNT), BIN_COUNT - 1)


def _write_image(path: Path, pixels: np.ndarray) -> Path:
    # PNG, so the stored values survive round-tripping exactly and the assertions
    # can name a single bin rather than a tolerance band.
    Image.fromarray(pixels.astype(np.uint8), mode="RGB").save(path)
    return path


def _flat(path: Path, rgb: tuple[int, int, int], size: int = 64) -> Path:
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[:, :] = rgb
    return _write_image(path, pixels)


class BinGeometryTests(unittest.TestCase):
    def test_edges_and_centres_span_the_unit_interval(self) -> None:
        edges = bin_edges()
        centres = bin_centres()
        self.assertEqual(edges.shape, (BIN_COUNT + 1,))
        self.assertEqual(centres.shape, (BIN_COUNT,))
        self.assertAlmostEqual(float(edges[0]), 0.0)
        self.assertAlmostEqual(float(edges[-1]), 1.0)
        self.assertTrue(np.all(np.diff(centres) > 0))


class ImageHistogramTests(unittest.TestCase):
    def test_flat_image_puts_all_mass_in_one_bin(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = _flat(Path(raw) / "grey.png", (128, 128, 128))
            counts, pixel_counts = image_histograms(path)

        self.assertEqual(pixel_counts["all"], 64 * 64)
        self.assertEqual(pixel_counts["fov"], 64 * 64)
        expected = _expected_bin(128)
        for population in POPULATIONS:
            for channel in CHANNELS:
                band = counts[f"{population}/{channel}"]
                self.assertEqual(band.sum(), 64 * 64)
                self.assertEqual(
                    int(np.argmax(band)),
                    expected,
                    f"{population}/{channel} landed in the wrong bin",
                )
                self.assertEqual(band[expected], 64 * 64)

    def test_channels_are_kept_apart(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = _flat(Path(raw) / "rgb.png", (20, 140, 240))
            counts, _ = image_histograms(path)

        for channel, value in (("red", 20), ("green", 140), ("blue", 240)):
            band = counts[f"all/{channel}"]
            self.assertEqual(int(np.argmax(band)), _expected_bin(value))

    def test_fov_excludes_the_black_surround(self) -> None:
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :32] = (200, 200, 200)  # half tissue, half surround
        with tempfile.TemporaryDirectory() as raw:
            path = _write_image(Path(raw) / "half.png", pixels)
            counts, pixel_counts = image_histograms(path)

        self.assertEqual(pixel_counts["all"], 64 * 64)
        self.assertEqual(pixel_counts["fov"], 64 * 32)
        # The surround shows up at zero in ``all`` and is absent from ``fov``.
        self.assertEqual(counts["all/gray"][0], 64 * 32)
        self.assertEqual(counts["fov/gray"][0], 0)
        self.assertEqual(counts["fov/gray"][_expected_bin(200)], 64 * 32)

    def test_unreadable_image_raises_rather_than_being_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            broken = Path(raw) / "broken.png"
            broken.write_bytes(b"not an image")
            with self.assertRaises(HistogramError):
                image_histograms(broken)
            with self.assertRaises(HistogramError):
                image_histograms(Path(raw) / "missing.png")


class AccumulationTests(unittest.TestCase):
    def _corpus(self, directory: Path) -> list[Path]:
        return [
            _flat(directory / "a.png", (60, 60, 60)),
            _flat(directory / "b.png", (60, 60, 60)),
            _flat(directory / "c.png", (180, 180, 180)),
        ]

    def test_densities_integrate_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            histogram = accumulate_paths("synthetic", self._corpus(Path(raw)))

        width = 1.0 / BIN_COUNT
        for population in POPULATIONS:
            for channel in CHANNELS:
                mass = histogram.density(population, channel).sum() * width
                self.assertAlmostEqual(mass, 1.0, places=10)

    def test_proportions_sum_to_one_and_scale_the_density(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            histogram = accumulate_paths("synthetic", self._corpus(Path(raw)))

        width = 1.0 / BIN_COUNT
        for population in POPULATIONS:
            for channel in CHANNELS:
                shares = histogram.proportions(population, channel)
                self.assertAlmostEqual(float(shares.sum()), 1.0, places=10)
                self.assertTrue(np.all(shares >= 0.0))
                np.testing.assert_allclose(
                    shares, histogram.density(population, channel) * width
                )

    def test_proportions_are_comparable_across_domain_sizes(self) -> None:
        # The same picture, once and three times over: an un-normalised curve
        # would treble, a normalised one must not move at all.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            one = _flat(root / "a.png", (110, 110, 110))
            small = accumulate_paths("small", [one])
            large = accumulate_paths(
                "large",
                [one, _flat(root / "b.png", (110, 110, 110)), _flat(root / "c.png", (110, 110, 110))],
            )

        np.testing.assert_allclose(
            small.proportions("fov", "gray"), large.proportions("fov", "gray")
        )
        self.assertAlmostEqual(
            float(large.counts("fov", "gray").sum()),
            3 * float(small.counts("fov", "gray").sum()),
            places=6,
        )

    def test_accumulation_weights_by_pixels_not_images(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            histogram = accumulate_paths("synthetic", self._corpus(Path(raw)))

        self.assertEqual(histogram.image_count, 3)
        self.assertEqual(histogram.pixel_counts["all"], 3 * 64 * 64)
        self.assertEqual(histogram.pixel_counts["fov"], 3 * 64 * 64)
        counts = histogram.counts("fov", "gray")
        # Two of the three images are dark, so that bin carries twice the mass.
        self.assertAlmostEqual(counts[_expected_bin(60)], 2 * 64 * 64, places=6)
        self.assertAlmostEqual(counts[_expected_bin(180)], 1 * 64 * 64, places=6)

    def test_counts_recover_the_population_total(self) -> None:
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :16] = (150, 150, 150)
        with tempfile.TemporaryDirectory() as raw:
            path = _write_image(Path(raw) / "quarter.png", pixels)
            histogram = accumulate_paths("synthetic", [path])

        for population in POPULATIONS:
            for channel in CHANNELS:
                total = histogram.counts(population, channel).sum()
                self.assertAlmostEqual(
                    total, histogram.pixel_counts[population], places=6
                )

    def test_worker_pool_matches_serial_accumulation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            paths = self._corpus(Path(raw))
            serial = accumulate_paths("synthetic", paths, workers=1)
            parallel = accumulate_paths("synthetic", paths, workers=2)

        for population in POPULATIONS:
            for channel in CHANNELS:
                np.testing.assert_allclose(
                    serial.density(population, channel),
                    parallel.density(population, channel),
                )

    def test_empty_corpus_is_an_error(self) -> None:
        with self.assertRaises(HistogramError):
            accumulate_paths("synthetic", [])


class OverlayTests(unittest.TestCase):
    def _two_domains(self, root: Path) -> list:
        return [
            accumulate_paths("refuge_zeiss", [_flat(root / "z.png", (70, 70, 70))]),
            accumulate_paths("rim_one_dl", [_flat(root / "r.png", (170, 170, 170))]),
        ]

    def test_overlay_is_written_and_not_blank(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = plot_domain_overlay(
                self._two_domains(root), root / "figures" / "overlay.png"
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 5_000)

    def test_annotations_change_the_legend(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            histograms = self._two_domains(root)
            plain = plot_domain_overlay(histograms, root / "plain.png")
            marked = plot_domain_overlay(
                histograms,
                root / "marked.png",
                annotations={"refuge_zeiss": "W1 to rest 0.1273"},
            )
            self.assertNotEqual(plain.read_bytes(), marked.read_bytes())

    def test_all_population_is_drawn_on_a_log_axis(self) -> None:
        # The surround spike would otherwise flatten the tissue distribution.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            histograms = self._two_domains(root)
            fov = plot_domain_overlay(histograms, root / "fov.png", population="fov")
            every = plot_domain_overlay(histograms, root / "all.png", population="all")
            self.assertNotEqual(fov.read_bytes(), every.read_bytes())

    def test_empty_and_unknown_arguments_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            histograms = self._two_domains(root)
            with self.assertRaises(HistogramError):
                plot_domain_overlay([], root / "x.png")
            with self.assertRaises(HistogramError):
                plot_domain_overlay(histograms, root / "x.png", population="nope")
            with self.assertRaises(HistogramError):
                plot_domain_overlay(histograms, root / "x.png", channel="alpha")


class SampleTests(unittest.TestCase):
    def test_sample_is_the_requested_size_and_a_real_subset(self) -> None:
        items = list(range(100))
        drawn = sample_images("refuge_zeiss", items, 50)
        self.assertEqual(len(drawn), 50)
        self.assertEqual(len(set(drawn)), 50, "sampled without replacement")
        self.assertTrue(set(drawn).issubset(items))
        self.assertEqual(drawn, sorted(drawn), "original order is preserved")

    def test_sample_is_reproducible_from_the_seed(self) -> None:
        items = list(range(100))
        self.assertEqual(
            sample_images("refuge_zeiss", items, 20, seed=7),
            sample_images("refuge_zeiss", items, 20, seed=7),
        )
        self.assertNotEqual(
            sample_images("refuge_zeiss", items, 20, seed=7),
            sample_images("refuge_zeiss", items, 20, seed=8),
        )

    def test_each_domain_draws_independently(self) -> None:
        items = list(range(100))
        self.assertNotEqual(
            sample_images("refuge_zeiss", items, 20, seed=0),
            sample_images("rim_one_dl", items, 20, seed=0),
        )

    def test_domain_smaller_than_the_sample_is_used_whole(self) -> None:
        items = list(range(30))
        self.assertEqual(sample_images("drishti_gs", items, 50), items)

    def test_sample_is_not_the_head_of_the_sequence(self) -> None:
        # Records arrive in filename order, which tracks release prefix; taking
        # the head would be a biased draw rather than a sample.
        items = list(range(500))
        self.assertNotEqual(sample_images("refuge_zeiss", items, 50), items[:50])

    def test_non_positive_sample_is_rejected(self) -> None:
        with self.assertRaises(HistogramError):
            sample_images("refuge_zeiss", list(range(10)), 0)


class PlotTests(unittest.TestCase):
    def _histogram(self, directory: Path) -> DomainHistograms:
        return accumulate_paths("synthetic", [_flat(directory / "a.png", (90, 90, 90))])

    def test_figure_is_written_and_not_blank(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            histogram = self._histogram(root)
            # A nested directory also pins that the parent is created.
            output = plot_global_histogram(histogram, root / "figures" / "hist.png")
            self.assertTrue(output.is_file())
            # plt.show() before savefig used to leave a blank canvas behind; a
            # real figure is comfortably larger than a few hundred bytes.
            self.assertGreater(output.stat().st_size, 5_000)
            with Image.open(output) as rendered:
                self.assertGreater(rendered.size[0], 100)

    def test_figure_is_normalised_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            histogram = self._histogram(root)
            normalised = plot_global_histogram(histogram, root / "norm.png")
            counted = plot_global_histogram(
                histogram, root / "counts.png", normalise=False
            )
            # Same data, different y-scale: the two renders must not be
            # identical, which is what would happen if the default silently
            # stopped applying.
            self.assertNotEqual(normalised.read_bytes(), counted.read_bytes())

    def test_unknown_population_or_channel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            histogram = self._histogram(root)
            with self.assertRaises(HistogramError):
                plot_global_histogram(histogram, root / "x.png", population="nope")
            with self.assertRaises(HistogramError):
                plot_global_histogram(histogram, root / "x.png", channel="alpha")


if __name__ == "__main__":
    unittest.main()
