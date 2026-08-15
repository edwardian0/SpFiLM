from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import (  # noqa: E402
    DatasetLayoutError,
    FundusRecord,
    decode_mask_channels,
    provider_partition,
    stratified_partition,
)


class MaskDecodingTests(unittest.TestCase):
    def test_refuge_values_become_nested_disc_and_cup_channels(self) -> None:
        source = np.full((5, 5), 255, dtype=np.uint8)
        source[1:4, 1:4] = 128
        source[2, 2] = 0
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.bmp"
            Image.fromarray(source).save(mask_path)
            record = FundusRecord(
                sample_id="sample",
                domain="refuge",
                image_path=Path(directory) / "unused.jpg",
                combined_mask_path=mask_path,
                mask_encoding="refuge_0_cup_128_disc_255_background",
            )
            masks = decode_mask_channels(record)
        self.assertEqual(masks.shape, (2, 5, 5))
        self.assertEqual(int(masks[0].sum()), 9)
        self.assertEqual(int(masks[1].sum()), 1)
        self.assertTrue(np.all(masks[1] <= masks[0]))

    def test_drishti_uses_three_of_four_consensus(self) -> None:
        disc = np.array([[0, 128], [191, 255]], dtype=np.uint8)
        cup = np.array([[0, 0], [191, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            disc_path = directory_path / "disc.png"
            cup_path = directory_path / "cup.png"
            Image.fromarray(disc).save(disc_path)
            Image.fromarray(cup).save(cup_path)
            record = FundusRecord(
                sample_id="sample",
                domain="drishti",
                image_path=directory_path / "unused.png",
                disc_mask_path=disc_path,
                cup_mask_path=cup_path,
                mask_encoding="drishti_softmap_three_of_four",
            )
            masks = decode_mask_channels(record)
        np.testing.assert_array_equal(masks[0], [[0, 0], [1, 1]])
        np.testing.assert_array_equal(masks[1], [[0, 0], [1, 1]])

    def test_cup_outside_disc_fails_closed(self) -> None:
        disc = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        cup = np.array([[0, 255], [0, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            disc_path = directory_path / "disc.png"
            cup_path = directory_path / "cup.png"
            Image.fromarray(disc).save(disc_path)
            Image.fromarray(cup).save(cup_path)
            record = FundusRecord(
                sample_id="bad",
                domain="rim",
                image_path=directory_path / "unused.png",
                disc_mask_path=disc_path,
                cup_mask_path=cup_path,
                mask_encoding="separate_binary_foreground_high",
            )
            with self.assertRaises(DatasetLayoutError):
                decode_mask_channels(record)


class SplitTests(unittest.TestCase):
    @staticmethod
    def _record(sample_id: str, stratum: str, split_hint: str | None = None):
        return FundusRecord(
            sample_id=sample_id,
            domain="test",
            image_path=Path(f"/{sample_id}.png"),
            mask_encoding="unused",
            stratum=stratum,
            split_hint=split_hint,
        )

    def test_refuge_split_is_256_64_80_and_disjoint(self) -> None:
        records = [self._record(f"g{i:03d}", "glaucoma") for i in range(40)]
        records += [
            self._record(f"n{i:03d}", "non_glaucoma") for i in range(360)
        ]
        splits = stratified_partition(records, 42, 0.2, 0.2)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 256,
            "val": 64,
            "test": 80,
        })
        all_ids = [row.sample_id for rows in splits.values() for row in rows]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_provider_test_stays_locked(self) -> None:
        records = [
            self._record(f"g{i:03d}", "glaucoma", "provider_train")
            for i in range(35)
        ]
        records += [
            self._record(f"n{i:03d}", "normal", "provider_train")
            for i in range(15)
        ]
        records += [
            self._record(f"t{i:03d}", "all", "provider_test") for i in range(51)
        ]
        splits = provider_partition(records, seed=42, val_fraction=0.2)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 40,
            "val": 10,
            "test": 51,
        })
        self.assertTrue(
            all(record.split_hint == "provider_test" for record in splits["test"])
        )


if __name__ == "__main__":
    unittest.main()

