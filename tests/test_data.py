from __future__ import annotations

import json
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
    RIM_ONE_DL_FELLOW_EYE_CAVEAT,
    RIM_ONE_DL_MANIFEST_SCHEMA_VERSION,
    decode_mask_channels,
    load_rim_one_dl_split_manifest,
    provider_partition,
    rim_one_dl_release_class_table,
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

    def test_rim_one_dl_repairs_only_the_pinned_source_defect(self) -> None:
        disc = np.zeros((4, 4), dtype=np.uint8)
        disc[0, 0] = 255
        cup = np.zeros((4, 4), dtype=np.uint8)
        cup[0, :4] = 255
        cup[1, :4] = 255  # one contained pixel and seven outside pixels
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            disc_path = directory_path / "disc.png"
            cup_path = directory_path / "cup.png"
            Image.fromarray(disc).save(disc_path)
            Image.fromarray(cup).save(cup_path)
            record = FundusRecord(
                sample_id="r2_Im357",
                domain="rim_one_dl",
                image_path=directory_path / "unused.png",
                disc_mask_path=disc_path,
                cup_mask_path=cup_path,
                mask_encoding="rim_one_dl_foreground_high",
                source_cup_repair_pixels=7,
            )
            masks = decode_mask_channels(record)
        self.assertEqual(int(masks[0].sum()), 1)
        self.assertEqual(int(masks[1].sum()), 1)
        self.assertTrue(np.all(masks[1] <= masks[0]))


class RimOneManifestTests(unittest.TestCase):
    @staticmethod
    def _records() -> list[FundusRecord]:
        records = [
            FundusRecord(
                sample_id=f"sample_{index:03d}",
                domain="rim_one_dl",
                image_path=Path(f"/sample_{index:03d}.png"),
                mask_encoding="unused",
                release_prefix=f"r{index % 3 + 1}",
                diagnosis_class=(
                    "glaucoma" if (index // 3) % 2 == 0 else "normal"
                ),
            )
            for index in range(485)
        ]
        return records

    @staticmethod
    def _payload(records: list[FundusRecord]) -> dict[str, object]:
        return {
            "schema_version": RIM_ONE_DL_MANIFEST_SCHEMA_VERSION,
            "dataset": "rim_one_dl",
            "seed": 42,
            "source_record_count": len(records),
            "provenance": {
                "generator_script": "generate_rim_one_dl_split.py",
                "git_commit": "0" * 40,
                "working_tree_dirty": False,
                "seed": 42,
                "generation_date_utc": "2026-08-26",
                "release_class_table": rim_one_dl_release_class_table(records),
                "release_only_fallback_releases": [],
                "fellow_eye_caveat": RIM_ONE_DL_FELLOW_EYE_CAVEAT,
            },
            "partitions": {
                "train": [record.sample_id for record in records[:340]],
                "val": [record.sample_id for record in records[340:388]],
                "test": [record.sample_id for record in records[388:]],
            },
        }

    def test_manifest_maps_all_485_discovered_stems_exactly_once(self) -> None:
        records = self._records()
        payload = self._payload(records)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "rim_one_dl.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            splits = load_rim_one_dl_split_manifest(records, manifest_path)
        self.assertEqual(
            {name: len(rows) for name, rows in splits.items()},
            {"train": 340, "val": 48, "test": 97},
        )
        listed = [record.sample_id for rows in splits.values() for record in rows]
        self.assertEqual(len(listed), len(set(listed)))

    def test_manifest_without_required_provenance_fails_closed(self) -> None:
        records = self._records()
        payload = self._payload(records)
        del payload["provenance"]
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "rim_one_dl.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(DatasetLayoutError):
                load_rim_one_dl_split_manifest(records, manifest_path)


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
