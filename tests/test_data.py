from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import (  # noqa: E402
    DatasetLayoutError,
    FundusRecord,
    FundusSegmentationDataset,
    RIM_ONE_DL_FELLOW_EYE_CAVEAT,
    RIM_ONE_DL_MANIFEST_SCHEMA_VERSION,
    decode_mask_channels,
    discover_refuge_validation,
    load_rim_one_dl_split_manifest,
    provider_partition,
    rim_one_dl_release_class_table,
    stratified_partition,
    validate_splits,
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

    def test_split_identity_includes_domain(self) -> None:
        records = [
            FundusRecord(
                sample_id="shared-id",
                domain=domain,
                image_path=Path(f"/{domain}.png"),
                mask_encoding="unused",
            )
            for domain in ("source-a", "source-b", "target")
        ]
        splits = {
            "train": [records[0]],
            "val": [records[1]],
            "test": [records[2]],
        }

        validate_splits(splits, records)


class MixedDomainDataTests(unittest.TestCase):
    def test_refuge_validation_adapter_pairs_all_400_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "canon-images"
            mask_root = root / "canon-masks"
            image_root.mkdir()
            mask_root.mkdir()
            for index in range(400):
                stem = f"v{index:04d}"
                (image_root / f"{stem}.jpg").touch()
                (mask_root / f"{stem}.bmp").touch()

            records = discover_refuge_validation(
                root,
                image_subdir="canon-images",
                mask_subdir="canon-masks",
            )

        self.assertEqual(len(records), 400)
        self.assertEqual({record.domain for record in records}, {"refuge_canon_val"})
        self.assertEqual(records[0].sample_id, "v0000")
        self.assertEqual(records[-1].sample_id, "v0399")

    def test_mixed_domain_metadata_collates_in_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            refuge_mask = np.full((4, 4), 255, dtype=np.uint8)
            refuge_mask[1:3, 1:3] = 128
            refuge_mask[1, 1] = 0
            binary_disc = np.zeros((4, 4), dtype=np.uint8)
            binary_disc[1:3, 1:3] = 255
            binary_cup = np.zeros((4, 4), dtype=np.uint8)
            binary_cup[1, 1] = 255

            refuge_image = root / "refuge.png"
            refuge_mask_path = root / "refuge-mask.png"
            rim_image = root / "rim.png"
            rim_disc = root / "rim-disc.png"
            rim_cup = root / "rim-cup.png"
            Image.fromarray(image).save(refuge_image)
            Image.fromarray(refuge_mask).save(refuge_mask_path)
            Image.fromarray(image).save(rim_image)
            Image.fromarray(binary_disc).save(rim_disc)
            Image.fromarray(binary_cup).save(rim_cup)

            records = [
                FundusRecord(
                    sample_id="refuge",
                    domain="refuge_zeiss",
                    image_path=refuge_image,
                    combined_mask_path=refuge_mask_path,
                    mask_encoding="refuge_0_cup_128_disc_255_background",
                ),
                FundusRecord(
                    sample_id="rim",
                    domain="rim_one_dl",
                    image_path=rim_image,
                    disc_mask_path=rim_disc,
                    cup_mask_path=rim_cup,
                    mask_encoding="separate_binary_foreground_high",
                    release_prefix="r1",
                    hospital_split="training_set",
                    diagnosis_class="normal",
                    native_size=(4, 4),
                ),
            ]
            dataset = FundusSegmentationDataset(records, image_size=4)
            _, _, metadata = next(iter(DataLoader(dataset, batch_size=2)))

        self.assertEqual(set(metadata["domain"]), {"refuge_zeiss", "rim_one_dl"})
        self.assertEqual(len(metadata["letterbox_scale"]), 2)
        self.assertEqual(len(metadata["release_prefix"]), 2)


if __name__ == "__main__":
    unittest.main()
