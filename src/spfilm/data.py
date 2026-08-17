from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DRISHTI_CONSENSUS_THRESHOLD = 191  # At least three of four annotators.


class DatasetLayoutError(RuntimeError):
    """Raised when a dataset cannot be paired without guessing."""


@dataclass(frozen=True)
class FundusRecord:
    """One image and the files needed to build its two target channels."""

    sample_id: str
    domain: str
    image_path: Path
    mask_encoding: str
    combined_mask_path: Path | None = None
    disc_mask_path: Path | None = None
    cup_mask_path: Path | None = None
    split_hint: str | None = None
    stratum: str = "all"

    @property
    def mask_paths(self) -> tuple[Path, ...]:
        if self.combined_mask_path is not None:
            return (self.combined_mask_path,)
        if self.disc_mask_path is not None and self.cup_mask_path is not None:
            return (self.disc_mask_path, self.cup_mask_path)
        return ()


def _files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def _unique_by_stem(paths: Iterable[Path], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        key = path.stem.lower()
        if key in result:
            raise DatasetLayoutError(
                f"Duplicate {label} stem {key!r}: {result[key]} and {path}"
            )
        result[key] = path
    return result


def discover_refuge_training(root: str | Path) -> list[FundusRecord]:
    """Pair the 400 images from REFUGE's training-camera domain."""

    root = Path(root).expanduser().resolve()
    image_candidates = [
        root / "Training400",
        root / "REFUGE-Training400" / "Training400",
    ]
    image_root = next(
        (
            candidate
            for candidate in image_candidates
            if len(_files(candidate, {".jpg", ".jpeg", ".png"})) == 400
        ),
        None,
    )
    mask_root = root / "Annotation-Training400" / "Disc_Cup_Masks"
    if image_root is None:
        counts = {
            str(path): len(_files(path, {".jpg", ".jpeg", ".png"}))
            for path in image_candidates
            if path.exists()
        }
        raise DatasetLayoutError(
            "REFUGE training-camera pool was not found with 400 images. "
            f"Candidate counts: {counts}"
        )
    if not mask_root.is_dir():
        raise DatasetLayoutError(f"REFUGE mask directory is missing: {mask_root}")

    images = _unique_by_stem(
        _files(image_root, {".jpg", ".jpeg", ".png"}), "REFUGE image"
    )
    masks = _unique_by_stem(_files(mask_root, {".bmp", ".png"}), "REFUGE mask")
    missing_masks = sorted(images.keys() - masks.keys())
    orphan_masks = sorted(masks.keys() - images.keys())
    if missing_masks or orphan_masks:
        raise DatasetLayoutError(
            "REFUGE image/mask pairing failed: "
            f"missing masks={missing_masks[:5]}, orphan masks={orphan_masks[:5]}"
        )

    return [
        FundusRecord(
            sample_id=stem,
            domain="refuge_zeiss",
            image_path=images[stem],
            combined_mask_path=masks[stem],
            mask_encoding="refuge_0_cup_128_disc_255_background",
            stratum=images[stem].parent.name.lower().replace("-", "_"),
        )
        for stem in sorted(images)
    ]


def discover_drishti(root: str | Path) -> list[FundusRecord]:
    """Pair Drishti-GS images with its four-reader consensus soft maps."""

    root = Path(root).expanduser().resolve()
    layouts = [
        (
            "provider_train",
            root
            / "Training-20211018T055246Z-001"
            / "Training"
            / "Images",
            root
            / "Training-20211018T055246Z-001"
            / "Training"
            / "GT",
        ),
        (
            "provider_test",
            root / "Test-20211018T060000Z-001" / "Test" / "Images",
            root / "Test-20211018T060000Z-001" / "Test" / "Test_GT",
        ),
    ]
    records: list[FundusRecord] = []
    for split_hint, image_root, ground_truth_root in layouts:
        if not image_root.is_dir() or not ground_truth_root.is_dir():
            raise DatasetLayoutError(
                f"Drishti-GS {split_hint} layout is incomplete: "
                f"images={image_root}, ground_truth={ground_truth_root}"
            )
        for image_path in _files(image_root, {".png", ".jpg", ".jpeg"}):
            sample_id = image_path.stem
            softmap_root = ground_truth_root / sample_id / "SoftMap"
            disc_path = softmap_root / f"{sample_id}_ODsegSoftmap.png"
            cup_path = softmap_root / f"{sample_id}_cupsegSoftmap.png"
            if not disc_path.is_file() or not cup_path.is_file():
                raise DatasetLayoutError(
                    f"Drishti-GS soft maps are missing for {sample_id}: "
                    f"disc={disc_path}, cup={cup_path}"
                )
            records.append(
                FundusRecord(
                    sample_id=sample_id,
                    domain="drishti_gs",
                    image_path=image_path,
                    disc_mask_path=disc_path,
                    cup_mask_path=cup_path,
                    mask_encoding="drishti_softmap_three_of_four",
                    split_hint=split_hint,
                    stratum=image_path.parent.name.lower(),
                )
            )

    counts = {
        hint: sum(record.split_hint == hint for record in records)
        for hint in ("provider_train", "provider_test")
    }
    if counts != {"provider_train": 50, "provider_test": 51}:
        raise DatasetLayoutError(
            f"Expected Drishti-GS provider split 50/51, found {counts}"
        )
    return sorted(records, key=lambda record: record.sample_id)


def load_rim_one_r3_manifest(
    root: str | Path, manifest_path: str | Path
) -> list[FundusRecord]:
    """Load RIM-ONE-r3 from an explicit pairing manifest.

    RIM-ONE-r3 has multiple annotation choices (two experts and averaged
    annotations), so this loader deliberately does not guess which files are the
    research target. Paths in the manifest are relative to ``root``.
    """

    root = Path(root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    required = {
        "sample_id",
        "split",
        "image_path",
        "disc_mask_path",
        "cup_mask_path",
        "encoding",
    }
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise DatasetLayoutError(
                f"RIM-ONE-r3 manifest is missing columns: {sorted(missing_columns)}"
            )
        rows = list(reader)

    records: list[FundusRecord] = []
    for row_number, row in enumerate(rows, start=2):
        encoding = row["encoding"].strip()
        if encoding not in {"foreground_high", "foreground_low"}:
            raise DatasetLayoutError(
                f"RIM-ONE-r3 manifest row {row_number} has unsupported encoding "
                f"{encoding!r}; use foreground_high or foreground_low"
            )
        split = row["split"].strip().lower()
        if split not in {"train", "test"}:
            raise DatasetLayoutError(
                f"RIM-ONE-r3 manifest row {row_number} split must be train or test"
            )
        paths = [
            (root / row[column].strip()).resolve()
            for column in ("image_path", "disc_mask_path", "cup_mask_path")
        ]
        if not all(path.is_file() for path in paths):
            raise DatasetLayoutError(
                f"RIM-ONE-r3 manifest row {row_number} references missing files: "
                f"{[str(path) for path in paths if not path.is_file()]}"
            )
        records.append(
            FundusRecord(
                sample_id=row["sample_id"].strip(),
                domain="rim_one_r3",
                image_path=paths[0],
                disc_mask_path=paths[1],
                cup_mask_path=paths[2],
                mask_encoding=f"separate_binary_{encoding}",
                split_hint=f"provider_{split}",
                stratum=(row.get("stratum") or "all").strip().lower(),
            )
        )

    split_counts = {
        split: sum(record.split_hint == f"provider_{split}" for record in records)
        for split in ("train", "test")
    }
    if split_counts != {"train": 99, "test": 60}:
        raise DatasetLayoutError(
            "RIM-ONE-r3 must use the published 99/60 split; "
            f"manifest contains {split_counts}"
        )
    if len({record.sample_id for record in records}) != len(records):
        raise DatasetLayoutError("RIM-ONE-r3 manifest contains duplicate sample IDs")
    return sorted(records, key=lambda record: record.sample_id)


def inspect_rim_download(root: str | Path) -> dict[str, object]:
    """Describe a RIM download without treating classification labels as masks."""

    root = Path(root).expanduser().resolve()
    images = _files(root, IMAGE_SUFFIXES)
    mask_tokens = ("mask", "ground", "truth", "segment", "disc", "cup")
    mask_like = [
        path
        for path in images
        if any(token in str(path.relative_to(root)).lower() for token in mask_tokens)
    ]
    license_text = ""
    license_path = root / "LICENSE.txt"
    if license_path.is_file():
        license_text = license_path.read_text(encoding="utf-8", errors="replace")
    is_rim_one_dl = "RIM-ONE DL" in license_text
    unique_names = {path.name for path in images}
    return {
        "root": str(root),
        "image_file_count": len(images),
        "unique_filename_count": len(unique_names),
        "mask_like_file_count": len(mask_like),
        "identified_release": "RIM-ONE DL" if is_rim_one_dl else "unknown",
        "stage2_compatible": len(images) == 159 and len(mask_like) >= 318,
        "diagnosis": (
            "This is the 485-image RIM-ONE DL classification release, stored in "
            "two alternative partition trees, and it contains no optic-disc/cup "
            "segmentation masks. Download the 159-image RIM-ONE-r3 segmentation "
            "release and make an explicit averaged-annotation manifest."
            if is_rim_one_dl and len(images) == 970 and len(mask_like) == 0
            else "Inspect this layout and provide a RIM-ONE-r3 pairing manifest."
        ),
    }


def _read_grayscale(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_mask_channels(record: FundusRecord) -> np.ndarray:
    """Return binary ``[2, H, W]`` channels ordered as disc, then cup."""

    if record.mask_encoding == "refuge_0_cup_128_disc_255_background":
        if record.combined_mask_path is None:
            raise DatasetLayoutError(f"{record.sample_id} has no combined mask")
        source = _read_grayscale(record.combined_mask_path)
        values = set(np.unique(source).tolist())
        if not values <= {0, 128, 255}:
            raise DatasetLayoutError(
                f"Unexpected REFUGE mask values for {record.sample_id}: {sorted(values)}"
            )
        disc = source <= 128
        cup = source == 0
    elif record.mask_encoding == "drishti_softmap_three_of_four":
        if record.disc_mask_path is None or record.cup_mask_path is None:
            raise DatasetLayoutError(f"{record.sample_id} has incomplete soft maps")
        disc_source = _read_grayscale(record.disc_mask_path)
        cup_source = _read_grayscale(record.cup_mask_path)
        allowed = {0, 64, 128, 191, 255}
        for label, source in (("disc", disc_source), ("cup", cup_source)):
            values = set(np.unique(source).tolist())
            if not values <= allowed:
                raise DatasetLayoutError(
                    f"Unexpected Drishti {label} soft-map values for "
                    f"{record.sample_id}: {sorted(values)}"
                )
        disc = disc_source >= DRISHTI_CONSENSUS_THRESHOLD
        cup = cup_source >= DRISHTI_CONSENSUS_THRESHOLD
    elif record.mask_encoding in {
        "separate_binary_foreground_high",
        "separate_binary_foreground_low",
    }:
        if record.disc_mask_path is None or record.cup_mask_path is None:
            raise DatasetLayoutError(f"{record.sample_id} has incomplete binary masks")
        disc_source = _read_grayscale(record.disc_mask_path)
        cup_source = _read_grayscale(record.cup_mask_path)
        if record.mask_encoding.endswith("foreground_high"):
            disc = disc_source >= 128
            cup = cup_source >= 128
        else:
            disc = disc_source < 128
            cup = cup_source < 128
    else:
        raise DatasetLayoutError(
            f"Unsupported mask encoding {record.mask_encoding!r} for {record.sample_id}"
        )

    if disc.shape != cup.shape:
        raise DatasetLayoutError(
            f"Disc/cup shape mismatch for {record.sample_id}: {disc.shape} vs {cup.shape}"
        )
    outside = cup & ~disc
    if outside.any():
        raise DatasetLayoutError(
            f"Cup is not contained in disc for {record.sample_id}; "
            f"{int(outside.sum())} pixels violate the internal contract"
        )
    if not disc.any() or not cup.any():
        raise DatasetLayoutError(
            f"Empty disc or cup mask after decoding {record.sample_id}"
        )
    return np.stack((disc, cup)).astype(np.uint8)


def audit_records(records: Sequence[FundusRecord]) -> dict[str, object]:
    if not records:
        raise DatasetLayoutError("Cannot audit an empty record list")

    disc_areas: list[int] = []
    cup_areas: list[int] = []
    image_sizes: set[tuple[int, int]] = set()
    image_hashes: dict[str, list[str]] = {}
    for record in records:
        with Image.open(record.image_path) as image:
            image_size = image.size
        masks = decode_mask_channels(record)
        mask_size = (masks.shape[2], masks.shape[1])
        if image_size != mask_size:
            raise DatasetLayoutError(
                f"Image/mask size mismatch for {record.sample_id}: "
                f"image={image_size}, mask={mask_size}"
            )
        image_sizes.add(image_size)
        image_hashes.setdefault(_sha256(record.image_path), []).append(record.sample_id)
        disc_areas.append(int(masks[0].sum()))
        cup_areas.append(int(masks[1].sum()))

    duplicate_images = [ids for ids in image_hashes.values() if len(ids) > 1]
    if duplicate_images:
        raise DatasetLayoutError(
            "Byte-identical source images would leak across a split: "
            f"{duplicate_images[:5]}"
        )

    def area_summary(values: list[int]) -> dict[str, float | int]:
        array = np.asarray(values)
        return {
            "min": int(array.min()),
            "median": float(np.median(array)),
            "max": int(array.max()),
        }

    return {
        "domain": records[0].domain,
        "sample_count": len(records),
        "mask_encoding": sorted({record.mask_encoding for record in records}),
        "split_hints": {
            str(hint): sum(record.split_hint == hint for record in records)
            for hint in sorted({record.split_hint for record in records}, key=str)
        },
        "strata": {
            stratum: sum(record.stratum == stratum for record in records)
            for stratum in sorted({record.stratum for record in records})
        },
        "image_size_count": len(image_sizes),
        "image_size_examples": sorted(image_sizes)[:5],
        "duplicate_image_hash_groups": 0,
        "disc_area_pixels": area_summary(disc_areas),
        "cup_area_pixels": area_summary(cup_areas),
        "contract": {
            "shape": "[2, H, W]",
            "dtype": "uint8 before transforms; float32 in the model",
            "channel_0": "optic_disc including optic_cup",
            "channel_1": "optic_cup",
            "values": [0, 1],
            "cup_subset_of_disc": True,
        },
        "status": "ok",
    }


def stratified_partition(
    records: Sequence[FundusRecord],
    seed: int,
    test_fraction: float,
    val_fraction_of_remaining: float,
) -> dict[str, list[FundusRecord]]:
    """Make a stable train/validation/test split within one domain."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if not 0 < val_fraction_of_remaining < 1:
        raise ValueError("val_fraction_of_remaining must be between zero and one")

    grouped: dict[str, list[FundusRecord]] = {}
    for record in records:
        grouped.setdefault(record.stratum, []).append(record)

    splits: dict[str, list[FundusRecord]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for stratum in sorted(grouped):
        group = sorted(grouped[stratum], key=lambda record: record.sample_id)
        rng.shuffle(group)
        test_count = round(len(group) * test_fraction)
        remaining = len(group) - test_count
        val_count = round(remaining * val_fraction_of_remaining)
        splits["test"].extend(group[:test_count])
        splits["val"].extend(group[test_count : test_count + val_count])
        splits["train"].extend(group[test_count + val_count :])

    for name in splits:
        splits[name] = sorted(splits[name], key=lambda record: record.sample_id)
    _validate_splits(splits, records)
    return splits


def provider_partition(
    records: Sequence[FundusRecord], seed: int, val_fraction: float
) -> dict[str, list[FundusRecord]]:
    """Keep a provider test set locked and split validation from provider train."""

    train_pool = [record for record in records if record.split_hint == "provider_train"]
    test = [record for record in records if record.split_hint == "provider_test"]
    if not train_pool or not test:
        raise DatasetLayoutError("Provider partition requires train and test split hints")
    grouped: dict[str, list[FundusRecord]] = {}
    for record in train_pool:
        grouped.setdefault(record.stratum, []).append(record)
    rng = random.Random(seed)
    train: list[FundusRecord] = []
    val: list[FundusRecord] = []
    for stratum in sorted(grouped):
        group = sorted(grouped[stratum], key=lambda record: record.sample_id)
        rng.shuffle(group)
        val_count = round(len(group) * val_fraction)
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    splits = {
        "train": sorted(train, key=lambda record: record.sample_id),
        "val": sorted(val, key=lambda record: record.sample_id),
        "test": sorted(test, key=lambda record: record.sample_id),
    }
    _validate_splits(splits, records)
    return splits


def _validate_splits(
    splits: dict[str, list[FundusRecord]], records: Sequence[FundusRecord]
) -> None:
    ids = {name: {record.sample_id for record in values} for name, values in splits.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise DatasetLayoutError("Train, validation, and test sample IDs must be disjoint")
    if ids["train"] | ids["val"] | ids["test"] != {
        record.sample_id for record in records
    }:
        raise DatasetLayoutError("Split does not cover every source record exactly once")
    if any(not values for values in splits.values()):
        raise DatasetLayoutError("Train, validation, and test splits must all be non-empty")


def _resize_and_pad(
    image: Image.Image, masks: np.ndarray, size: int
) -> tuple[Image.Image, list[Image.Image]]:
    width, height = image.size
    scale = size / max(width, height)
    resized = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    image = image.resize(resized, Image.Resampling.BILINEAR)
    mask_images = [
        Image.fromarray(channel * 255, mode="L").resize(
            resized, Image.Resampling.NEAREST
        )
        for channel in masks
    ]
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    image_canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    image_canvas.paste(image, offset)
    mask_canvases: list[Image.Image] = []
    for mask_image in mask_images:
        canvas = Image.new("L", (size, size), color=0)
        canvas.paste(mask_image, offset)
        mask_canvases.append(canvas)
    return image_canvas, mask_canvases


class FundusSegmentationDataset(Dataset):
    """PyTorch dataset exposing the normalized two-channel target contract."""

    def __init__(
        self,
        records: Sequence[FundusRecord],
        image_size: int,
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
        rotation_degrees: float = 10.0,
        brightness_contrast: float = 0.1,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.augment = augment
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rotation_degrees = rotation_degrees
        self.brightness_contrast = brightness_contrast

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        masks = decode_mask_channels(record)
        if image.size != (masks.shape[2], masks.shape[1]):
            raise DatasetLayoutError(
                f"Image/mask size mismatch for {record.sample_id}: "
                f"image={image.size}, mask={(masks.shape[2], masks.shape[1])}"
            )
        image, mask_images = _resize_and_pad(image, masks, self.image_size)

        if self.augment:
            if random.random() < self.horizontal_flip_probability:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask_images = [
                    mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    for mask in mask_images
                ]
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = image.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=(0, 0, 0),
                )
                mask_images = [
                    mask.rotate(
                        angle,
                        resample=Image.Resampling.NEAREST,
                        fillcolor=0,
                    )
                    for mask in mask_images
                ]
            if self.brightness_contrast > 0:
                span = self.brightness_contrast
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(1 - span, 1 + span)
                )
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(1 - span, 1 + span)
                )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_array = np.stack(
            [np.asarray(mask, dtype=np.uint8) >= 128 for mask in mask_images]
        ).astype(np.float32)
        # Independent nearest-neighbour resampling of the two channels can, on a thin
        # rim, leave a few cup pixels outside the disc. Repair instead of raising: an
        # augmentation-dependent crash would throw away every epoch since the last
        # checkpoint. The count travels in the metadata so it survives DataLoader
        # workers and can be reported rather than silently absorbed.
        cup_repair_pixels = int(np.count_nonzero(mask_array[1] > mask_array[0]))
        if cup_repair_pixels:
            mask_array[1] = np.minimum(mask_array[1], mask_array[0])
        mask_tensor = torch.from_numpy(mask_array)
        metadata = {
            "sample_id": record.sample_id,
            "domain": record.domain,
            "image_path": str(record.image_path),
            "cup_repair_pixels": cup_repair_pixels,
        }
        return image_tensor, mask_tensor, metadata


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compose_lodo_fold(
        domain_partitions: list,
        held_out_domain,
):
    pass
