from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data import DatasetLayoutError, FundusRecord, decode_mask_channels


MANIFEST_NAME = "roi_manifest.csv"
SPEC_NAME = "roi_cache.json"
MANIFEST_FIELDS = (
    "sample_id",
    "domain",
    "stratum",
    "split_hint",
    "mask_encoding",
    "source_image_path",
    "cache_path",
    "source_width",
    "source_height",
    "centre_x",
    "centre_y",
    "crop_x0",
    "crop_y0",
    "crop_size",
    "output_size",
    "pad_left",
    "pad_top",
    "pad_right",
    "pad_bottom",
    "native_disc_pixels",
    "roi_disc_pixels",
    "roi_cup_pixels",
    "roi_vcdr",
)


@dataclass(frozen=True)
class RoiCropSpec:
    """The locked region-of-interest contract: an 800px native crop at 256px."""

    crop_size: int = 800
    output_size: int = 256
    centre_source: str = "ground_truth_disc_centroid"

    def __post_init__(self) -> None:
        if self.crop_size <= 0 or self.output_size <= 0:
            raise ValueError("crop_size and output_size must both be positive")


@dataclass(frozen=True)
class RoiWindow:
    """One crop box in native pixel coordinates, with the padding it requires."""

    x0: int
    y0: int
    size: int
    centre_x: float
    centre_y: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    @property
    def padded(self) -> bool:
        return bool(self.pad_left or self.pad_top or self.pad_right or self.pad_bottom)


def disc_centroid(disc: np.ndarray) -> tuple[float, float]:
    """Return the ``(x, y)`` area centroid of the ground-truth disc channel."""

    rows, columns = np.nonzero(disc)
    if rows.size == 0:
        raise DatasetLayoutError("Cannot centre a region of interest on an empty disc")
    return float(columns.mean()), float(rows.mean())


def roi_window(disc: np.ndarray, spec: RoiCropSpec) -> RoiWindow:
    """Place a fixed-size window on the disc centroid, padding rather than sliding.

    Sliding the box back inside the frame would silently move the disc off centre
    for border cases, so the centroid stays at the crop centre and any shortfall
    is recorded as padding that the manifest can be audited on.
    """

    height, width = disc.shape
    centre_x, centre_y = disc_centroid(disc)
    half = spec.crop_size / 2.0
    x0 = int(round(centre_x - half))
    y0 = int(round(centre_y - half))
    return RoiWindow(
        x0=x0,
        y0=y0,
        size=spec.crop_size,
        centre_x=centre_x,
        centre_y=centre_y,
        pad_left=max(0, -x0),
        pad_top=max(0, -y0),
        pad_right=max(0, x0 + spec.crop_size - width),
        pad_bottom=max(0, y0 + spec.crop_size - height),
    )


def _vertical_extent(mask: np.ndarray) -> int:
    rows = np.nonzero(mask.any(axis=1))[0]
    return 0 if rows.size == 0 else int(rows.max() - rows.min() + 1)


def _crop_array(source: np.ndarray, window: RoiWindow) -> np.ndarray:
    height, width = source.shape[:2]
    canvas = np.zeros((window.size, window.size, *source.shape[2:]), dtype=source.dtype)
    source_x0 = max(0, window.x0)
    source_y0 = max(0, window.y0)
    source_x1 = min(width, window.x0 + window.size)
    source_y1 = min(height, window.y0 + window.size)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        raise DatasetLayoutError("Region of interest falls entirely outside the image")
    canvas[
        source_y0 - window.y0 : source_y1 - window.y0,
        source_x0 - window.x0 : source_x1 - window.x0,
    ] = source[source_y0:source_y1, source_x0:source_x1]
    return canvas


def crop_and_resize(
    record: FundusRecord, spec: RoiCropSpec
) -> tuple[np.ndarray, np.ndarray, RoiWindow]:
    """Crop the locked region at native resolution, then resize to the model input.

    Returns ``(image [S, S, 3] uint8, masks [2, S, S] uint8, window)`` using the
    same two-channel disc/cup contract as ``decode_mask_channels``.
    """

    masks = decode_mask_channels(record)
    with Image.open(record.image_path) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != masks.shape[1:]:
        raise DatasetLayoutError(
            f"Image/mask size mismatch for {record.sample_id}: "
            f"image={image.shape[:2]}, mask={masks.shape[1:]}"
        )

    window = roi_window(masks[0], spec)
    cropped_image = _crop_array(image, window)
    cropped_masks = np.stack([_crop_array(channel, window) for channel in masks])

    native_disc_pixels = int(masks[0].sum())
    if int(cropped_masks[0].sum()) < native_disc_pixels:
        raise DatasetLayoutError(
            f"Region of interest clipped the disc for {record.sample_id}: "
            f"{native_disc_pixels - int(cropped_masks[0].sum())} disc pixels fall "
            f"outside a {spec.crop_size}px window centred on the disc centroid"
        )

    output_shape = (spec.output_size, spec.output_size)
    resized_image = np.asarray(
        Image.fromarray(cropped_image).resize(
            output_shape, Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    resized_masks = np.stack(
        [
            np.asarray(
                Image.fromarray(channel * 255, mode="L").resize(
                    output_shape, Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
            >= 128
            for channel in cropped_masks
        ]
    ).astype(np.uint8)

    if np.any(resized_masks[1] & ~resized_masks[0]):
        raise DatasetLayoutError(
            f"Region-of-interest resize broke the cup-within-disc contract for "
            f"{record.sample_id}"
        )
    if not resized_masks[0].any() or not resized_masks[1].any():
        raise DatasetLayoutError(
            f"Region-of-interest crop produced an empty disc or cup for "
            f"{record.sample_id}"
        )
    return resized_image, resized_masks, window


def cache_path_for(cache_dir: Path, record: FundusRecord) -> Path:
    return cache_dir / record.domain / f"{record.sample_id}.npz"


def load_roi_sample(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image [S, S, 3] uint8, masks [2, S, S] uint8)`` from one cache file."""

    with np.load(Path(path)) as payload:
        return payload["image"], payload["masks"]


def build_roi_cache(
    records: Sequence[FundusRecord],
    cache_dir: str | Path,
    spec: RoiCropSpec = RoiCropSpec(),
    overwrite: bool = False,
) -> Path:
    """Write one npz per record plus a manifest describing every crop box.

    The manifest records the centroid, the box, and the padding for each sample,
    so a later detector-based region of interest can replace the ground-truth
    centroid without changing anything downstream of this cache.
    """

    if not records:
        raise DatasetLayoutError("Cannot build a region-of-interest cache with no records")
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for record in records:
        output_path = cache_path_for(cache_dir, record)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise DatasetLayoutError(
                f"{output_path} already exists; rerun with overwrite=True so every "
                "cached crop provably comes from one spec"
            )
        masks_native = decode_mask_channels(record)
        image, masks, window = crop_and_resize(record, spec)
        np.savez_compressed(output_path, image=image, masks=masks)
        with Image.open(record.image_path) as handle:
            source_width, source_height = handle.size
        disc_extent = _vertical_extent(masks[0])
        rows.append(
            {
                "sample_id": record.sample_id,
                "domain": record.domain,
                "stratum": record.stratum,
                "split_hint": record.split_hint or "",
                "mask_encoding": record.mask_encoding,
                "source_image_path": str(record.image_path),
                "cache_path": str(output_path),
                "source_width": source_width,
                "source_height": source_height,
                "centre_x": round(window.centre_x, 2),
                "centre_y": round(window.centre_y, 2),
                "crop_x0": window.x0,
                "crop_y0": window.y0,
                "crop_size": window.size,
                "output_size": spec.output_size,
                "pad_left": window.pad_left,
                "pad_top": window.pad_top,
                "pad_right": window.pad_right,
                "pad_bottom": window.pad_bottom,
                "native_disc_pixels": int(masks_native[0].sum()),
                "roi_disc_pixels": int(masks[0].sum()),
                "roi_cup_pixels": int(masks[1].sum()),
                "roi_vcdr": round(
                    _vertical_extent(masks[1]) / disc_extent if disc_extent else float("nan"),
                    4,
                ),
            }
        )

    manifest_path = cache_dir / MANIFEST_NAME
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    padded = sum(1 for row in rows if row["pad_left"] or row["pad_top"] or row["pad_right"] or row["pad_bottom"])
    (cache_dir / SPEC_NAME).write_text(
        json.dumps(
            {
                "spec": asdict(spec),
                "sample_count": len(rows),
                "domains": {
                    domain: sum(1 for row in rows if row["domain"] == domain)
                    for domain in sorted({str(row["domain"]) for row in rows})
                },
                "padded_sample_count": padded,
                "contract": {
                    "image": f"[{spec.output_size}, {spec.output_size}, 3] uint8 RGB",
                    "masks": f"[2, {spec.output_size}, {spec.output_size}] uint8",
                    "channel_0": "optic_disc including optic_cup",
                    "channel_1": "optic_cup",
                    "image_interpolation": "bilinear",
                    "mask_interpolation": "nearest",
                    "metric_frame": (
                        f"metrics computed in this {spec.output_size}px region-of-interest "
                        "frame, not resampled back to native resolution"
                    ),
                    "leakage_note": (
                        "the crop centre is the ground-truth disc centroid; this is a "
                        "known ground-truth dependency at test time and must be replaced "
                        "by a detector before any cross-domain claim"
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def read_roi_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    with Path(manifest_path).expanduser().resolve().open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def verify_roi_cache(manifest_path: str | Path) -> dict[str, object]:
    """Reopen every cached file and confirm shapes, dtypes and the mask contract."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    rows = read_roi_manifest(manifest_path)
    if not rows:
        raise DatasetLayoutError(f"{manifest_path} lists no cached samples")
    spec = json.loads((manifest_path.parent / SPEC_NAME).read_text(encoding="utf-8"))
    output_size = int(spec["spec"]["output_size"])

    missing: list[str] = []
    contract_violations: list[str] = []
    for row in rows:
        cache_path = Path(row["cache_path"])
        if not cache_path.is_file():
            missing.append(row["sample_id"])
            continue
        image, masks = load_roi_sample(cache_path)
        if image.shape != (output_size, output_size, 3) or image.dtype != np.uint8:
            contract_violations.append(f"{row['sample_id']}: image {image.shape} {image.dtype}")
        elif masks.shape != (2, output_size, output_size) or masks.dtype != np.uint8:
            contract_violations.append(f"{row['sample_id']}: masks {masks.shape} {masks.dtype}")
        elif set(np.unique(masks).tolist()) - {0, 1}:
            contract_violations.append(f"{row['sample_id']}: masks are not binary")
        elif np.any(masks[1] & ~masks[0]):
            contract_violations.append(f"{row['sample_id']}: cup outside disc")
        elif not masks[0].any() or not masks[1].any():
            contract_violations.append(f"{row['sample_id']}: empty disc or cup")

    clipped = [
        row["sample_id"]
        for row in rows
        if int(row["roi_disc_pixels"]) == 0 or int(row["native_disc_pixels"]) == 0
    ]
    padded = [row["sample_id"] for row in rows if any(int(row[key]) for key in ("pad_left", "pad_top", "pad_right", "pad_bottom"))]
    return {
        "manifest": str(manifest_path),
        "sample_count": len(rows),
        "missing_cache_files": missing,
        "contract_violations": contract_violations,
        "clipped_disc_count": len(clipped),
        "padded_samples": padded,
        "roi_vcdr_median": float(
            np.median([float(row["roi_vcdr"]) for row in rows])
        ),
        "status": "ok" if not missing and not contract_violations else "failed",
    }


def save_roi_contact_sheet(
    manifest_path: str | Path,
    output_path: str | Path,
    count: int = 12,
    seed: int = 42,
) -> Path:
    """Draw cached crops with disc and cup contours so the boxes can be checked by eye."""

    rows = read_roi_manifest(manifest_path)
    if not rows:
        raise DatasetLayoutError(f"{manifest_path} lists no cached samples")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = random.Random(seed).sample(rows, k=min(count, len(rows)))

    columns = 4
    sheet_rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(
        sheet_rows, columns, figsize=(3.4 * columns, 3.6 * sheet_rows), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")

    for index, row in enumerate(selected):
        image, masks = load_roi_sample(row["cache_path"])
        overlay = image.astype(np.float32) / 255.0
        rim = masks[0].astype(bool) & ~masks[1].astype(bool)
        cup = masks[1].astype(bool)
        overlay[rim] = 0.6 * overlay[rim] + 0.4 * np.array([0.0, 1.0, 0.0])
        overlay[cup] = 0.6 * overlay[cup] + 0.4 * np.array([1.0, 0.2, 0.0])
        axis = axes[index // columns, index % columns]
        axis.imshow(np.clip(overlay, 0, 1))
        axis.axhline(image.shape[0] / 2, color="white", linewidth=0.5, alpha=0.6)
        axis.axvline(image.shape[1] / 2, color="white", linewidth=0.5, alpha=0.6)
        axis.set_title(
            f"{row['sample_id']}  vCDR={float(row['roi_vcdr']):.2f}\n"
            f"box=({row['crop_x0']},{row['crop_y0']}) {row['crop_size']}px  "
            f"pad={row['pad_left']},{row['pad_top']},{row['pad_right']},{row['pad_bottom']}",
            fontsize=8,
        )
        axis.axis("off")

    figure.suptitle(
        "Region-of-interest crops - green: rim, orange: cup, crosshair: crop centre",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return output_path
