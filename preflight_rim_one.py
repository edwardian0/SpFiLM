#!/usr/bin/env python3
"""Fail-closed preflight for the RIM-ONE-DL Step 2 baseline.

Run unskipped on a CREATE GPU compute node before ``sbatch``. ``--skip-runtime``
exists only for data/manifest verification on a machine without CUDA.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_rimone_create.json"
DEFAULT_OUT_PARENT = PROJECT_ROOT / "artifacts" / "runs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import (  # noqa: E402
    RIM_ONE_DL_IMAGE_COUNT,
    RIM_ONE_DL_MASK_COUNT,
    RIM_ONE_DL_RELEASE_COUNTS,
    RIM_ONE_DL_SOURCE_CUP_REPAIRS,
    RIM_ONE_DL_SPLIT_COUNTS,
    audit_records,
    decode_mask_channels,
    discover_rim_one_dl,
)
from spfilm.engine import (  # noqa: E402
    Stage2Config,
    build_splits,
)


def hdr(name: str) -> None:
    print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def check_torch_cuda() -> bool:
    hdr("torch / CUDA")
    try:
        import torch
    except Exception as exc:
        print(f"  FAIL: cannot import torch ({exc})")
        return False
    print(f"  torch                 {torch.__version__}")
    print(f"  torch.version.cuda    {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"  cuda.is_available()   {available}")
    if torch.version.cuda is None or not available:
        print("  FAIL: run this preflight on a GPU compute node with the CUDA wheel")
        return False
    properties = torch.cuda.get_device_properties(0)
    print(f"  device                {properties.name}")
    print(f"  capability            sm_{properties.major}{properties.minor}")
    print(f"  total VRAM            {properties.total_memory / 1024**3:.1f} GiB")
    visible_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    print(f"  visible CPUs          {visible_cpus}")
    return True


def check_amp_numerics() -> bool:
    hdr("AMP + InstanceNorm + BCE/soft Dice")
    import torch

    from spfilm.losses import BCEDiceLoss
    from spfilm.model import PlainUNet

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = PlainUNet(base_channels=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    inputs = torch.randn(1, 3, 128, 128, device=device)
    targets = torch.zeros(1, 2, 128, 128, device=device)
    targets[:, 0, 24:104, 24:104] = 1
    targets[:, 1, 50:78, 50:78] = 1
    criterion = BCEDiceLoss()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        loss = criterion(model(inputs), targets)
    print(f"  probe loss            {float(loss):.6f}")
    if not torch.isfinite(loss):
        print("  FAIL: non-finite loss under CUDA autocast")
        return False
    scaler.scale(loss).backward()
    gradients_finite = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    print(f"  gradients finite      {gradients_finite}")
    return gradients_finite


def check_config_and_data(
    config_path: Path, data_root_override: Path | None
) -> tuple[bool, Stage2Config | None]:
    hdr("config / discovery / manifest")
    if not config_path.is_file():
        print(f"  FAIL: config is missing: {config_path}")
        return False, None
    try:
        config = Stage2Config.from_json(config_path)
    except Exception as exc:
        print(f"  FAIL: cannot load config: {exc}")
        return False, None
    if config.dataset != "rim_one_dl":
        print(f"  FAIL: expected dataset='rim_one_dl', got {config.dataset!r}")
        return False, config
    if config.rim_manifest is None:
        print("  FAIL: config has no rim_manifest split path")
        return False, config

    configured_root = _resolve(PROJECT_ROOT, config.data_root)
    data_root = (
        data_root_override.expanduser().resolve()
        if data_root_override is not None
        else configured_root
    )
    manifest_path = _resolve(PROJECT_ROOT, config.rim_manifest)
    print(f"  config                {config_path.resolve()}")
    print(f"  data_root (config)    {configured_root}")
    if data_root_override is not None:
        print(f"  data_root (override)  {data_root}")
    print(f"  image tree            {data_root / 'RIM-ONE_DL_images' / 'partitioned_by_hospital'}")
    print(f"  mask tree             {data_root / 'RIM-ONE-DL_masks'}")
    print(f"  split manifest        {manifest_path}")

    try:
        records = discover_rim_one_dl(data_root)
        splits = build_splits(config, records, PROJECT_ROOT)
        audit = audit_records(records)
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False, config

    print(f"  paired images         {len(records)} (expected {RIM_ONE_DL_IMAGE_COUNT})")
    print(
        f"  paired mask PNGs      {sum(len(record.mask_paths) for record in records)} "
        f"(expected {RIM_ONE_DL_MASK_COUNT})"
    )
    hospital_class = Counter(
        (record.hospital_split, record.diagnosis_class) for record in records
    )
    print(f"  hospital/class        {dict(sorted(hospital_class.items()))}")
    print(
        "  release totals        "
        f"{dict(sorted(Counter(record.release_prefix for record in records).items()))}"
    )
    split_counts = {split: len(rows) for split, rows in splits.items()}
    print(f"  manifest counts       {split_counts}")
    if split_counts != RIM_ONE_DL_SPLIT_COUNTS:
        print(
            f"  FAIL: manifest counts differ from {RIM_ONE_DL_SPLIT_COUNTS}"
        )
        return False, config
    for split in ("train", "val", "test"):
        release = dict(
            sorted(Counter(record.release_prefix for record in splits[split]).items())
        )
        diagnosis = dict(
            sorted(Counter(record.diagnosis_class for record in splits[split]).items())
        )
        joint = dict(
            sorted(Counter(record.stratum for record in splits[split]).items())
        )
        print(
            f"  {split:<20}release={release} class={diagnosis} joint={joint}"
        )
        if set(release) != set(RIM_ONE_DL_RELEASE_COUNTS):
            print(f"  FAIL: {split} does not contain r1, r2, and r3")
            return False, config
    print(f"  duplicate image hash groups  {audit['duplicate_image_hash_groups']}")
    return True, config


def check_masks(data_root: Path) -> bool:
    hdr("polarity / containment / dimensions")
    try:
        import numpy as np
        from PIL import Image

        records = discover_rim_one_dl(data_root)
        high_fractions = {"disc": [], "cup": []}
        low_outside_pixels = 0
        raw_repairs: dict[str, int] = {}
        resolutions: Counter[tuple[int, int]] = Counter()
        modes: Counter[tuple[str, str]] = Counter()
        dtypes: Counter[tuple[str, str]] = Counter()
        values = {"disc": set(), "cup": set()}
        for record in records:
            with Image.open(record.image_path) as image:
                image_size = image.size
            resolutions[image_size] += 1
            raw = {}
            for label, path in (
                ("disc", record.disc_mask_path),
                ("cup", record.cup_mask_path),
            ):
                if path is None:
                    raise RuntimeError(f"{record.sample_id} has no {label} mask")
                with Image.open(path) as mask_image:
                    array = np.asarray(mask_image)
                    mask_size = mask_image.size
                    modes[(label, mask_image.mode)] += 1
                if mask_size != image_size:
                    raise RuntimeError(
                        f"{record.sample_id} {label} size {mask_size} != {image_size}"
                    )
                dtypes[(label, str(array.dtype))] += 1
                values[label].update(int(value) for value in np.unique(array))
                raw[label] = array >= 128
                high_fractions[label].append(float(raw[label].mean()))
            outside = raw["cup"] & ~raw["disc"]
            if outside.any():
                raw_repairs[record.sample_id] = int(outside.sum())
            low_outside_pixels += int(np.count_nonzero(~raw["cup"] & raw["disc"]))
            decoded = decode_mask_channels(record)
            if set(np.unique(decoded).tolist()) - {0, 1}:
                raise RuntimeError(f"{record.sample_id} decoded non-binary masks")
            if np.any(decoded[1] > decoded[0]):
                raise RuntimeError(f"{record.sample_id} decoded cup outside disc")
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        return False

    print(f"  mask modes            {dict(sorted(modes.items()))}")
    print(f"  mask dtypes           {dict(sorted(dtypes.items()))}")
    print(f"  unique values         { {key: sorted(item) for key, item in values.items()} }")
    for label in ("cup", "disc"):
        array = np.asarray(high_fractions[label])
        print(
            f"  {label + ' white fraction':<22}min={array.min():.6f} "
            f"median={np.median(array):.6f} mean={array.mean():.6f} "
            f"max={array.max():.6f}"
        )
    print("  polarity              white/high pixels are foreground (no inversion)")
    print(f"  raw cup repairs       {raw_repairs}")
    print(f"  raw repair pixels     {sum(raw_repairs.values())}")
    print(f"  inverted containment  {low_outside_pixels} violating pixels")
    print(f"  checked dimensions    {len(records)} image/disc/cup triplets")
    print(
        f"  native resolutions    {len(resolutions)} distinct; "
        f"min={min(resolutions)} max={max(resolutions)}"
    )
    if raw_repairs != RIM_ONE_DL_SOURCE_CUP_REPAIRS:
        print(
            "  FAIL: source containment defects differ from the pinned Phase 0 result"
        )
        return False
    if values != {"disc": {0, 255}, "cup": {0, 255}}:
        print("  FAIL: masks are not exactly binary 0/255")
        return False
    return True


def check_disk(out_parent: Path) -> None:
    hdr("disk")
    target = out_parent.resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    free_gb = shutil.disk_usage(target).free / 1024**3
    print(f"  checked               {target}")
    print(f"  free                  {free_gb:.1f} GiB")
    if free_gb < 10:
        print("  WARN: under 10 GiB free")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="explicit local verification override; omit on CREATE",
    )
    parser.add_argument("--out-parent", type=Path, default=DEFAULT_OUT_PARENT)
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="skip CUDA and AMP only; data/manifest gates still run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"host: {os.uname().nodename}")
    print(f"cwd:  {Path.cwd()}")
    results: dict[str, bool] = {}
    data_ok, config = check_config_and_data(args.config.resolve(), args.data_root)
    results["discovery/manifest"] = data_ok
    if config is not None:
        configured_root = _resolve(PROJECT_ROOT, config.data_root)
        data_root = (
            args.data_root.expanduser().resolve()
            if args.data_root is not None
            else configured_root
        )
        results["masks/polarity"] = check_masks(data_root)

    if args.skip_runtime:
        hdr("runtime probes skipped")
        print("  SKIP: rerun without --skip-runtime on a CREATE GPU compute node")
    else:
        results["torch/CUDA"] = check_torch_cuda()
        if results["torch/CUDA"]:
            results["AMP numerics"] = check_amp_numerics()
    check_disk(args.out_parent)

    hdr("summary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(results.values()):
        print("\nAt least one hard gate failed. Do not submit.")
        return 1
    if args.skip_runtime:
        print("\nAll data gates passed; CUDA runtime gates remain to be run on CREATE.")
    else:
        print("\nAll hard gates passed. Safe to sbatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
