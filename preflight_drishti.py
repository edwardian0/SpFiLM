#!/usr/bin/env python3
"""Preflight checks for a Step 2 fundus baseline on CREATE.

Run this explicit gate on a compute node before ``sbatch``. It is deliberately
not invoked from ``submit_drishti_s2.sh``, because that would run only after the
job had already been submitted:

    srun -p interruptible_gpu --gres=gpu:1 --time=0:10:00 \
      bash -l /users/k23123868/edward/spfilm/oncompute.sh \
      python -u /users/k23123868/edward/spfilm/preflight_drishti.py \
      --config /users/k23123868/edward/spfilm/configs/stage2_drishti_create.json

Exit 0 = every hard gate passed, safe to sbatch.
Exit 1 = at least one hard gate failed; the message says which.

WARN lines are informational and do not affect the exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage2_drishti_create.json"
DEFAULT_OUT_PARENT = PROJECT_ROOT / "artifacts" / "runs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Expected paired record totals after dataset-specific discovery. Drishti is the
# provider's 50-image training pool plus its locked 51-image test pool.
EXPECTED_RECORD_COUNTS = {"refuge": 400, "drishti": 101}

WARNINGS: list[str] = []


def hdr(name: str) -> None:
    print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN: {msg}")


# --------------------------------------------------------------------------
# 1. torch / CUDA
# --------------------------------------------------------------------------
def check_torch() -> bool:
    hdr("torch / CUDA")
    try:
        import torch
    except Exception as exc:
        print(f"  FAIL: cannot import torch ({exc}) — wrong conda env?")
        return False

    print(f"  torch                 {torch.__version__}")
    print(f"  torch.version.cuda    {torch.version.cuda}")

    ok = True
    if torch.version.cuda is None:
        print("  FAIL: CPU-only wheel. Reinstall torch from the CUDA index "
              "inside the spfilm env before doing anything else.")
        ok = False

    avail = torch.cuda.is_available()
    print(f"  cuda.is_available()   {avail}")
    if not avail:
        print("  FAIL: no visible GPU. Are you on a compute node with --gres=gpu:1?")
        ok = False
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"  device                {props.name}")
        print(f"  capability            sm_{props.major}{props.minor}")
        print(f"  total VRAM            {props.total_memory / 1024**3:.1f} GiB")

    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"  visible CPUs          {ncpu}")
    return ok


# --------------------------------------------------------------------------
# 2. config + data_root
# --------------------------------------------------------------------------
def find_key(obj, key):
    """Depth-first search for `key` anywhere in a nested dict/list config."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def resolve_case_insensitive(path: Path) -> Path | None:
    """Walk `path` component by component matching case-insensitively.

    Returns the real on-disk path if one exists under different casing.
    This is the check that would have caught Refuge/REFUGE before Linux did.
    """
    parts = list(path.parts)
    if path.is_absolute():
        cur = Path(parts[0])
        parts = parts[1:]
    else:
        cur = Path(".")
    for part in parts:
        if not cur.is_dir():
            return None
        matches = [c for c in cur.iterdir() if c.name.lower() == part.lower()]
        if not matches:
            return None
        cur = matches[0]
    return cur


def check_data(
    config_path: Path,
) -> tuple[bool, Path | None, str | None, list[Any]]:
    hdr("config / data_root")
    if not config_path.exists():
        print(f"  FAIL: config not found at {config_path.resolve()}")
        return False, None, None, []
    print(f"  config                {config_path.resolve()}")

    cfg = json.loads(config_path.read_text())
    dataset = find_key(cfg, "dataset")
    if dataset not in EXPECTED_RECORD_COUNTS:
        print(
            "  FAIL: unsupported config dataset "
            f"{dataset!r}; expected one of {sorted(EXPECTED_RECORD_COUNTS)}"
        )
        return False, None, None, []
    raw = find_key(cfg, "data_root")
    if raw is None:
        print("  FAIL: no 'data_root' key anywhere in the config")
        return False, None, dataset, []

    root = Path(str(raw)).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    print(f"  dataset               {dataset}")
    print(f"  data_root (config)    {root}")

    if not root.exists():
        real = resolve_case_insensitive(root)
        if real is not None:
            print(f"  FAIL: path does not exist, but a case variant does:")
            print(f"        on disk -> {real}")
            print("        Fix the config on the Mac and re-sync. Do not patch it on CREATE.")
        else:
            print("  FAIL: data_root does not exist and no case variant found.")
            parent = root.parent
            if parent.is_dir():
                print(f"        contents of {parent}:")
                for c in sorted(parent.iterdir())[:20]:
                    print(f"          {c.name}")
        return False, None, dataset, []

    print("  exists                yes")

    counts: dict[Path, int] = {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        n = sum(1 for f in d.iterdir() if f.is_file() and not f.name.startswith("."))
        if n:
            counts[d.relative_to(root)] = n
    if not counts:
        print("  FAIL: data_root exists but contains no files in any subdirectory")
        return False, None, dataset, []

    print("  per-directory counts:")
    for d, n in counts.items():
        print(f"    {n:>6}  {d}")

    total = sum(counts.values())
    print(f"    {total:>6}  TOTAL")

    try:
        from spfilm.data import discover_drishti, discover_refuge_training

        discover = {
            "refuge": discover_refuge_training,
            "drishti": discover_drishti,
        }[dataset]
        records = discover(root)
    except Exception as exc:
        print(f"  FAIL: {dataset} dataset discovery failed: {exc}")
        return False, root, dataset, []

    expected = EXPECTED_RECORD_COUNTS[dataset]
    actual = len(records)
    print(f"  paired records        {actual} (expected {expected})")
    if actual != expected:
        print(
            f"  FAIL: {dataset} expected {expected} paired records, found {actual}"
        )
        return False, root, dataset, records

    return True, root, dataset, records


# --------------------------------------------------------------------------
# 3. mask convention (soft check)
# --------------------------------------------------------------------------
def check_mask_values(root: Path, dataset: str, records: list[Any]) -> None:
    hdr("mask convention (soft)")
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        warn(f"cannot import numpy/PIL ({exc}); skipping mask check")
        return

    if not records:
        warn("dataset discovery returned no records; skipping mask check")
        return

    record = sorted(records, key=lambda item: item.sample_id)[0]
    paths = record.mask_paths
    labels = ("combined",) if len(paths) == 1 else ("disc", "cup")
    allowed_by_dataset = {
        "refuge": {0, 128, 255},
        "drishti": {0, 64, 128, 191, 255},
    }
    allowed = allowed_by_dataset[dataset]
    print(f"  sample id             {record.sample_id}")
    for label, sample in zip(labels, paths):
        arr = np.array(Image.open(sample))
        vals = np.unique(arr)
        print(f"  {label + ' mask':<22}{sample.relative_to(root)}")
        print(f"    shape / dtype       {arr.shape} {arr.dtype}")
        print(
            f"    unique values       {vals[:10].tolist()}"
            f"{' ...' if vals.size > 10 else ''}"
        )
        unexpected = set(vals.tolist()) - allowed
        if unexpected:
            warn(
                f"{dataset} {label} mask contains values outside "
                f"{sorted(allowed)}: {sorted(unexpected)}"
            )
    if dataset == "refuge":
        print("  convention            {0=cup, 128=rim, 255=background}")
    else:
        print("  convention            separate disc/cup soft maps; consensus >=191")


# --------------------------------------------------------------------------
# 4. AMP + InstanceNorm + soft Dice numerics
# --------------------------------------------------------------------------
def soft_dice(logits, target, eps: float = 1e-6):
    import torch
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    num = 2.0 * (probs * target).sum(dims)
    den = probs.sum(dims) + target.sum(dims)
    return 1.0 - ((num + eps) / (den + eps)).mean()


def check_amp_numerics() -> bool:
    hdr("AMP + InstanceNorm + soft Dice")
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return True

    torch.manual_seed(0)
    dev = torch.device("cuda")
    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.InstanceNorm2d(16, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.InstanceNorm2d(16, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 2, 1),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    try:
        scaler = torch.amp.GradScaler("cuda")
        autocast = lambda: torch.autocast("cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
        autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)

    x = torch.randn(2, 3, 512, 512, device=dev)
    y = torch.zeros(2, 2, 512, 512, device=dev)
    y[:, 0, 120:400, 120:400] = 1.0          # disc, ~30% of pixels
    y[:, 1, 248:284, 248:284] = 1.0          # cup, ~0.5% of pixels — the risky denominator
    bce = nn.BCEWithLogitsLoss()

    torch.cuda.reset_peak_memory_stats()
    ok = True
    for step in range(5):
        opt.zero_grad(set_to_none=True)
        with autocast():
            logits = net(x)
            loss = bce(logits, y) + soft_dice(logits, y)
        if not torch.isfinite(loss):
            print(f"  FAIL: loss is {loss.item()} at step {step} under fp16 autocast.")
            print("        Fix: compute the Dice term in float32 (cast logits outside "
                  "autocast), or disable AMP for today's run — 32 steps/epoch does not need it.")
            ok = False
            break
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        print(f"  step {step}  loss={loss.item():.4f}  scale={scaler.get_scale():.0f}")

    if ok:
        grads_finite = all(
            torch.isfinite(p.grad).all().item()
            for p in net.parameters() if p.grad is not None
        )
        print(f"  gradients finite      {grads_finite}")
        if not grads_finite:
            print("  FAIL: non-finite gradients after 5 steps")
            ok = False

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  peak VRAM (probe net) {peak:.2f} GiB  — probe only, not your U-Net")
    return ok


# --------------------------------------------------------------------------
# 5. output disk space
# --------------------------------------------------------------------------
def check_disk(out_parent: Path) -> None:
    hdr("disk")
    target = out_parent
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1024**3
    print(f"  checked               {target}")
    print(f"  free                  {free_gb:.1f} GiB")
    if free_gb < 10:
        warn(f"under 10 GiB free at {target}; move --out-dir to /scratch before submitting")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    ap.add_argument("--out-parent", default=DEFAULT_OUT_PARENT, type=Path,
                    help="where checkpoints will be written, for the free-space check")
    ap.add_argument("--skip-amp", action="store_true",
                    help="skip the AMP numerics probe")
    args = ap.parse_args()

    print(f"host: {os.uname().nodename}")
    print(f"cwd:  {Path.cwd()}")

    results: dict[str, bool] = {}
    results["torch/cuda"] = check_torch()

    data_ok, root, dataset, records = check_data(args.config)
    results["data_root"] = data_ok
    if root is not None and dataset is not None:
        check_mask_values(root, dataset, records)

    if args.skip_amp or not results["torch/cuda"]:
        print("\n=== AMP probe skipped " + "=" * 40)
    else:
        results["amp numerics"] = check_amp_numerics()

    check_disk(args.out_parent)

    hdr("summary")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for w in WARNINGS:
        print(f"  WARN  {w}")

    if all(results.values()):
        print("\nAll hard gates passed. Safe to sbatch.")
        return 0
    print("\nAt least one hard gate failed. Do not submit yet.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
