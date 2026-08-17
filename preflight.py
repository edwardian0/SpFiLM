#!/usr/bin/env python3
"""Preflight checks for the Step 2 REFUGE baseline on CREATE.

Run on a compute node, never the login node:

    srun -p interruptible_gpu --gres=gpu:1 --time=0:10:00 \
      bash -l oncompute.sh python -u preflight.py 2>/dev/null

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

DEFAULT_CONFIG = "configs/stage2_refuge_create.json"

# Expected per-directory file counts under data_root, order-independent.
# REFUGE Training400: 40 glaucoma / 360 non-glaucoma, images and masks.
EXPECTED_COUNTS = [40, 40, 360, 360]

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


def check_data(config_path: Path) -> tuple[bool, Path | None]:
    hdr("config / data_root")
    if not config_path.exists():
        print(f"  FAIL: config not found at {config_path.resolve()}")
        return False, None
    print(f"  config                {config_path.resolve()}")

    cfg = json.loads(config_path.read_text())
    raw = find_key(cfg, "data_root")
    if raw is None:
        print("  FAIL: no 'data_root' key anywhere in the config")
        return False, None

    root = Path(str(raw)).expanduser()
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
        return False, None

    print("  exists                yes")

    counts: dict[Path, int] = {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        n = sum(1 for f in d.iterdir() if f.is_file() and not f.name.startswith("."))
        if n:
            counts[d.relative_to(root)] = n
    if not counts:
        print("  FAIL: data_root exists but contains no files in any subdirectory")
        return False, None

    print("  per-directory counts:")
    for d, n in counts.items():
        print(f"    {n:>6}  {d}")

    total = sum(counts.values())
    print(f"    {total:>6}  TOTAL")
    if sorted(counts.values()) != sorted(EXPECTED_COUNTS):
        warn(f"counts {sorted(counts.values())} != expected {sorted(EXPECTED_COUNTS)} "
             "— check this is Training400 and nothing is missing or duplicated")

    return True, root


# --------------------------------------------------------------------------
# 3. mask convention (soft check)
# --------------------------------------------------------------------------
def check_mask_values(root: Path) -> None:
    hdr("mask convention (soft)")
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        warn(f"cannot import numpy/PIL ({exc}); skipping mask check")
        return

    candidates = [p for p in root.rglob("*")
                  if p.suffix.lower() in {".bmp", ".png", ".gif", ".tif", ".tiff"}]
    if not candidates:
        warn("no mask-like files (.bmp/.png/.gif/.tif) found under data_root; skipping")
        return

    sample = sorted(candidates)[0]
    arr = np.array(Image.open(sample))
    vals = np.unique(arr)
    print(f"  sample                {sample.relative_to(root)}")
    print(f"  shape / dtype         {arr.shape} {arr.dtype}")
    print(f"  unique values         {vals[:10].tolist()}{' ...' if vals.size > 10 else ''}")
    if set(vals.tolist()) <= {0, 128, 255}:
        print("  convention            {0=cup, 128=rim, 255=background} as expected")
        cup_frac = float((arr == 0).mean())
        print(f"  cup fraction          {cup_frac:.4%}  (expect ~0.47% on average)")
    else:
        warn("values are not a subset of {0,128,255} — confirm this file is a mask "
             "and that data.py's inversion still applies")


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
    ap.add_argument("--out-parent", default=Path("artifacts/runs"), type=Path,
                    help="where checkpoints will be written, for the free-space check")
    ap.add_argument("--skip-amp", action="store_true",
                    help="skip the AMP numerics probe")
    args = ap.parse_args()

    print(f"host: {os.uname().nodename}")
    print(f"cwd:  {Path.cwd()}")

    results: dict[str, bool] = {}
    results["torch/cuda"] = check_torch()

    data_ok, root = check_data(args.config)
    results["data_root"] = data_ok
    if root is not None:
        check_mask_values(root)

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