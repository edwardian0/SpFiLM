#!/usr/bin/env python3
"""Train on every active domain at once, test on each domain separately.

The supervisor's third Step 4 protocol (2026-09-12): one model per seed is
trained on the pooled budgeted training partitions of the active domains --
with the true domain code for the Global FiLM arm, codes 0, 1, 2 -- and scored
on each domain's own 50-image budgeted test partition with that domain's own
code. Nothing is held out. Compared with leave-one-domain-out this asks whether
conditioning helps when the camera is *known*, and the per-domain fixed-code
sweep asks whether the network uses the code at all (the wrong-code penalty).

Same locked budgeted manifest, backbone, budget, seeds, augmentation, optimiser
and schedule as the LODO arms; the plain and FiLM configs differ only in ``arm``.
Each domain's test images are the same 50 every other Step 4 arm scores.
RIM-ONE-DL stays configured (the manifests cover it) but is inactive.

    python run_stage4_all_domains.py --config configs/stage4_all_domains_global_film_3dom.json check --skip-mask-audit
    python run_stage4_all_domains.py --config configs/stage4_all_domains_global_film_3dom.json run --seed 42 --smoke --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.all_domains import (  # noqa: E402
    ALL_DOMAINS_PROTOCOL_NAME,
    AllDomainsFold,
    AllDomainsFoldError,
    all_domains_fold_splits,
    compose_all_domains_fold,
    engine_splits,
    select_all_domains_smoke_views,
)
from spfilm.data import FundusRecord  # noqa: E402
from spfilm.engine import (  # noqa: E402
    RESUME_STATE_FILENAME,
    choose_device,
    run_experiment,
)
from spfilm.lodo import (  # noqa: E402
    Domain,
    LodoManifestError,
    SampleKey,
    load_lodo_manifest,
)
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    SingleSourceManifestError,
    load_single_source_manifest,
)
from spfilm.stage3 import (  # noqa: E402
    Stage3ConfigError,
    Stage3DataError,
    audit_lodo_domains,
    discover_lodo_records,
    resolve_manifest_records,
    resolve_project_output,
)
from spfilm.stage3_single_source import (  # noqa: E402
    Stage3SingleSourceConfig,
    parent_lodo_manifest_path,
    single_source_manifest_path,
    validate_single_source_manifest_against_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage4_all_domains_global_film_3dom.json"
CONFIG_STAGE = ALL_DOMAINS_PROTOCOL_NAME
CONFIG_DOMAINS_KEY = "active_domains"
ALL_DOMAINS_SPLIT_POLICY = (
    "train on all active domains: pooled budgeted train and val partitions of "
    "every active domain; each active domain's budgeted test partition scored "
    "separately with its own code; inactive domains excluded"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one model on every active domain and test on each domain "
            "separately (plain or Global FiLM arm)"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Stage 4 train-on-all JSON config (place before the command)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Revalidate discovery, config, shared manifest, and decoded masks"
    )
    check_parser.add_argument("--skip-mask-audit", action="store_true")

    run_parser = subparsers.add_parser(
        "run", help="Train one configured seed or every configured seed"
    )
    run_parser.add_argument("--seed", type=int, help="Configured run seed")
    run_parser.add_argument("--all", action="store_true", help="Run every configured seed")
    run_parser.add_argument(
        "--smoke",
        action="store_true",
        help="One-epoch, 128px plumbing rehearsal; never a scientific result",
    )
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    run_parser.add_argument(
        "--out-dir", type=Path, help="Exact base output directory for a single run"
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=15,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return f"{commit}{'-dirty' if dirty else ''}"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


def all_domains_fold(
    manifest: SingleSourceManifest, config: Stage3SingleSourceConfig
) -> AllDomainsFold:
    return compose_all_domains_fold(manifest.budgeted_partitions, config.active_domains)


def _load_locked_runtime(
    config: Stage3SingleSourceConfig,
) -> tuple[SingleSourceManifest, dict[Domain, list[FundusRecord]], dict[SampleKey, FundusRecord], Path]:
    manifest_path = single_source_manifest_path(config, PROJECT_ROOT)
    if not manifest_path.is_file():
        raise Stage3DataError(
            f"Shared budgeted manifest is missing: {manifest_path}. It is "
            "produced by run_stage3_lodo_1_3.py prepare and must be committed."
        )
    manifest = load_single_source_manifest(manifest_path)
    records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
    validate_single_source_manifest_against_config(
        config, manifest, records_by_domain, PROJECT_ROOT
    )
    parent_manifest = load_lodo_manifest(parent_lodo_manifest_path(config, PROJECT_ROOT))
    records_by_key = resolve_manifest_records(parent_manifest, records_by_domain)
    return manifest, records_by_domain, records_by_key, manifest_path


def _require_arm_policy(config: Stage3SingleSourceConfig) -> None:
    if config.arm != "plain" and config.test_conditioning != "oracle":
        raise Stage3ConfigError(
            "Train-on-all tests each domain with its own known code, so the "
            "film block must set test_conditioning='oracle'; got "
            f"{config.test_conditioning!r}"
        )


def check(config: Stage3SingleSourceConfig, skip_mask_audit: bool) -> int:
    _require_arm_policy(config)
    manifest, records_by_domain, _, manifest_path = _load_locked_runtime(config)
    fold = all_domains_fold(manifest, config)
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    report: dict[str, object] = {
        "status": "OK",
        "protocol": ALL_DOMAINS_PROTOCOL_NAME,
        "arm": config.arm,
        "active_domains": [d.value for d in config.active_domains],
        "inactive_domains": [
            p.domain.value for p in manifest.budgeted_partitions
            if p.domain not in set(config.active_domains)
        ],
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": _sha256(parent_path),
        "discovered_counts": {
            d.value: len(r) for d, r in sorted(records_by_domain.items())
        },
        "fold": {
            "train": len(fold.train),
            "val": len(fold.val),
            "test_by_domain": {d.value: len(t) for d, t in fold.tests},
        },
    }
    report["mask_audit"] = (
        "SKIPPED by explicit flag" if skip_mask_audit else audit_lodo_domains(records_by_domain)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _selected_seeds(config: Stage3SingleSourceConfig, args: argparse.Namespace) -> list[int]:
    if args.all:
        if args.seed is not None:
            raise Stage3ConfigError("--all cannot be combined with --seed")
        if args.out_dir is not None:
            raise Stage3ConfigError("--out-dir is only valid for a single run")
        return list(config.run_seeds)
    if args.seed is None:
        raise Stage3ConfigError("run requires --seed, or explicit --all")
    if args.seed not in config.run_seeds:
        raise Stage3ConfigError(
            f"Seed {args.seed} is not one of the configured seeds {list(config.run_seeds)}"
        )
    return [args.seed]


def _run_output_dir(
    config: Stage3SingleSourceConfig, seed: int, explicit_output: Path | None
) -> Path:
    if explicit_output is not None:
        return resolve_project_output(PROJECT_ROOT, str(explicit_output))
    return resolve_project_output(PROJECT_ROOT, config.output_dir) / f"seed_{seed}"


def _require_fresh_output(base_output: Path, smoke: bool) -> Path:
    actual_output = base_output.with_name(f"{base_output.name}_smoke") if smoke else base_output
    if actual_output.exists() and any(actual_output.iterdir()):
        if (actual_output / RESUME_STATE_FILENAME).is_file():
            return actual_output
        raise Stage3DataError(
            f"Refusing to overwrite non-empty run directory {actual_output}; choose a new --out-dir"
        )
    return actual_output


def _run_one(
    config: Stage3SingleSourceConfig,
    config_path: Path,
    manifest: SingleSourceManifest,
    records_by_key: dict[SampleKey, FundusRecord],
    manifest_path: Path,
    seed: int,
    smoke: bool,
    requested_device: str | None,
    explicit_output: Path | None,
) -> dict[str, Any]:
    fold = all_domains_fold(manifest, config)
    locked_views = all_domains_fold_splits(fold, records_by_key)
    executed_views = select_all_domains_smoke_views(locked_views) if smoke else locked_views
    splits, extra_test_sets = engine_splits(executed_views, fold.domains)
    records = [
        record for name in ("train", "val", "test") for record in splits[name]
    ]

    base_output = _run_output_dir(config, seed, explicit_output)
    actual_output = _require_fresh_output(base_output, smoke)
    relative_output = base_output.relative_to(PROJECT_ROOT)
    # training_config keys dataset/data_root off one domain; with explicit
    # records and splits those fields are provenance only.
    engine_config = replace(
        config.training_config(
            fold.domains[0], seed, str(relative_output), requested_device=requested_device
        ),
        experiment_name=f"{config.experiment_name}_seed_{seed}",
    )
    device = choose_device(engine_config.requested_device)
    active = [d.value for d in fold.domains]
    print(
        f"Train-on-all | arm={config.arm} | train on {len(active)}, test on each | "
        f"seed={seed} | device={device} | {'SMOKE' if smoke else 'FULL'}",
        flush=True,
    )
    print(
        f"active domains={active} | locked train={len(locked_views['train'])} "
        f"val={len(locked_views['val'])} | tests: "
        + ", ".join(f"{d.value}={len(locked_views[d.value])}" for d in fold.domains),
        flush=True,
    )
    if smoke:
        print(
            "smoke subset "
            + ", ".join(f"{name}={len(executed_views[name])}" for name in executed_views)
            + "; this output is not a scientific result",
            flush=True,
        )

    report = run_experiment(
        engine_config,
        PROJECT_ROOT,
        smoke=smoke,
        records=records,
        split_records=splits,
        split_policy=ALL_DOMAINS_SPLIT_POLICY,
        allow_resume=True,
        extra_test_sets=extra_test_sets,
    )

    # The engine scores exactly one primary test set, so it was given the pooled
    # union of the per-domain tests. That pooled Dice averages three cameras into
    # one number and is not a result of this experiment; rename it so nobody
    # quotes it by habit, and give the per-domain block the obvious key.
    pooled = report.pop("test")
    pooled["pooling"] = {
        "pooled_over_domains": active,
        "warning": (
            "Pooled across acquisition domains and therefore not a per-domain "
            "result. Report test_by_domain instead."
        ),
    }
    report["test_pooled"] = pooled
    report["test_by_domain"] = report.pop("test_by_name")
    report["reporting_rule"] = (
        "Disc and cup metrics are separate; no combined Dice is reported. Each "
        "active domain is scored separately in test_by_domain, which is the "
        "reportable result; test_pooled averages them and must not be quoted."
    )

    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    metadata = {
        "protocol": ALL_DOMAINS_PROTOCOL_NAME,
        "arm": config.experiment_name,
        "conditioning_arm": config.arm,
        "test_conditioning": config.test_conditioning if config.arm != "plain" else None,
        "active_domains": active,
        "inactive_domains": [
            p.domain.value for p in manifest.budgeted_partitions
            if p.domain not in set(fold.domains)
        ],
        "fold_shape": f"train on {len(active)}, test on each",
        "run_seed": seed,
        "budget": {
            "train": config.train_budget,
            "val": config.val_budget,
            "test": config.test_budget,
            "subsample_seed": config.subsample_seed,
        },
        "paired_with": config.paired_arm,
        "source_test_policy": "exclude",
        "source_test_rationale": (
            "Every sample keeps one role: a domain's budgeted train and val "
            "partitions are trained on and validated on, its budgeted test "
            "partition is only ever tested on."
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_manifest_path": str(parent_path),
        "parent_manifest_sha256": _sha256(parent_path),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "git_revision": _git_revision(),
        "started_from_locked_membership": True,
        "resumed_from_epoch": report.get("resumed_from_epoch"),
        "smoke_rehearsal": smoke,
        "scientific_result": not smoke,
        "locked_split_counts": {
            "train": len(locked_views["train"]),
            "val": len(locked_views["val"]),
            "test_by_domain": {d.value: len(locked_views[d.value]) for d in fold.domains},
        },
        "executed_split_counts": {
            "train": len(executed_views["train"]),
            "val": len(executed_views["val"]),
            "test_by_domain": {d.value: len(executed_views[d.value]) for d in fold.domains},
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["all_domains"] = metadata
    _write_json(actual_output / "test_metrics.json", report)
    _write_json(
        actual_output / "resolved_stage4_config.json",
        {"source_config": asdict(config), "device_override": requested_device, "execution": metadata},
    )
    _write_json(
        actual_output / "all_domains_run.json",
        {"all_domains": metadata, "artifacts": report["artifacts"]},
    )
    print("per-domain test Dice (the reportable result):", flush=True)
    for domain in fold.domains:
        block = report["test_by_domain"][domain.value]
        line = (
            f"  {domain.value}: disc={block['disc']['dice_mean']:.4f} "
            f"cup={block['cup']['dice_mean']:.4f} n={block['evaluated_sample_count']}"
        )
        sweep = block.get("conditioning", {}).get("used_minus_worst_fixed_code_dice")
        if sweep:
            line += (
                f" | wrong-code penalty disc={sweep['disc']:+.4f} cup={sweep['cup']:+.4f}"
            )
        print(line, flush=True)
    print(f"completed seed {seed}: {actual_output}")
    return report


def run(config: Stage3SingleSourceConfig, config_path: Path, args: argparse.Namespace) -> int:
    _require_arm_policy(config)
    seeds = _selected_seeds(config, args)
    manifest, _, records_by_key, manifest_path = _load_locked_runtime(config)
    for seed in seeds:
        _run_one(
            config=config,
            config_path=config_path,
            manifest=manifest,
            records_by_key=records_by_key,
            manifest_path=manifest_path,
            seed=seed,
            smoke=args.smoke,
            requested_device=args.device,
            explicit_output=args.out_dir,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        config = Stage3SingleSourceConfig.from_json(
            config_path, expected_stage=CONFIG_STAGE, domains_key=CONFIG_DOMAINS_KEY
        )
        if args.command == "check":
            return check(config, args.skip_mask_audit)
        if args.command == "run":
            return run(config, config_path, args)
        raise Stage3ConfigError(f"Unsupported command {args.command!r}")
    except (
        AllDomainsFoldError,
        LodoManifestError,
        SingleSourceManifestError,
        Stage3ConfigError,
        Stage3DataError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
