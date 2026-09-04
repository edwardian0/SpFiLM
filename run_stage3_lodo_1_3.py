#!/usr/bin/env python3
"""Prepare, validate, and execute fixed-budget single-source Stage 3 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib-cache")
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
    SINGLE_SOURCE_PROTOCOL_NAME,
    SingleSourceManifest,
    SingleSourceManifestError,
    load_single_source_manifest,
    write_single_source_manifest,
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
    build_single_source_manifest,
    parent_lodo_manifest_path,
    select_single_source_smoke_splits,
    single_source_fold_splits,
    single_source_manifest_path,
    validate_single_source_manifest_against_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo_single.json"
SINGLE_SOURCE_SPLIT_POLICY = (
    "locked single-source: source train/val only; three named target tests; "
    "source test and target train/val excluded"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 plain-U-Net train-on-one, test-on-three protocol with "
            "fixed budgets and frozen, revalidated membership"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Stage 3 single-source JSON config (place this before the command)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Discover data and write the canonical budgeted-fold manifest",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing manifest only when intentional membership changed",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Revalidate discovery, budgets, membership, and decoded masks",
    )
    check_parser.add_argument(
        "--skip-mask-audit",
        action="store_true",
        help="Skip full image/mask decoding; membership checks still run",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Train one configured source/seed or every configured combination",
    )
    run_parser.add_argument(
        "--source-domain",
        choices=tuple(domain.value for domain in Domain),
        help="The only domain used for training and checkpoint selection",
    )
    run_parser.add_argument("--seed", type=int, help="Configured run seed")
    run_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configured source-domain/seed combinations sequentially",
    )
    run_parser.add_argument(
        "--smoke",
        action="store_true",
        help="One-epoch, 128px plumbing rehearsal; never a scientific result",
    )
    run_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Override only the requested compute device",
    )
    run_parser.add_argument(
        "--out-dir",
        type=Path,
        help="Exact base output directory for a single source/seed run",
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
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return f"{commit}{'-dirty' if dirty else ''}"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def _manifest_summary(manifest: SingleSourceManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "parent_manifest_sha256": manifest.parent_manifest_sha256,
        "budget": {
            "train": manifest.train_budget,
            "val": manifest.val_budget,
            "test": manifest.test_budget,
            "subsample_seed": manifest.subsample_seed,
        },
        "budgeted_partitions": {
            partition.domain.value: {
                name: len(getattr(partition, name))
                for name in ("train", "val", "test")
            }
            for partition in manifest.budgeted_partitions
        },
        "folds": {
            fold.source_domain.value: {
                "train": len(fold.train),
                "val": len(fold.val),
                "test": len(fold.test),
                "test_by_domain": {
                    domain.value: len(samples) for domain, samples in fold.tests
                },
            }
            for fold in manifest.folds
        },
    }


def _load_locked_runtime(
    config: Stage3SingleSourceConfig,
) -> tuple[
    SingleSourceManifest,
    dict[Domain, list[FundusRecord]],
    dict[SampleKey, FundusRecord],
    Path,
]:
    manifest_path = single_source_manifest_path(config, PROJECT_ROOT)
    if not manifest_path.is_file():
        raise Stage3DataError(
            f"Locked single-source manifest is missing: {manifest_path}. Run "
            "prepare, inspect it, and commit it before check or run."
        )
    manifest = load_single_source_manifest(manifest_path)
    records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
    validate_single_source_manifest_against_config(
        config,
        manifest,
        records_by_domain,
        PROJECT_ROOT,
    )
    parent_manifest = load_lodo_manifest(
        parent_lodo_manifest_path(config, PROJECT_ROOT)
    )
    records_by_key = resolve_manifest_records(parent_manifest, records_by_domain)
    return manifest, records_by_domain, records_by_key, manifest_path


def prepare(config: Stage3SingleSourceConfig, force: bool) -> int:
    records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
    manifest = build_single_source_manifest(
        config,
        records_by_domain,
        PROJECT_ROOT,
    )
    output_path = single_source_manifest_path(config, PROJECT_ROOT)
    status = "created"
    if output_path.exists():
        existing = load_single_source_manifest(output_path)
        if existing == manifest:
            status = "unchanged"
        elif not force:
            raise Stage3DataError(
                f"Refusing to replace changed membership at {output_path}; "
                "inspect the diff and rerun prepare --force only if the change "
                "is intended"
            )
        else:
            status = "replaced"
    if status != "unchanged":
        write_single_source_manifest(manifest, output_path)
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(output_path),
                "sha256": _sha256(output_path),
                **_manifest_summary(manifest),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if status != "unchanged":
        print("Review and commit this manifest before launching Stage 3.")
    return 0


def check(config: Stage3SingleSourceConfig, skip_mask_audit: bool) -> int:
    manifest, records_by_domain, _, manifest_path = _load_locked_runtime(config)
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    report: dict[str, object] = {
        "status": "OK",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": _sha256(parent_path),
        "discovered_counts": {
            domain.value: len(records)
            for domain, records in sorted(records_by_domain.items())
        },
        **_manifest_summary(manifest),
    }
    if skip_mask_audit:
        report["mask_audit"] = "SKIPPED by explicit flag"
    else:
        report["mask_audit"] = audit_lodo_domains(records_by_domain)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _selected_runs(
    config: Stage3SingleSourceConfig,
    args: argparse.Namespace,
) -> list[tuple[Domain, int]]:
    if args.all:
        if args.source_domain is not None or args.seed is not None:
            raise Stage3ConfigError(
                "--all cannot be combined with --source-domain or --seed"
            )
        if args.out_dir is not None:
            raise Stage3ConfigError("--out-dir is only valid for a single run")
        return [
            (domain, seed)
            for domain in config.source_domains
            for seed in config.run_seeds
        ]
    if args.source_domain is None or args.seed is None:
        raise Stage3ConfigError(
            "run requires both --source-domain and --seed, or explicit --all"
        )
    domain = Domain(args.source_domain)
    if domain not in config.source_domains:
        raise Stage3ConfigError(f"Domain {domain.value} is not configured")
    if args.seed not in config.run_seeds:
        raise Stage3ConfigError(
            f"Seed {args.seed} is not one of the configured seeds "
            f"{list(config.run_seeds)}"
        )
    return [(domain, args.seed)]


def _run_output_dir(
    config: Stage3SingleSourceConfig,
    source_domain: Domain,
    seed: int,
    explicit_output: Path | None,
) -> Path:
    if explicit_output is not None:
        return resolve_project_output(PROJECT_ROOT, str(explicit_output))
    base = resolve_project_output(PROJECT_ROOT, config.output_dir)
    return base / source_domain.value / f"seed_{seed}"


def _require_fresh_output(base_output: Path, smoke: bool) -> Path:
    actual_output = (
        base_output.with_name(f"{base_output.name}_smoke") if smoke else base_output
    )
    if actual_output.exists() and any(actual_output.iterdir()):
        # A requeued job keeps its SLURM_JOB_ID and so lands here again. A resume
        # file means the previous attempt was preempted mid-training and can be
        # continued; run_experiment revalidates it against config and splits.
        if (actual_output / RESUME_STATE_FILENAME).is_file():
            return actual_output
        raise Stage3DataError(
            f"Refusing to overwrite non-empty run directory {actual_output}; "
            "choose a new --out-dir"
        )
    return actual_output


def _split_counts(
    views: dict[str, list[FundusRecord]],
    target_domains: tuple[Domain, ...],
) -> dict[str, object]:
    test_by_domain = {
        domain.value: len(views[domain.value]) for domain in target_domains
    }
    return {
        "train": len(views["train"]),
        "val": len(views["val"]),
        "test": sum(test_by_domain.values()),
        "test_by_domain": test_by_domain,
    }


def _run_one(
    config: Stage3SingleSourceConfig,
    config_path: Path,
    manifest: SingleSourceManifest,
    records_by_key: dict[SampleKey, FundusRecord],
    manifest_path: Path,
    source_domain: Domain,
    seed: int,
    smoke: bool,
    requested_device: str | None,
    explicit_output: Path | None,
) -> dict[str, Any]:
    fold = next(
        fold for fold in manifest.folds if fold.source_domain == source_domain
    )
    locked_views = single_source_fold_splits(fold, records_by_key)
    executed_views = (
        select_single_source_smoke_splits(locked_views) if smoke else locked_views
    )
    target_domains = fold.target_domains
    pooled_test = sorted(
        (
            record
            for domain in target_domains
            for record in executed_views[domain.value]
        ),
        key=lambda record: (record.domain, record.sample_id),
    )
    engine_splits = {
        "train": executed_views["train"],
        "val": executed_views["val"],
        "test": pooled_test,
    }
    extra_test_sets = {
        domain.value: executed_views[domain.value] for domain in target_domains
    }
    records = [
        record
        for name in ("train", "val", "test")
        for record in engine_splits[name]
    ]
    base_output = _run_output_dir(config, source_domain, seed, explicit_output)
    actual_output = _require_fresh_output(base_output, smoke)
    relative_output = base_output.relative_to(PROJECT_ROOT)
    engine_config = config.training_config(
        source_domain,
        seed,
        str(relative_output),
        requested_device=requested_device,
    )
    device = choose_device(engine_config.requested_device)
    target_text = ", ".join(
        f"{domain.value}={len(locked_views[domain.value])}"
        for domain in target_domains
    )
    print(
        f"Stage 3 single-source | source={source_domain.value} | seed={seed} | "
        f"device={device} | {'SMOKE' if smoke else 'FULL'}",
        flush=True,
    )
    print(
        f"locked source train={len(locked_views['train'])}, "
        f"val={len(locked_views['val'])} | target tests: {target_text}",
        flush=True,
    )
    if smoke:
        smoke_target_text = ", ".join(
            f"{domain.value}={len(executed_views[domain.value])}"
            for domain in target_domains
        )
        print(
            f"smoke subset source train={len(executed_views['train'])}, "
            f"val={len(executed_views['val'])} | target tests: "
            f"{smoke_target_text}; this output is not a scientific result",
            flush=True,
        )

    report = run_experiment(
        engine_config,
        PROJECT_ROOT,
        smoke=smoke,
        records=records,
        split_records=engine_splits,
        extra_test_sets=extra_test_sets,
        split_policy=SINGLE_SOURCE_SPLIT_POLICY,
        allow_resume=True,
    )
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    single_source_metadata = {
        "protocol": SINGLE_SOURCE_PROTOCOL_NAME,
        "arm": config.experiment_name,
        "source_domain": source_domain.value,
        "target_domains": [domain.value for domain in target_domains],
        "run_seed": seed,
        "budget": {
            "train": config.train_budget,
            "val": config.val_budget,
            "test": config.test_budget,
            "subsample_seed": config.subsample_seed,
        },
        "source_test_policy": "exclude",
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
        "locked_split_counts": _split_counts(locked_views, target_domains),
        "executed_split_counts": _split_counts(executed_views, target_domains),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["single_source"] = single_source_metadata
    _write_json(actual_output / "test_metrics.json", report)
    _write_json(
        actual_output / "resolved_stage3_config.json",
        {
            "source_config": asdict(config),
            "device_override": requested_device,
            "execution": single_source_metadata,
        },
    )
    _write_json(
        actual_output / "single_source_run.json",
        {
            "single_source": single_source_metadata,
            "artifacts": report["artifacts"],
        },
    )
    print(f"completed {source_domain.value} seed {seed}: {actual_output}")
    return report


def run(
    config: Stage3SingleSourceConfig,
    config_path: Path,
    args: argparse.Namespace,
) -> int:
    selections = _selected_runs(config, args)
    manifest, _, records_by_key, manifest_path = _load_locked_runtime(config)
    for source_domain, seed in selections:
        _run_one(
            config=config,
            config_path=config_path,
            manifest=manifest,
            records_by_key=records_by_key,
            manifest_path=manifest_path,
            source_domain=source_domain,
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
        config = Stage3SingleSourceConfig.from_json(config_path)
        if args.command == "prepare":
            return prepare(config, args.force)
        if args.command == "check":
            return check(config, args.skip_mask_audit)
        if args.command == "run":
            return run(config, config_path, args)
        raise Stage3ConfigError(f"Unsupported command {args.command!r}")
    except (
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
