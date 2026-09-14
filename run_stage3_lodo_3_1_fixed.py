#!/usr/bin/env python3
"""Fixed-budget leave-one-domain-out: hold one domain out, train on the others.

Which domains take part is the config's ``protocol.held_out_domains`` list (the
*active* set); every listed domain is held out in turn and the remaining listed
domains are the sources. The Stage 3 configs list all four domains (train on
three, test on the fourth). The Step 4 configs drop RIM-ONE-DL for now and list
three, so each fold trains on two and tests on the third; RIM-ONE-DL stays
discovered and validated because the locked manifests cover it, it simply does
not enter a fold. Each domain's budgeted partitions are the same either way, so
a held-out domain's 50 test images are identical across both protocols and
across the plain and Global FiLM arms.

This is the paired counterpart to ``run_stage3_lodo_1_3.py``. Both arms draw
their membership from the same committed budgeted manifest, so for a given
held-out domain the 50 test images here are *the same 50 images* the train-on-one
arm scored. Under the four-domain protocol the only difference between those
arms is training volume:

    train-on-one    40 train, 10 val   -> scored on each unseen domain's 50
    train-on-three  120 train, 30 val  -> scored on the held-out domain's 50

That makes the comparison paired and removes the confound in the original
full-data arm (``run_stage3_lodo_3_1.py``), whose pooled training set swung
between 552 and 852 images depending on which domain was dropped, and whose test
sets ranged from 51 to 97 images.

The original full-data arm is untouched and remains reportable in its own right;
this arm does not supersede it, it controls it.
"""

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
    LodoFold,
    LodoManifestError,
    SampleKey,
    compose_all_lodo_folds,
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
    fold_record_splits,
    resolve_manifest_records,
    resolve_project_output,
    select_lodo_smoke_splits,
)
from spfilm.stage3_single_source import (  # noqa: E402
    Stage3SingleSourceConfig,
    parent_lodo_manifest_path,
    single_source_manifest_path,
    validate_single_source_manifest_against_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3_lodo_fixed.json"
CONFIG_STAGE = "lodo_fixed_budget"
CONFIG_DOMAINS_KEY = "held_out_domains"
FIXED_LODO_PROTOCOL_NAME = "leave_one_domain_out_fixed_budget"
FIXED_LODO_SPLIT_POLICY = (
    "fixed-budget LODO over the config's active domains: each active source "
    "domain contributes its budgeted train and val partitions; the held-out "
    "domain's budgeted test partition is the only test set; source tests, "
    "held-out train/val, and inactive domains excluded"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-budget leave-one-domain-out over the config's active domains "
            "(plain or Global FiLM arm), paired with the train-on-one arm"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Stage 3 fixed-budget JSON config (place before the command)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Revalidate discovery, config, shared manifest, and decoded masks",
    )
    check_parser.add_argument(
        "--skip-mask-audit",
        action="store_true",
        help="Skip full image/mask decoding; membership checks still run",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Train one configured fold/seed or every configured combination",
    )
    run_parser.add_argument(
        "--held-out-domain",
        choices=tuple(domain.value for domain in Domain),
        help="Domain used only as this run's locked test partition",
    )
    run_parser.add_argument("--seed", type=int, help="Configured run seed")
    run_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all configured domain/seed combinations sequentially",
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
        help="Exact base output directory for a single fold/seed run",
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


def fixed_lodo_folds(
    manifest: SingleSourceManifest,
    active_domains: Sequence[Domain] | None = None,
) -> tuple[LodoFold, ...]:
    """Compose the LODO folds from the shared budgeted partitions.

    No new composition logic: the existing, already-tested ``compose_all_lodo_folds``
    is applied to partitions that have been capped to the common budget, which is
    what makes every fold (sources x 40)/(sources x 10)/50 instead of the original
    arm's 552-852/51-97. ``active_domains`` restricts the partitions that take
    part; ``None`` means every domain in the manifest.
    """

    partitions = manifest.budgeted_partitions
    if active_domains is not None:
        active = set(active_domains)
        missing = sorted(active - {partition.domain for partition in partitions})
        if missing:
            raise Stage3DataError(
                "Active domains absent from the budgeted manifest: "
                f"{[domain.value for domain in missing]}"
            )
        partitions = tuple(
            partition for partition in partitions if partition.domain in active
        )
        if len(partitions) < 2:
            raise Stage3ConfigError(
                "Leave-one-domain-out needs at least two active domains, got "
                f"{[partition.domain.value for partition in partitions]}"
            )
    return compose_all_lodo_folds(partitions)


CONDITIONING_REFERENCE_PARTITION = "held_out_budgeted_train"


def held_out_reference_keys(
    manifest: SingleSourceManifest, held_out_domain: Domain
) -> tuple[SampleKey, ...]:
    """The held-out domain's unlabelled reference sample for a conditioned arm.

    Under ``test_conditioning="nearest_domain"`` the conditioned model needs one
    code for the whole held-out domain, decided from images of that domain. The
    budgeted training partition is used: it is excluded from every fold's
    training and validation (the fold trains on the *other* domains), its labels
    are never read, and it is disjoint from the budgeted test partition, so the
    test images play no part in the decision. Any partition would give the same
    kind of evidence; this one is the locked, already-committed choice.
    """

    for partition in manifest.budgeted_partitions:
        if partition.domain == held_out_domain:
            return tuple(partition.train)
    raise Stage3DataError(
        f"{held_out_domain.value} has no budgeted partition in the manifest"
    )


def _manifest_summary(
    manifest: SingleSourceManifest, config: Stage3SingleSourceConfig
) -> dict[str, object]:
    return {
        "active_domains": [domain.value for domain in config.active_domains],
        "inactive_domains": [
            partition.domain.value
            for partition in manifest.budgeted_partitions
            if partition.domain not in set(config.active_domains)
        ],
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
        "fixed_lodo_folds": {
            fold.held_out_domain.value: {
                name: len(getattr(fold, name))
                for name in ("train", "val", "test")
            }
            for fold in fixed_lodo_folds(manifest, config.active_domains)
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
            f"Shared budgeted manifest is missing: {manifest_path}. It is "
            "produced by run_stage3_lodo_1_3.py prepare and must be committed "
            "before either arm runs."
        )
    manifest = load_single_source_manifest(manifest_path)
    records_by_domain = discover_lodo_records(config, PROJECT_ROOT)
    # The same validation the train-on-one arm runs. Passing it under this
    # config is the proof that both arms resolve to one membership.
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


def check(config: Stage3SingleSourceConfig, skip_mask_audit: bool) -> int:
    manifest, records_by_domain, _, manifest_path = _load_locked_runtime(config)
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    report: dict[str, object] = {
        "status": "OK",
        "protocol": FIXED_LODO_PROTOCOL_NAME,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": _sha256(parent_path),
        "discovered_counts": {
            domain.value: len(records)
            for domain, records in sorted(records_by_domain.items())
        },
        **_manifest_summary(manifest, config),
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
        if args.held_out_domain is not None or args.seed is not None:
            raise Stage3ConfigError(
                "--all cannot be combined with --held-out-domain or --seed"
            )
        if args.out_dir is not None:
            raise Stage3ConfigError("--out-dir is only valid for a single run")
        return [
            (domain, seed)
            for domain in config.held_out_domains
            for seed in config.run_seeds
        ]
    if args.held_out_domain is None or args.seed is None:
        raise Stage3ConfigError(
            "run requires both --held-out-domain and --seed, or explicit --all"
        )
    domain = Domain(args.held_out_domain)
    if domain not in config.held_out_domains:
        raise Stage3ConfigError(f"Domain {domain.value} is not configured")
    if args.seed not in config.run_seeds:
        raise Stage3ConfigError(
            f"Seed {args.seed} is not one of the configured seeds "
            f"{list(config.run_seeds)}"
        )
    return [(domain, args.seed)]


def _run_output_dir(
    config: Stage3SingleSourceConfig,
    held_out_domain: Domain,
    seed: int,
    explicit_output: Path | None,
) -> Path:
    if explicit_output is not None:
        return resolve_project_output(PROJECT_ROOT, str(explicit_output))
    base = resolve_project_output(PROJECT_ROOT, config.output_dir)
    return base / held_out_domain.value / f"seed_{seed}"


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


def _run_one(
    config: Stage3SingleSourceConfig,
    config_path: Path,
    manifest: SingleSourceManifest,
    records_by_key: dict[SampleKey, FundusRecord],
    manifest_path: Path,
    held_out_domain: Domain,
    seed: int,
    smoke: bool,
    requested_device: str | None,
    explicit_output: Path | None,
) -> dict[str, Any]:
    fold = next(
        fold
        for fold in fixed_lodo_folds(manifest, config.active_domains)
        if fold.held_out_domain == held_out_domain
    )
    locked_splits = fold_record_splits(fold, records_by_key)
    executed_splits = (
        select_lodo_smoke_splits(locked_splits) if smoke else locked_splits
    )
    records = [
        record
        for name in ("train", "val", "test")
        for record in executed_splits[name]
    ]
    conditioning_reference: list[FundusRecord] = []
    if config.arm != "plain" and config.test_conditioning == "nearest_domain":
        reference_keys = held_out_reference_keys(manifest, held_out_domain)
        try:
            conditioning_reference = [records_by_key[key] for key in reference_keys]
        except KeyError as error:
            raise Stage3DataError(
                f"Reference image {error.args[0]!r} for {held_out_domain.value} "
                "is not among the discovered records"
            ) from None
        if smoke:
            # A rehearsal needs the decision path, not the full sample.
            conditioning_reference = conditioning_reference[:3]
    base_output = _run_output_dir(config, held_out_domain, seed, explicit_output)
    actual_output = _require_fresh_output(base_output, smoke)
    relative_output = base_output.relative_to(PROJECT_ROOT)
    engine_config = config.training_config(
        held_out_domain,
        seed,
        str(relative_output),
        requested_device=requested_device,
    )
    device = choose_device(engine_config.requested_device)
    source_domains = sorted({record.domain for record in locked_splits["train"]})
    active_domains = [domain.value for domain in config.active_domains]
    print(
        f"Fixed-budget LODO | arm={config.arm} | "
        f"train on {len(source_domains)}, test on 1 | "
        f"held out={held_out_domain.value} | seed={seed} | device={device} | "
        f"{'SMOKE' if smoke else 'FULL'}",
        flush=True,
    )
    print(
        f"active domains={active_domains} | sources={source_domains} | locked splits "
        + ", ".join(
            f"{name}={len(locked_splits[name])}" for name in ("train", "val", "test")
        ),
        flush=True,
    )
    if smoke:
        print(
            "smoke subset "
            + ", ".join(
                f"{name}={len(executed_splits[name])}"
                for name in ("train", "val", "test")
            )
            + "; this output is not a scientific result",
            flush=True,
        )

    report = run_experiment(
        engine_config,
        PROJECT_ROOT,
        smoke=smoke,
        records=records,
        split_records=executed_splits,
        split_policy=FIXED_LODO_SPLIT_POLICY,
        allow_resume=True,
        conditioning_reference=conditioning_reference or None,
    )
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    fixed_lodo_metadata = {
        "protocol": FIXED_LODO_PROTOCOL_NAME,
        # The arm is the config's experiment_name, so a reporting tool reads the
        # arm rather than inferring it from a directory name. This must stay
        # distinct from the full-data LODO arm and from the train-on-one arm.
        "arm": config.experiment_name,
        "held_out_domain": held_out_domain.value,
        "source_domains": source_domains,
        # The protocol's domain set. Folds are composed from these alone; a
        # configured domain outside this list (RIM-ONE-DL in Step 4) is
        # discovered and validated but never trained on or tested.
        "active_domains": active_domains,
        "fold_shape": f"train on {len(source_domains)}, test on 1",
        "conditioning_reference": (
            {
                "partition": CONDITIONING_REFERENCE_PARTITION,
                "domain": held_out_domain.value,
                "image_count": len(conditioning_reference),
                "labels_used": False,
                "disjoint_from_test": True,
            }
            if conditioning_reference
            else None
        ),
        "run_seed": seed,
        "budget": {
            "train": config.train_budget,
            "val": config.val_budget,
            "test": config.test_budget,
            "subsample_seed": config.subsample_seed,
        },
        "conditioning_arm": config.arm,
        # The plain arm is paired with the train-on-one arm (training volume is
        # the variable); a conditioned arm names the plain fixed-budget arm it is
        # paired with (conditioning is the variable). Same manifest either way.
        "paired_with": config.paired_arm
        or ("stage3_single_source_plain_unet" if config.arm == "plain" else None),
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
        "locked_split_counts": {
            name: len(locked_splits[name]) for name in ("train", "val", "test")
        },
        "executed_split_counts": {
            name: len(executed_splits[name]) for name in ("train", "val", "test")
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["fixed_lodo"] = fixed_lodo_metadata
    _write_json(actual_output / "test_metrics.json", report)
    _write_json(
        actual_output / "resolved_stage3_config.json",
        {
            "source_config": asdict(config),
            "device_override": requested_device,
            "execution": fixed_lodo_metadata,
        },
    )
    _write_json(
        actual_output / "fixed_lodo_run.json",
        {"fixed_lodo": fixed_lodo_metadata, "artifacts": report["artifacts"]},
    )
    test_metrics = report["test"]
    print(
        f"held-out {held_out_domain.value} (trained on {', '.join(source_domains)}) "
        "test Dice: "
        f"disc={test_metrics['disc']['dice_mean']:.4f} "
        f"cup={test_metrics['cup']['dice_mean']:.4f} "
        f"n={test_metrics['evaluated_sample_count']}",
        flush=True,
    )
    print(f"completed {held_out_domain.value} seed {seed}: {actual_output}")
    return report


def run(
    config: Stage3SingleSourceConfig,
    config_path: Path,
    args: argparse.Namespace,
) -> int:
    selections = _selected_runs(config, args)
    manifest, _, records_by_key, manifest_path = _load_locked_runtime(config)
    for held_out_domain, seed in selections:
        _run_one(
            config=config,
            config_path=config_path,
            manifest=manifest,
            records_by_key=records_by_key,
            manifest_path=manifest_path,
            held_out_domain=held_out_domain,
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
            config_path,
            expected_stage=CONFIG_STAGE,
            domains_key=CONFIG_DOMAINS_KEY,
        )
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
