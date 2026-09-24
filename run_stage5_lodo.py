#!/usr/bin/env python3
"""Step 5: leave-one-domain-out over the active domains, one conditioning arm per run.

Step 5 of the project brief puts the conditioning arms head to head under the
brief's own protocol (Section 5): train on every active domain but one, test on
the held-out one, and rotate through every domain. Over the three active domains
(RIM-ONE-DL has been inactive since Step 4) each fold trains on two domains'
budgeted partitions (80 train / 20 val) and scores the held-out domain's
budgeted test partition (50). The plain U-Net and the Global FiLM arm are
separate runs -- separate SLURM jobs -- whose configs differ only in ``arm``.
Nothing here branches on which conditioning a run uses, only on whether it has
one, so once ``spatial_film`` is an arm SpFiLM joins these folds with a config.

The folds are the fixed-budget LODO folds, composed by the same
``fixed_lodo_folds`` from the same locked budgeted manifest as Stage 3's
fixed-budget arm and the Step 4 LODO configs. A held-out domain's 50 test images
are therefore the ones the train-on-one and train-on-all arms score too, and the
arms of this step pair on identical images. What this runner adds is identity:
its own config stage, protocol name and metadata block (``stage5_lodo`` in
``test_metrics.json`` and ``stage5_lodo_run.json``), so Step 5 runs are read by
``aggregate_stage5_lodo.py`` alone and never enter a Stage 3 or Step 4 report,
whose tools read ``fixed_lodo``.

A conditioned arm has no code for the held-out domain. It gets one code for the
whole domain: the source domain whose training colour centroid is nearest to the
mean of the held-out domain's unlabelled reference sample -- its budgeted train
partition, never trained on, labels unused, disjoint from the test images. That
is the rule agreed with the supervisor on 2026-09-12. The oracle code does not
exist for an unseen domain and is refused.

    python run_stage5_lodo.py --config configs/stage5_lodo_global_film_3dom.json check --skip-mask-audit
    python run_stage5_lodo.py --config configs/stage5_lodo_global_film_3dom.json run --held-out-domain drishti_gs --seed 42 --smoke --device cpu
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_stage3_lodo_3_1_fixed import (  # noqa: E402
    CONDITIONING_REFERENCE_PARTITION,
    fixed_lodo_folds,
    held_out_reference_keys,
)
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


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage5_lodo_global_film_3dom.json"
STAGE5_LODO_PROTOCOL_NAME = "stage5_lodo_fixed_budget"
# The config's ``stage`` is the protocol name, so a Step 4 LODO config (stage
# "lodo_fixed_budget") cannot run here and a Step 5 config cannot run under the
# Stage 3 fixed-budget runner.
CONFIG_STAGE = STAGE5_LODO_PROTOCOL_NAME
CONFIG_DOMAINS_KEY = "held_out_domains"
STAGE5_LODO_METADATA_KEY = "stage5_lodo"
STAGE5_LODO_RUN_RECORD = "stage5_lodo_run.json"
STAGE5_LODO_RESOLVED_CONFIG = "resolved_stage5_config.json"
# A smoke rehearsal needs the code-decision path, not the full reference sample.
SMOKE_REFERENCE_IMAGES = 3
STAGE5_LODO_SPLIT_POLICY = (
    "Step 5 leave-one-domain-out over the config's active domains: each active "
    "source domain contributes its budgeted train and val partitions; the "
    "held-out domain's budgeted test partition is the only test set; source "
    "tests and inactive domains excluded; the held-out domain's train and val "
    "partitions are never trained or validated on (a conditioned arm reads its "
    "train images, unlabelled, to choose the domain's code)"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5 leave-one-domain-out over the config's active domains: train "
            "on all but one, test on the held-out domain (one arm per run)"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Step 5 LODO JSON config (place before the command)",
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


def stage5_lodo_folds(
    manifest: SingleSourceManifest, config: Stage3SingleSourceConfig
) -> tuple[LodoFold, ...]:
    """The Step 5 folds: the fixed-budget LODO folds over the config's active domains."""

    return fixed_lodo_folds(manifest, config.active_domains)


def stage5_lodo_fold(
    manifest: SingleSourceManifest,
    config: Stage3SingleSourceConfig,
    held_out_domain: Domain,
) -> LodoFold:
    for fold in stage5_lodo_folds(manifest, config):
        if fold.held_out_domain == held_out_domain:
            return fold
    raise Stage3ConfigError(
        f"{held_out_domain.value} is not an active domain of this protocol "
        f"{[domain.value for domain in config.active_domains]}"
    )


def conditioning_reference_keys(
    config: Stage3SingleSourceConfig,
    manifest: SingleSourceManifest,
    held_out_domain: Domain,
) -> tuple[SampleKey, ...]:
    """The held-out domain's unlabelled reference sample, or none when the arm needs none.

    Only a conditioned arm under the per-domain rule decides its code from
    images of the held-out domain; the per-image ablation and the plain arm do
    not look at any held-out image before testing.
    """

    if config.arm == "plain" or config.test_conditioning != "nearest_domain":
        return ()
    return held_out_reference_keys(manifest, held_out_domain)


def _require_arm_policy(config: Stage3SingleSourceConfig) -> None:
    if config.arm != "plain" and config.test_conditioning == "oracle":
        raise Stage3ConfigError(
            "Leave-one-domain-out tests a domain the model has no code for, so "
            "there is no oracle code; the film block must set test_conditioning "
            "to 'nearest_domain' (the agreed per-domain rule) or 'nearest_image' "
            "(the per-image ablation)"
        )


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
    # The same validation every fixed-budget arm runs: passing it under this
    # config proves Step 5 resolves to the one locked membership.
    validate_single_source_manifest_against_config(
        config, manifest, records_by_domain, PROJECT_ROOT
    )
    parent_manifest = load_lodo_manifest(parent_lodo_manifest_path(config, PROJECT_ROOT))
    records_by_key = resolve_manifest_records(parent_manifest, records_by_domain)
    return manifest, records_by_domain, records_by_key, manifest_path


def _resolve_reference(
    keys: Sequence[SampleKey],
    records_by_key: dict[SampleKey, FundusRecord],
    held_out_domain: Domain,
) -> list[FundusRecord]:
    try:
        return [records_by_key[key] for key in keys]
    except KeyError as error:
        raise Stage3DataError(
            f"Reference image {error.args[0]!r} for {held_out_domain.value} "
            "is not among the discovered records"
        ) from None


def _inactive_domains(
    manifest: SingleSourceManifest, config: Stage3SingleSourceConfig
) -> list[str]:
    active = set(config.active_domains)
    return [
        partition.domain.value
        for partition in manifest.budgeted_partitions
        if partition.domain not in active
    ]


def check(config: Stage3SingleSourceConfig, skip_mask_audit: bool) -> int:
    _require_arm_policy(config)
    manifest, records_by_domain, records_by_key, manifest_path = _load_locked_runtime(config)
    folds: dict[str, object] = {}
    for fold in stage5_lodo_folds(manifest, config):
        reference = _resolve_reference(
            conditioning_reference_keys(config, manifest, fold.held_out_domain),
            records_by_key,
            fold.held_out_domain,
        )
        folds[fold.held_out_domain.value] = {
            "source_domains": sorted({sample.domain.value for sample in fold.train}),
            "train": len(fold.train),
            "val": len(fold.val),
            "test": len(fold.test),
            "conditioning_reference": len(reference) if reference else None,
        }
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    report: dict[str, object] = {
        "status": "OK",
        "protocol": STAGE5_LODO_PROTOCOL_NAME,
        "experiment": config.experiment_name,
        "arm": config.arm,
        "test_conditioning": config.test_conditioning if config.arm != "plain" else None,
        "paired_arm": config.paired_arm,
        "active_domains": [domain.value for domain in config.active_domains],
        "inactive_domains": _inactive_domains(manifest, config),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": _sha256(parent_path),
        "discovered_counts": {
            domain.value: len(records)
            for domain, records in sorted(records_by_domain.items())
        },
        "budget": {
            "train": manifest.train_budget,
            "val": manifest.val_budget,
            "test": manifest.test_budget,
            "subsample_seed": manifest.subsample_seed,
        },
        "folds": folds,
    }
    report["mask_audit"] = (
        "SKIPPED by explicit flag" if skip_mask_audit else audit_lodo_domains(records_by_domain)
    )
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
        raise Stage3ConfigError(
            f"Domain {domain.value} is not an active domain of this protocol "
            f"{[d.value for d in config.active_domains]}"
        )
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
        # A relaunched job pointed at the same --out-dir lands here again. A
        # resume file means the previous attempt was preempted mid-training and
        # can be continued; run_experiment revalidates it against config and splits.
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
    fold = stage5_lodo_fold(manifest, config, held_out_domain)
    locked_splits = fold_record_splits(fold, records_by_key)
    executed_splits = (
        select_lodo_smoke_splits(locked_splits) if smoke else locked_splits
    )
    records = [
        record
        for name in ("train", "val", "test")
        for record in executed_splits[name]
    ]
    conditioning_reference = _resolve_reference(
        conditioning_reference_keys(config, manifest, held_out_domain),
        records_by_key,
        held_out_domain,
    )
    if smoke:
        conditioning_reference = conditioning_reference[:SMOKE_REFERENCE_IMAGES]
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
        f"Step 5 LODO | arm={config.arm} | train on {len(source_domains)}, test on 1 | "
        f"held out={held_out_domain.value} | seed={seed} | device={device} | "
        f"{'SMOKE' if smoke else 'FULL'}",
        flush=True,
    )
    print(
        f"active domains={active_domains} | sources={source_domains} | locked splits "
        + ", ".join(
            f"{name}={len(locked_splits[name])}" for name in ("train", "val", "test")
        )
        + (
            f" | code decided from {len(conditioning_reference)} unlabelled "
            f"{held_out_domain.value} reference images"
            if conditioning_reference
            else ""
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
        split_policy=STAGE5_LODO_SPLIT_POLICY,
        allow_resume=True,
        conditioning_reference=conditioning_reference or None,
    )
    parent_path = parent_lodo_manifest_path(config, PROJECT_ROOT)
    metadata = {
        "protocol": STAGE5_LODO_PROTOCOL_NAME,
        # The arm is the config's experiment_name, so a reporting tool reads the
        # arm rather than inferring it from a directory name.
        "arm": config.experiment_name,
        "conditioning_arm": config.arm,
        "test_conditioning": config.test_conditioning if config.arm != "plain" else None,
        "held_out_domain": held_out_domain.value,
        "source_domains": source_domains,
        "active_domains": active_domains,
        "inactive_domains": _inactive_domains(manifest, config),
        "fold_shape": f"train on {len(source_domains)}, test on 1",
        "fold_composition": (
            "fixed_lodo_folds over the shared budgeted manifest: the Stage 3 "
            "fixed-budget and Step 4 LODO folds"
        ),
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
        # Within Step 5 the arms pair on identical test images and differ only
        # in the conditioning; the config names the partner.
        "paired_with": config.paired_arm,
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
    report[STAGE5_LODO_METADATA_KEY] = metadata
    _write_json(actual_output / "test_metrics.json", report)
    _write_json(
        actual_output / STAGE5_LODO_RESOLVED_CONFIG,
        {"source_config": asdict(config), "device_override": requested_device, "execution": metadata},
    )
    _write_json(
        actual_output / STAGE5_LODO_RUN_RECORD,
        {STAGE5_LODO_METADATA_KEY: metadata, "artifacts": report["artifacts"]},
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
    conditioning = report.get("conditioning")
    if isinstance(conditioning, dict):
        decision = conditioning.get("domain_decision") or {}
        gap = conditioning.get("nearest_domain_minus_best_fixed_code_dice") or {}
        best = conditioning.get("best_fixed_code") or {}
        print(
            f"code used={decision.get('chosen_domain', conditioning.get('test_conditioning'))}"
            + "".join(
                f" | {structure}: best fixed code={best[structure]}, "
                f"used - best={gap[structure]:+.4f}"
                for structure in ("disc", "cup")
                if structure in best and structure in gap
            ),
            flush=True,
        )
    print(f"completed {held_out_domain.value} seed {seed}: {actual_output}")
    return report


def run(
    config: Stage3SingleSourceConfig,
    config_path: Path,
    args: argparse.Namespace,
) -> int:
    _require_arm_policy(config)
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
