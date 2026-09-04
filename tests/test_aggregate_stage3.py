from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from math import isnan as statistics_isnan, sqrt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.lodo import Domain, load_lodo_manifest  # noqa: E402
from spfilm.metrics import (  # noqa: E402
    PER_IMAGE_FIELDNAMES,
    summarise_per_image_rows,
)

from aggregate_stage3 import (  # noqa: E402
    DEFAULT_MANIFEST,
    RIM_PER_IMAGE_FIELDNAMES,
    HD95_UNIT_GRID,
    HD95_UNIT_NATIVE,
    OPTIONAL_PER_IMAGE_FIELDS,
    PerImageScore,
    Stage3ReportError,
    aggregate,
    build_domain_reports,
    build_paired_substrate,
    discover_stage3_runs,
    fold_test_image_ids,
    holm_adjust,
    load_run_scores,
    main,
    paired_arm_test,
    render_domain_table,
    render_markdown_report,
    render_paired_table,
    seed_confidence_interval,
    select_scientific_runs,
    verify_run_summary,
    write_markdown_report,
    write_report_csv,
)


MANIFEST = load_lodo_manifest(DEFAULT_MANIFEST)
# Runs record the digest of the manifest they were scored against, and the
# aggregation refuses to report a run against any other manifest, so fixtures
# must claim the real one.
MANIFEST_SHA256 = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
SEEDS = (42, 43, 44, 45, 46)
PLAIN_ARM = "stage3_lodo_plain_unet"
SPFILM_ARM = "stage3_lodo_spfilm"
# The interval is Student-t on 4 degrees of freedom; hard-coded so the test does
# not re-derive it from the same library the implementation uses.
T_CRITICAL_4_DF = 2.7764451051977987
# The engine's overlap smoothing constant (src/spfilm/metrics.py).
OVERLAP_SMOOTH = 1e-8

# The two metric frames real Stage 3 output carries, copied from what the engine
# writes rather than paraphrased: three domains stay on the letterboxed grid and
# RIM-ONE-DL is converted to native source pixels.
GRID_FRAME = (
    "metrics computed on the 512px aspect-preserving letterboxed full-image "
    "grid; HD95 is in 512x512 letterboxed-grid pixels"
)
NATIVE_FRAME = (
    "metrics computed on the 512px full-source-image grid; each HD95 value is "
    "divided by that image's letterbox_scale and reported in native-source "
    "pixels"
)
# RIM-ONE-DL runs get provenance columns appended to the per-image CSV after the
# metrics are written, so their header is not the bare metric schema.
RIM_CONTEXT = {
    "release_prefix": "r1",
    "hospital_split": "test_set",
    "diagnosis_class": "normal",
    "native_width": 524,
    "native_height": 524,
    "letterbox_scale": "0.977099236641",
    "hd95_unit": "native_px",
}


def _row(image_id: str, structure: str, dice: float, hd95: float) -> dict[str, object]:
    """One per-image CSV row whose counts are self-consistent with its overlap."""

    # The counts are the primary quantity and every score is derived from them
    # with the engine's own smoothed formulae, because the loader recomputes all
    # three and refuses a row whose scores its counts do not support.
    scale = 1_000_000
    true_positive = int(round(scale * dice))
    false_positive = scale - true_positive
    false_negative = scale - true_positive
    true_negative = 14 * scale + true_positive
    pixels = true_positive + false_positive + false_negative + true_negative
    smooth = OVERLAP_SMOOTH
    return {
        "image_id": image_id,
        "structure": structure,
        "dice": (2 * true_positive + smooth)
        / (2 * true_positive + false_positive + false_negative + smooth),
        "iou": (true_positive + smooth)
        / (true_positive + false_positive + false_negative + smooth),
        "hd95": hd95,
        "acc": (true_positive + true_negative) / pixels,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }


def build_rows(
    image_ids,
    *,
    seed: int,
    arm_offset: float = 0.0,
    seed_jitter: float = 0.004,
    degenerate_ids=(),
):
    """Deterministic per-image rows: image difficulty dominates, seeds wobble."""

    rows: list[dict[str, object]] = []
    for index, image_id in enumerate(sorted(image_ids)):
        # A wide, image-driven spread, exactly the dispersion that must never be
        # mistaken for seed-to-seed variation.
        difficulty = 0.70 + 0.25 * ((index * 37) % 100) / 100.0
        wobble = seed_jitter * (((seed * 13 + index) % 5) - 2) / 2.0
        for structure, shift in (("disc", 0.0), ("cup", -0.08)):
            dice = min(0.999, max(0.001, difficulty + shift + wobble + arm_offset))
            hd95 = float("nan") if image_id in degenerate_ids else 4.0 + 20.0 * (1.0 - dice)
            rows.append(_row(image_id, structure, dice, hd95))
    return rows


def write_run(
    base: Path,
    *,
    arm: str,
    domain: Domain,
    seed: int,
    rows,
    smoke: bool = False,
    directory_name: str | None = None,
    summary_override=None,
    protocol: str = "leave_one_domain_out_locked_test",
    record_arm: bool = False,
    context_columns: bool | None = None,
    metric_frame: str | None = None,
    hd95_unit: str | None = None,
) -> Path:
    """Write one run directory in the shape run_stage3_lodo_3_1.py produces."""

    run_dir = base / (directory_name or f"{domain.value}/seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # RIM-ONE-DL is the domain the engine annotates, so it is also the domain
    # whose fixtures carry the annotated header unless a test says otherwise.
    if context_columns is None:
        context_columns = domain is Domain.RIM_ONE_DL
    if metric_frame is None:
        metric_frame = NATIVE_FRAME if context_columns else GRID_FRAME
    if hd95_unit is None and context_columns:
        hd95_unit = HD95_UNIT_NATIVE
    csv_path = run_dir / "test_per_image_metrics.csv"
    if context_columns:
        fieldnames = [
            PER_IMAGE_FIELDNAMES[0],
            *OPTIONAL_PER_IMAGE_FIELDS,
            *PER_IMAGE_FIELDNAMES[1:],
        ]
        written = [{**row, **RIM_CONTEXT} for row in rows]
    else:
        fieldnames = list(PER_IMAGE_FIELDNAMES)
        written = list(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written)

    summary = summary_override or summarise_per_image_rows(rows)
    # The whole lodo block the runner writes, not a convenient subset: the
    # loader cross-checks these fields against each other, and a fixture that
    # omits half of them stops standing in for real output.
    split_counts = fold_split_counts(domain)
    lodo = {
        "protocol": protocol,
        "held_out_domain": domain.value,
        "source_domains": sorted(
            other.value for other in Domain if other is not domain
        ),
        "run_seed": seed,
        "source_test_policy": "exclude",
        "manifest_sha256": MANIFEST_SHA256,
        "config_sha256": "b" * 64,
        "git_revision": "c" * 40,
        "started_from_locked_membership": True,
        "scientific_result": not smoke,
        "smoke_rehearsal": smoke,
        "locked_split_counts": split_counts,
        "executed_split_counts": dict(split_counts),
    }
    if record_arm:
        lodo["arm"] = arm
    suffix = f"_{domain.value}_seed_{seed}" + ("_smoke" if smoke else "")
    (run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "experiment_name": arm + suffix,
                "best_epoch": 1 if smoke else 259,
                "epochs_run": 1 if smoke else 300,
                "epochs_configured": 1 if smoke else 300,
                "training_seconds": 12.5 if smoke else 1800.0,
                "early_stopping": {
                    "would_have_stopped_at_epoch": None if smoke else 146
                },
                "test": {
                    "disc": summary["disc"],
                    "cup": summary["cup"],
                    "evaluated_sample_count": len(rows) // 2,
                    "metric_frame": metric_frame,
                    # An absolute path from the machine that trained, stale for
                    # anything copied off the cluster; the loader must ignore it
                    # and resolve the CSV beside the JSON.
                    "per_image_csv": f"/scratch/train/{run_dir.name}/{csv_path.name}",
                    "sample_ids": sorted({str(row["image_id"]) for row in rows}),
                    **({} if hd95_unit is None else {"hd95_unit": hd95_unit}),
                },
                "lodo": lodo,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "resolved_stage3_config.json").write_text(
        json.dumps(
            {
                "source_config": {
                    "experiment_name": arm,
                    "base_channels": 16,
                    "epochs": 300,
                    "batch_size": 8,
                    "image_size": 512,
                    "learning_rate": 0.001,
                    "weight_decay": 1e-5,
                    "threshold": 0.5,
                    "horizontal_flip_probability": 0.5,
                    "rotation_degrees": 10.0,
                    "brightness_contrast": 0.1,
                    "early_stopping_mode": "monitor",
                    "patience": 20,
                    "min_epochs": 30,
                },
                "execution": lodo,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def fold_split_counts(domain: Domain) -> dict[str, int]:
    """The locked partition sizes for one fold, taken from the manifest itself.

    Hard-coding these makes a fixture that claims a fold composition the locked
    manifest does not have, which is exactly what the loader refuses.
    """

    fold = next(
        fold for fold in MANIFEST.folds if fold.held_out_domain == domain
    )
    return {
        "train": len(fold.train),
        "val": len(fold.val),
        "test": len(fold.test),
    }


def retag_run(run_dir: Path, key: str, value: object) -> None:
    """Rewrite one lodo field in both files that record it.

    test_metrics.json and resolved_stage3_config.json carry the same execution
    block, and the loader refuses a run whose two copies disagree. A fixture
    that edits one and not the other therefore exercises that cross-check
    instead of whatever it meant to exercise.
    """

    for name, section in (
        ("test_metrics.json", "lodo"),
        ("resolved_stage3_config.json", "execution"),
    ):
        path = run_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[section][key] = value
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rewrite_first_csv_value(run_dir: Path, field: str, value: object) -> None:
    csv_path = run_dir / "test_per_image_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    rows[0][field] = value
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_grid(
    base: Path,
    *,
    arm: str = PLAIN_ARM,
    seeds=SEEDS,
    arm_offset: float = 0.0,
    seed_jitter: float = 0.004,
    degenerate: bool = False,
) -> Path:
    """The full 4-domain x 5-seed grid the report is meant to consume."""

    root = base / arm
    for domain in Domain:
        image_ids = fold_test_image_ids(MANIFEST, domain)
        degenerate_ids = image_ids[:2] if degenerate else ()
        for seed in seeds:
            write_run(
                root,
                arm=arm,
                domain=domain,
                seed=seed,
                rows=build_rows(
                    image_ids,
                    seed=seed,
                    arm_offset=arm_offset,
                    seed_jitter=seed_jitter,
                    degenerate_ids=degenerate_ids,
                ),
            )
    return root


class _TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.base = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)


# --------------------------------------------------------------------------
# Stage A: discovery
# --------------------------------------------------------------------------


class DiscoveryTests(_TempCase):
    def test_finds_every_run_in_the_grid(self) -> None:
        root = build_grid(self.base)

        runs = discover_stage3_runs([root])

        self.assertEqual(len(runs), 20)
        self.assertEqual(
            {run.identity.held_out_domain for run in runs}, set(Domain)
        )
        self.assertEqual({run.identity.run_seed for run in runs}, set(SEEDS))
        self.assertEqual({run.identity.arm for run in runs}, {PLAIN_ARM})

    def test_discovery_is_by_file_not_by_directory_shape(self) -> None:
        """Both the local and the CREATE layouts must be found identically."""

        for name in ("stage3_lodo/drishti_gs/seed_42", "runs/lodo_s3_x_42_9911"):
            write_run(
                self.base,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=42 if name.endswith("42") else 43,
                rows=build_rows(
                    fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS), seed=42
                ),
                directory_name=name,
            )
        # The second run has to differ in seed to be a distinct grid cell.
        runs = discover_stage3_runs([self.base])

        self.assertEqual(len(runs), 2)
        self.assertEqual(
            sorted(run.run_dir.name for run in runs),
            ["lodo_s3_x_42_9911", "seed_42"],
        )

    def test_output_from_other_stages_is_ignored(self) -> None:
        root = build_grid(self.base)
        stage2 = self.base / "stage2_refuge"
        stage2.mkdir()
        (stage2 / "test_metrics.json").write_text(
            json.dumps({"test": {"disc": {}, "cup": {}}}), encoding="utf-8"
        )

        self.assertEqual(len(discover_stage3_runs([self.base])), 20)

    def test_smoke_runs_are_refused(self) -> None:
        root = self.base / "smoke"
        write_run(
            root,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS), seed=42),
            smoke=True,
        )

        runs = discover_stage3_runs([root])
        with self.assertRaisesRegex(
            Stage3ReportError, r"record scientific_result=false"
        ):
            select_scientific_runs(runs)

    def test_smoke_runs_can_be_excluded_deliberately(self) -> None:
        root = build_grid(self.base)
        write_run(
            root,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=99,
            rows=build_rows(fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS), seed=99),
            smoke=True,
            directory_name="drishti_gs/seed_99_smoke",
        )

        kept = select_scientific_runs(discover_stage3_runs([root]), skip_smoke=True)

        self.assertEqual(len(kept), 20)
        self.assertTrue(all(run.scientific_result for run in kept))

    def test_skip_smoke_allows_a_rehearsal_and_real_run_for_the_same_cell(
        self,
    ) -> None:
        domain = Domain.REFUGE_ZEISS
        image_ids = fold_test_image_ids(MANIFEST, domain)
        for smoke, directory_name in (
            (True, "lodo_s3_refuge_zeiss_seed_42_smoke_1"),
            (True, "lodo_s3_refuge_zeiss_seed_42_smoke_2"),
            (False, "lodo_s3_refuge_zeiss_seed_42_real"),
        ):
            write_run(
                self.base,
                arm=PLAIN_ARM,
                domain=domain,
                seed=42,
                rows=build_rows(image_ids, seed=42),
                smoke=smoke,
                directory_name=directory_name,
            )

        discovered = discover_stage3_runs([self.base])
        kept = select_scientific_runs(discovered, skip_smoke=True)

        self.assertEqual(len(discovered), 3)
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].scientific_result)

    def test_duplicate_domain_and_seed_is_rejected(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.REFUGE_ZEISS)
        for name in ("first", "second"):
            write_run(
                self.base,
                arm=PLAIN_ARM,
                domain=Domain.REFUGE_ZEISS,
                seed=42,
                rows=build_rows(image_ids, seed=42),
                directory_name=name,
            )

        with self.assertRaisesRegex(
            Stage3ReportError, r"Duplicate Stage 3 run for .*refuge_zeiss/seed_42"
        ):
            discover_stage3_runs([self.base])

    def test_the_same_arm_in_two_layouts_is_one_arm(self) -> None:
        """The arm names the config that ran, never the directory it landed in."""

        image_ids = fold_test_image_ids(MANIFEST, Domain.REFUGE_ZEISS)
        for seed, name in ((42, "stage3_lodo/a"), (43, "runs/lodo_s3_b_7712")):
            write_run(
                self.base,
                arm=PLAIN_ARM,
                domain=Domain.REFUGE_ZEISS,
                seed=seed,
                rows=build_rows(image_ids, seed=seed),
                directory_name=name,
            )

        runs = discover_stage3_runs([self.base])

        self.assertEqual({run.identity.arm for run in runs}, {PLAIN_ARM})

    def test_a_recorded_arm_is_used(self) -> None:
        write_run(
            self.base,
            arm="stage3_lodo_globalfilm",
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42),
            record_arm=True,
        )

        runs = discover_stage3_runs([self.base])

        self.assertEqual(runs[0].identity.arm, "stage3_lodo_globalfilm")

    def test_arm_evidence_must_corroborate(self) -> None:
        """A copied config that mislabels an arm invalidates every pairing."""

        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
        )
        resolved = run_dir / "resolved_stage3_config.json"
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        payload["source_config"]["experiment_name"] = "stage3_lodo_spfilm"
        resolved.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(
            Stage3ReportError, "Conflicting conditioning-arm evidence"
        ):
            discover_stage3_runs([self.base])

    def test_a_stamped_arm_that_contradicts_the_config_is_rejected(self) -> None:
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
        )
        retag_run(run_dir, "arm", "stage3_lodo_globalfilm")

        with self.assertRaisesRegex(
            Stage3ReportError, "Conflicting conditioning-arm evidence"
        ):
            discover_stage3_runs([self.base])

    def test_the_two_provenance_copies_must_agree(self) -> None:
        """The lodo block is written twice; a run where they differ is not one run."""

        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
        )
        metrics_path = run_dir / "test_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["lodo"]["config_sha256"] = "e" * 64
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(Stage3ReportError, "disagrees with"):
            discover_stage3_runs([self.base])

    def test_a_malformed_digest_is_rejected(self) -> None:
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
        )
        retag_run(run_dir, "config_sha256", "not-a-digest")

        with self.assertRaisesRegex(
            Stage3ReportError, "64 lowercase hexadecimal characters"
        ):
            discover_stage3_runs([self.base])

    def test_duplicate_json_keys_are_rejected(self) -> None:
        domain = Domain.DRISHTI_GS
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        metrics_path = run_dir / "test_metrics.json"
        text = metrics_path.read_text(encoding="utf-8")
        metrics_path.write_text(
            text.replace('"lodo": {', '"lodo": {}, "lodo": {', 1),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(Stage3ReportError, "duplicate JSON object key"):
            discover_stage3_runs([self.base])

    def test_locked_split_counts_must_be_complete_and_positive(self) -> None:
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
        )
        retag_run(run_dir, "locked_split_counts", {"train": 806, "test": 97})

        with self.assertRaisesRegex(
            Stage3ReportError, "locked_split_counts with exactly"
        ):
            discover_stage3_runs([self.base])

    def test_a_scientific_run_must_execute_every_locked_sample(self) -> None:
        domain = Domain.DRISHTI_GS
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        counts = fold_split_counts(domain)
        counts["train"] -= 1
        retag_run(run_dir, "executed_split_counts", counts)

        with self.assertRaisesRegex(
            Stage3ReportError, "marked scientific but executed_split_counts"
        ):
            discover_stage3_runs([self.base])

    def test_result_flags_cannot_claim_scientific_and_smoke(self) -> None:
        domain = Domain.DRISHTI_GS
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        retag_run(run_dir, "smoke_rehearsal", True)

        with self.assertRaisesRegex(Stage3ReportError, "contradictory result flags"):
            discover_stage3_runs([self.base])

    def test_run_must_start_from_locked_membership(self) -> None:
        domain = Domain.DRISHTI_GS
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        retag_run(run_dir, "started_from_locked_membership", False)

        with self.assertRaisesRegex(
            Stage3ReportError, "started_from_locked_membership=true"
        ):
            discover_stage3_runs([self.base])

    def test_source_domains_must_be_the_other_three(self) -> None:
        domain = Domain.DRISHTI_GS
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        retag_run(run_dir, "source_domains", [Domain.REFUGE_ZEISS.value])

        with self.assertRaisesRegex(Stage3ReportError, "source_domains must be"):
            discover_stage3_runs([self.base])

    def test_a_foreign_protocol_is_rejected(self) -> None:
        write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42),
            protocol="pooled_random_split",
        )

        with self.assertRaisesRegex(
            Stage3ReportError, r"records protocol 'pooled_random_split'"
        ):
            discover_stage3_runs([self.base])

    def test_missing_per_image_csv_is_rejected(self) -> None:
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42),
        )
        (run_dir / "test_per_image_metrics.csv").unlink()

        with self.assertRaisesRegex(
            Stage3ReportError, r"no test_per_image_metrics\.csv"
        ):
            discover_stage3_runs([self.base])


# --------------------------------------------------------------------------
# Stage B: per-image rows and locked-membership validation
# --------------------------------------------------------------------------


class MembershipTests(_TempCase):
    def _one_run(self, domain: Domain = Domain.DRISHTI_GS, **kwargs):
        image_ids = fold_test_image_ids(MANIFEST, domain)
        rows = kwargs.pop("rows", None)
        if rows is None:
            rows = build_rows(image_ids, seed=42)
        run_dir = write_run(
            self.base, arm=PLAIN_ARM, domain=domain, seed=42, rows=rows, **kwargs
        )
        return discover_stage3_runs([self.base])[0], run_dir

    def test_full_membership_loads(self) -> None:
        run, _ = self._one_run()

        scores = load_run_scores(run, MANIFEST)

        expected = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        self.assertEqual(len(scores), 2 * len(expected))
        self.assertEqual({score.image_id for score in scores}, set(expected))
        self.assertEqual({score.structure for score in scores}, {"disc", "cup"})

    def test_a_truncated_csv_is_rejected(self) -> None:
        run, run_dir = self._one_run()
        csv_path = run_dir / "test_per_image_metrics.csv"
        lines = csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
        csv_path.write_text("".join(lines[:-6]), encoding="utf-8")

        with self.assertRaisesRegex(
            Stage3ReportError,
            r"rows do not match the locked drishti_gs test partition "
            r"\(48 of 51 images\)",
        ):
            load_run_scores(run, MANIFEST)

    def test_rows_for_the_wrong_domain_are_rejected(self) -> None:
        """A run whose output landed in another domain's directory is caught."""

        write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
        )

        # The JSON's own record of what it scored contradicts the locked fold,
        # so this never reaches the per-image rows.
        with self.assertRaisesRegex(
            Stage3ReportError, "evaluated_sample_count"
        ):
            discover_stage3_runs([self.base])

    def test_an_unexpected_header_is_rejected(self) -> None:
        run, run_dir = self._one_run()
        csv_path = run_dir / "test_per_image_metrics.csv"
        body = csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
        body[0] = "image_id,structure,dice,iou,hd95,acc,tp,fp,fn\n"
        csv_path.write_text("".join(body), encoding="utf-8")

        with self.assertRaisesRegex(Stage3ReportError, r"header is \['image_id'"):
            load_run_scores(run, MANIFEST)

    def test_a_row_longer_than_its_declared_header_is_rejected(self) -> None:
        run, run_dir = self._one_run()
        csv_path = run_dir / "test_per_image_metrics.csv"
        body = csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
        body[1] = body[1].rstrip("\n") + ",undeclared\n"
        csv_path.write_text("".join(body), encoding="utf-8")

        with self.assertRaisesRegex(
            Stage3ReportError, "values beyond the declared CSV header"
        ):
            load_run_scores(run, MANIFEST)

    def test_a_repeated_structure_row_is_rejected(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        rows = build_rows(image_ids, seed=42)
        rows.append(dict(rows[0]))
        run, _ = self._one_run(rows=rows)

        with self.assertRaisesRegex(Stage3ReportError, r"repeats disc for image"):
            load_run_scores(run, MANIFEST)

    def test_an_unknown_structure_is_rejected(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        rows = build_rows(image_ids, seed=42)
        rows[0] = dict(rows[0], structure="rim")
        run, _ = self._one_run(rows=rows)

        with self.assertRaisesRegex(Stage3ReportError, r"unknown structure 'rim'"):
            load_run_scores(run, MANIFEST)

    def test_a_non_numeric_score_is_rejected(self) -> None:
        run, run_dir = self._one_run()
        csv_path = run_dir / "test_per_image_metrics.csv"
        body = csv_path.read_text(encoding="utf-8").splitlines(keepends=True)
        columns = body[1].split(",")
        columns[2] = "n/a"
        body[1] = ",".join(columns)
        csv_path.write_text("".join(body), encoding="utf-8")

        with self.assertRaisesRegex(Stage3ReportError, r"dice is not a number: 'n/a'"):
            load_run_scores(run, MANIFEST)

    def test_an_out_of_range_overlap_is_rejected(self) -> None:
        run, run_dir = self._one_run()
        rewrite_first_csv_value(run_dir, "dice", "1.2")

        with self.assertRaisesRegex(Stage3ReportError, r"dice must be finite"):
            load_run_scores(run, MANIFEST)

    def test_scores_must_agree_with_their_confusion_counts(self) -> None:
        run, run_dir = self._one_run()
        rewrite_first_csv_value(run_dir, "dice", "0.5")

        with self.assertRaisesRegex(
            Stage3ReportError, r"disagrees with confusion-count recomputation"
        ):
            load_run_scores(run, MANIFEST)

    def test_infinite_hd95_is_not_treated_as_an_exclusion(self) -> None:
        run, run_dir = self._one_run()
        rewrite_first_csv_value(run_dir, "hd95", "inf")

        with self.assertRaisesRegex(
            Stage3ReportError, r"hd95 must be non-negative or nan"
        ):
            load_run_scores(run, MANIFEST)

    def test_negative_confusion_count_is_rejected(self) -> None:
        run, run_dir = self._one_run()
        rewrite_first_csv_value(run_dir, "fp", "-1")

        with self.assertRaisesRegex(Stage3ReportError, r"fp must be non-negative"):
            load_run_scores(run, MANIFEST)

    def test_json_sample_ids_must_match_the_locked_fold(self) -> None:
        run, run_dir = self._one_run()
        metrics_path = run_dir / "test_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["test"]["sample_ids"][0] = "not-in-the-manifest"
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(Stage3ReportError, r"test.sample_ids"):
            load_run_scores(run, MANIFEST)

    def test_all_locked_split_counts_must_match_the_manifest(self) -> None:
        run, run_dir = self._one_run()
        counts = fold_split_counts(Domain.DRISHTI_GS)
        counts["train"] -= 1
        retag_run(run_dir, "locked_split_counts", counts)
        retag_run(run_dir, "executed_split_counts", counts)
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(Stage3ReportError, r"manifest fold has"):
            load_run_scores(run, MANIFEST)


# --------------------------------------------------------------------------
# Stage C: recomputed summaries must equal the stored ones
# --------------------------------------------------------------------------


class SummaryVerificationTests(_TempCase):
    def test_every_run_in_the_grid_agrees_with_its_stored_summary(self) -> None:
        root = build_grid(self.base, degenerate=True)

        runs = select_scientific_runs(discover_stage3_runs([root]))

        self.assertEqual(len(runs), 20)
        for run in runs:
            summaries = verify_run_summary(run)
            self.assertEqual(set(summaries), {"disc", "cup"})
            stored = run.stored_summary["disc"]["dice_mean"]
            self.assertAlmostEqual(summaries["disc"].dice_mean, stored, places=12)

    def test_a_tampered_stored_summary_is_rejected(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.REFUGE_ZEISS)
        rows = build_rows(image_ids, seed=42)
        summary = summarise_per_image_rows(rows)
        summary["disc"]["dice_mean"] = float(summary["disc"]["dice_mean"]) + 0.05
        write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.REFUGE_ZEISS,
            seed=42,
            rows=rows,
            summary_override=summary,
        )
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(
            Stage3ReportError, r"disc\.dice_mean recomputed as .* but test_metrics"
        ):
            verify_run_summary(run)

    def test_a_missing_stored_summary_field_fails_as_a_report_error(self) -> None:
        domain = Domain.REFUGE_ZEISS
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        metrics_path = run_dir / "test_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        del payload["test"]["disc"]["dice_mean"]
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(Stage3ReportError, r"disc\.dice_mean"):
            verify_run_summary(run)

    def test_hd95_exclusions_are_carried_not_silently_dropped(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.REFUGE_ZEISS)
        rows = build_rows(image_ids, seed=42, degenerate_ids=image_ids[:3])
        write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.REFUGE_ZEISS,
            seed=42,
            rows=rows,
        )
        run = discover_stage3_runs([self.base])[0]

        summaries = verify_run_summary(run)

        self.assertEqual(summaries["disc"].sample_count, len(image_ids))
        self.assertEqual(summaries["disc"].hd95_excluded_count, 3)
        self.assertEqual(summaries["disc"].hd95_sample_count, len(image_ids) - 3)


# --------------------------------------------------------------------------
# Stage D: the confidence interval over seeds
# --------------------------------------------------------------------------


class SeedIntervalTests(unittest.TestCase):
    def test_interval_matches_a_manual_calculation(self) -> None:
        values = (0.900, 0.910, 0.920, 0.930, 0.940)
        expected_mean = statistics.fmean(values)
        expected_std = statistics.stdev(values)
        expected_half = T_CRITICAL_4_DF * expected_std / sqrt(len(values))

        interval = seed_confidence_interval("dice", SEEDS, values)

        self.assertAlmostEqual(interval.mean, expected_mean, places=12)
        self.assertAlmostEqual(interval.std, expected_std, places=12)
        self.assertAlmostEqual(interval.half_width, expected_half, places=12)
        self.assertAlmostEqual(interval.low, expected_mean - expected_half, places=12)
        self.assertAlmostEqual(interval.high, expected_mean + expected_half, places=12)
        self.assertEqual(interval.count, 5)

    def test_interval_uses_the_sample_standard_deviation(self) -> None:
        values = (0.900, 0.910, 0.920, 0.930, 0.940)

        interval = seed_confidence_interval("dice", SEEDS, values)

        population_std = statistics.pstdev(values)
        self.assertNotAlmostEqual(interval.std, population_std, places=6)
        self.assertAlmostEqual(interval.std, statistics.stdev(values), places=12)

    def test_one_seed_cannot_produce_an_interval(self) -> None:
        with self.assertRaisesRegex(Stage3ReportError, r"needs at least 2 seeds"):
            seed_confidence_interval("dice", (42,), (0.9,))

    def test_seed_and_value_counts_must_agree(self) -> None:
        with self.assertRaisesRegex(
            Stage3ReportError, r"got 2 values for 5 seeds"
        ):
            seed_confidence_interval("dice", SEEDS, (0.9, 0.91))


def load_grid(root: Path):
    """Discover, validate, and load a grid the way ``aggregate`` does."""

    runs = select_scientific_runs(discover_stage3_runs([root]))
    summaries = {run.identity: verify_run_summary(run) for run in runs}
    scores = {run.identity: load_run_scores(run, MANIFEST) for run in runs}
    return runs, summaries, scores


# --------------------------------------------------------------------------
# Stage E: the per-domain table
# --------------------------------------------------------------------------


class DomainTableTests(_TempCase):
    def test_one_cell_per_domain_and_structure_with_no_combined_figure(self) -> None:
        runs, summaries, scores = load_grid(build_grid(self.base))

        reports = build_domain_reports(runs, summaries, scores)

        self.assertEqual(len(reports), len(Domain) * 2)
        self.assertEqual(
            {(report.held_out_domain, report.structure) for report in reports},
            {(domain, structure) for domain in Domain for structure in ("disc", "cup")},
        )
        self.assertEqual({report.structure for report in reports}, {"disc", "cup"})

    def test_each_domain_gets_its_own_block_and_no_pooled_average(self) -> None:
        runs, summaries, scores = load_grid(build_grid(self.base))

        table = render_domain_table(build_domain_reports(runs, summaries, scores))

        for domain in Domain:
            self.assertIn(f"held-out domain: {domain.value}", table)
        self.assertNotIn("overall", table.lower())
        self.assertNotIn("combined", table.lower())
        self.assertNotIn("across domains", table.lower())

    def test_image_counts_are_the_locked_partition_sizes(self) -> None:
        runs, summaries, scores = load_grid(build_grid(self.base))

        reports = build_domain_reports(runs, summaries, scores)

        for report in reports:
            self.assertEqual(
                report.image_count,
                len(fold_test_image_ids(MANIFEST, report.held_out_domain)),
            )

    def test_a_missing_held_out_domain_is_refused(self) -> None:
        """A three-domain grid still renders tidily, so it has to be refused."""

        root = self.base / "partial"
        for domain in (Domain.DRISHTI_GS, Domain.REFUGE_ZEISS, Domain.RIM_ONE_DL):
            for seed in SEEDS:
                write_run(
                    root,
                    arm=PLAIN_ARM,
                    domain=domain,
                    seed=seed,
                    rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=seed),
                )
        runs, summaries, scores = load_grid(root)

        with self.assertRaisesRegex(
            Stage3ReportError,
            r"does not cover every held-out domain: missing=\['refuge_canon_val'\]",
        ):
            build_domain_reports(runs, summaries, scores)

    def test_an_incomplete_grid_can_be_reported_deliberately(self) -> None:
        root = self.base / "partial"
        for seed in SEEDS:
            write_run(
                root,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=seed,
                rows=build_rows(
                    fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS), seed=seed
                ),
            )
        runs, summaries, scores = load_grid(root)

        reports = build_domain_reports(
            runs, summaries, scores, expected_domains=None
        )

        self.assertEqual({report.held_out_domain for report in reports},
                         {Domain.DRISHTI_GS})

    def test_a_stable_hd95_subset_is_reported_as_stable(self) -> None:
        root = build_grid(self.base, degenerate=True)
        runs, summaries, scores = load_grid(root)

        table = render_domain_table(build_domain_reports(runs, summaries, scores))

        self.assertIn("the same images every seed", table)
        self.assertNotIn("different image sets", table)
        # A stable subset needs no restricted figure, so none is offered.
        self.assertNotIn("HD95*", table)

    def test_a_shifting_hd95_subset_is_called_out(self) -> None:
        """Equal exclusion counts must not be allowed to imply a stable subset."""

        root = self.base / "shifting"
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        for index, seed in enumerate(SEEDS):
            write_run(
                root,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=seed,
                rows=build_rows(
                    image_ids,
                    seed=seed,
                    degenerate_ids=image_ids[index * 2 : index * 2 + 2],
                ),
            )
        runs, summaries, scores = load_grid(root)

        reports = build_domain_reports(
            runs, summaries, scores, expected_domains=None
        )
        disc = next(report for report in reports if report.structure == "disc")

        # Every seed excludes exactly two images, but never the same two.
        self.assertEqual(disc.hd95_excluded_counts, (2, 2, 2, 2, 2))
        self.assertEqual(disc.hd95_sample_counts, (49, 49, 49, 49, 49))
        self.assertLess(disc.hd95_common_finite_count, 49)
        self.assertFalse(disc.hd95_subset_is_common)
        table = render_domain_table(reports)
        self.assertIn("different image sets", table)
        # A caveat alone leaves no usable number, so the restricted figure that
        # every seed does share a denominator for is reported beside it.
        self.assertIn("HD95*", table)
        self.assertIsNotNone(disc.hd95_common_interval)
        common = disc.hd95_common_interval
        assert common is not None
        self.assertEqual(common.count, len(SEEDS))

    def test_an_undefined_hd95_reports_nothing_rather_than_zero(self) -> None:
        """Every image degenerate in every seed: no mean, and a note saying why."""

        root = self.base / "allnan"
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        for seed in SEEDS:
            write_run(
                root,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=seed,
                rows=build_rows(image_ids, seed=seed, degenerate_ids=image_ids),
            )
        runs, summaries, scores = load_grid(root)

        reports = build_domain_reports(
            runs, summaries, scores, expected_domains=None
        )
        disc = next(report for report in reports if report.structure == "disc")

        self.assertIsNone(disc.intervals["hd95"])
        self.assertIsNotNone(disc.intervals["dice"])
        self.assertEqual(disc.hd95_sample_counts, (0,) * len(SEEDS))
        self.assertFalse(disc.hd95_subset_is_common)
        table = render_domain_table(reports)
        self.assertIn("undefined for all 51 images in every seed", table)
        self.assertNotIn("n/ano", table)
        self.assertNotIn("the same images every seed", table)
        for line in table.splitlines():
            self.assertEqual(line, line.rstrip())
            self.assertLessEqual(len(line), 92, line)

    def test_the_table_is_clean_and_fits_a_terminal(self) -> None:
        runs, summaries, scores = load_grid(build_grid(self.base, degenerate=True))

        table = render_domain_table(build_domain_reports(runs, summaries, scores))

        for line in table.splitlines():
            self.assertEqual(line, line.rstrip(), "trailing whitespace")
            # An 80-column terminal must not wrap a row, because a wrapped row
            # destroys the column alignment the numbers are read from.
            self.assertLessEqual(len(line), 80, line)

    def test_a_tiny_seed_spread_is_not_rendered_as_zero(self) -> None:
        interval = seed_confidence_interval(
            "hd95", SEEDS, (7.5000, 7.5001, 7.5002, 7.5003, 7.5004)
        )

        self.assertGreater(interval.std, 0.0)
        self.assertNotEqual(f"{interval.std:.3g}", "0.00")


# --------------------------------------------------------------------------
# Stage F: the paired between-arm substrate and test
# --------------------------------------------------------------------------


class PairedSubstrateTests(_TempCase):
    def test_substrate_averages_the_seeds_for_each_image(self) -> None:
        runs, _summaries, scores = load_grid(build_grid(self.base))

        substrate = build_paired_substrate(runs, scores)

        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        # One entry per image per structure -- not one per image per seed.
        self.assertEqual(
            len(substrate.image_ids(PLAIN_ARM, Domain.DRISHTI_GS, "disc")),
            len(image_ids),
        )
        chosen = image_ids[0]
        per_seed = [
            score.dice
            for run in runs
            if run.identity.held_out_domain is Domain.DRISHTI_GS
            for score in scores[run.identity]
            if score.image_id == chosen and score.structure == "disc"
        ]
        self.assertEqual(len(per_seed), len(SEEDS))
        self.assertAlmostEqual(
            substrate.values[(PLAIN_ARM, Domain.DRISHTI_GS, "disc", chosen)]["dice"],
            statistics.fmean(per_seed),
            places=12,
        )

    def test_hd95_is_not_offered_for_pairing(self) -> None:
        """HD95 per-run means are not over a common image set, so they cannot pair."""

        score = PerImageScore("i", "disc", 0.9, 0.8, 4.0, 0.99, 1, 1, 1, 1)

        with self.assertRaisesRegex(Stage3ReportError, r"not poolable per image"):
            score.metric_value("hd95")

    def test_paired_test_refuses_when_only_one_arm_exists(self) -> None:
        runs, _summaries, scores = load_grid(build_grid(self.base))
        substrate = build_paired_substrate(runs, scores)

        with self.assertRaisesRegex(
            Stage3ReportError,
            r"needs two arms scored on the same locked folds",
        ):
            paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

    def test_paired_test_refuses_to_compare_an_arm_with_itself(self) -> None:
        runs, _summaries, scores = load_grid(build_grid(self.base))
        substrate = build_paired_substrate(runs, scores)

        with self.assertRaisesRegex(Stage3ReportError, r"a paired test needs two arms"):
            paired_arm_test(substrate, PLAIN_ARM, PLAIN_ARM)


class PairedTestTests(_TempCase):
    def _two_arms(self, *, offset: float, jitter: float = 0.004):
        build_grid(self.base, arm=PLAIN_ARM)
        build_grid(self.base, arm=SPFILM_ARM, arm_offset=offset, seed_jitter=jitter)
        runs, _summaries, scores = load_grid(self.base)
        return build_paired_substrate(runs, scores)

    def test_a_real_shift_is_detected_on_every_fold(self) -> None:
        substrate = self._two_arms(offset=0.02)

        results = paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

        self.assertEqual(len(results), len(Domain) * 2)
        for result in results:
            self.assertGreater(result.median_difference, 0.0)
            self.assertLess(result.p_value_holm, 0.05)
            self.assertEqual(result.arm_a, PLAIN_ARM)
            self.assertEqual(result.arm_b, SPFILM_ARM)

    def test_two_identical_arms_produce_a_null_result(self) -> None:
        """A null result is a result; it must not be reported as a difference."""

        substrate = self._two_arms(offset=0.0)

        results = paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

        for result in results:
            self.assertEqual(result.median_difference, 0.0)
            self.assertEqual(result.n_informative_pairs, 0)
            self.assertEqual(result.p_value, 1.0)
            self.assertEqual(result.p_value_holm, 1.0)

    def test_pair_counts_are_the_locked_test_sizes(self) -> None:
        substrate = self._two_arms(offset=0.02)

        results = paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

        by_domain = {
            (result.held_out_domain, result.structure): result.n_pairs
            for result in results
        }
        for domain in Domain:
            expected = len(fold_test_image_ids(MANIFEST, domain))
            self.assertEqual(by_domain[(domain, "disc")], expected)
            self.assertEqual(by_domain[(domain, "cup")], expected)

    def test_the_pairs_the_test_actually_ranked_are_reported(self) -> None:
        """Signed-rank drops exact ties, so n alone would overstate the evidence."""

        import numpy as np

        from aggregate_stage3 import _paired_statistic

        differences = np.array([0.0] * 78 + [0.01, 0.02])
        _statistic, _p_value, informative = _paired_statistic(differences, "wilcoxon")

        self.assertEqual(informative, 2)

    def test_signed_rank_refuses_a_single_non_tied_pair(self) -> None:
        import numpy as np

        from aggregate_stage3 import _paired_statistic

        with self.assertRaisesRegex(
            Stage3ReportError, r"at least 2 non-tied pairs; only 1 of 40"
        ):
            _paired_statistic(np.array([0.0] * 39 + [0.01]), "wilcoxon")

    def test_the_t_test_can_be_chosen_deliberately(self) -> None:
        substrate = self._two_arms(offset=0.02)

        results = paired_arm_test(
            substrate, PLAIN_ARM, SPFILM_ARM, method="ttest"
        )

        for result in results:
            self.assertEqual(result.method, "ttest")
            self.assertEqual(result.n_informative_pairs, result.n_pairs)

    def test_t_test_handles_observed_exact_equality(self) -> None:
        substrate = self._two_arms(offset=0.0)

        results = paired_arm_test(
            substrate, PLAIN_ARM, SPFILM_ARM, method="ttest"
        )

        for result in results:
            self.assertEqual(result.statistic, 0.0)
            self.assertEqual(result.p_value, 1.0)
            self.assertEqual(result.p_value_holm, 1.0)

    def test_an_unknown_method_is_refused(self) -> None:
        substrate = self._two_arms(offset=0.02)

        with self.assertRaisesRegex(Stage3ReportError, r"Unknown paired method"):
            paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM, method="bootstrap")

    def test_mismatched_image_sets_cannot_be_paired(self) -> None:
        """Pairing is only valid on identical image sets, so it is asserted."""

        substrate = self._two_arms(offset=0.02)
        dropped = next(
            key
            for key in substrate.values
            if key[0] == SPFILM_ARM and key[1] is Domain.DRISHTI_GS
        )
        thinned = {
            key: value for key, value in substrate.values.items() if key != dropped
        }
        broken = type(substrate)(
            metrics=substrate.metrics,
            seed_counts=substrate.seed_counts,
            values=thinned,
        )

        with self.assertRaisesRegex(
            Stage3ReportError, r"scored on different images, so they cannot be paired"
        ):
            paired_arm_test(broken, PLAIN_ARM, SPFILM_ARM)

    def test_results_carry_holm_adjusted_p_values(self) -> None:
        substrate = self._two_arms(offset=0.02)

        results = paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

        for result in results:
            self.assertIsNotNone(result.p_value_holm)
            self.assertGreaterEqual(result.p_value_holm, result.p_value)
            self.assertLessEqual(result.p_value_holm, 1.0)

    def test_the_paired_table_shows_effect_size_beside_every_p_value(self) -> None:
        substrate = self._two_arms(offset=0.02)
        results = paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

        table = render_paired_table(results)

        self.assertIn("median d", table)
        self.assertIn("p(Holm)", table)
        for line in table.splitlines():
            self.assertEqual(line, line.rstrip())
            self.assertLessEqual(len(line), 92, line)


class HolmTests(unittest.TestCase):
    def test_known_values(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])

        self.assertEqual([round(value, 10) for value in adjusted],
                         [0.04, 0.09, 0.09, 0.20])

    def test_adjustment_is_monotone_conservative_and_capped(self) -> None:
        raw = [0.001, 0.02, 0.021, 0.3, 0.4, 0.5, 0.9, 0.95]

        adjusted = holm_adjust(raw)

        self.assertEqual(len(adjusted), len(raw))
        for value, original in zip(adjusted, raw):
            self.assertGreaterEqual(value, original)
            self.assertLessEqual(value, 1.0)
        ordered = [adjusted[index] for index in sorted(range(len(raw)), key=lambda i: raw[i])]
        self.assertEqual(ordered, sorted(ordered))

    def test_eight_tests_are_the_family_the_brief_implies(self) -> None:
        """4 held-out domains x 2 structures; the first p is multiplied by 8."""

        adjusted = holm_adjust([0.001] + [0.9] * 7)

        self.assertAlmostEqual(adjusted[0], 0.008, places=12)

    def test_an_out_of_range_p_value_is_refused(self) -> None:
        with self.assertRaisesRegex(Stage3ReportError, r"p-value out of range"):
            holm_adjust([0.5, 1.5])

    def test_no_tests_adjust_to_nothing(self) -> None:
        self.assertEqual(holm_adjust([]), ())


class PerImageSchemaTests(_TempCase):
    """RIM-ONE-DL output is not the bare metric schema, and must still load."""

    def test_the_optional_columns_are_the_ones_the_engine_appends(self) -> None:
        """A drifting engine column must break here, not in a report."""

        from spfilm.engine import RIM_ONE_DL_PER_IMAGE_CONTEXT

        self.assertEqual(OPTIONAL_PER_IMAGE_FIELDS, RIM_ONE_DL_PER_IMAGE_CONTEXT)

    def test_the_annotated_rim_one_dl_header_is_accepted(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL)
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(image_ids, seed=42),
        )
        header = (run_dir / "test_per_image_metrics.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertNotEqual(header.split(","), list(PER_IMAGE_FIELDNAMES))

        run = discover_stage3_runs([self.base])[0]
        scores = load_run_scores(run, MANIFEST)

        self.assertEqual(len(scores), 2 * len(image_ids))
        self.assertEqual(
            {score.image_id for score in scores}, set(image_ids)
        )

    def test_rim_context_rows_must_name_native_pixel_hd95(self) -> None:
        domain = Domain.RIM_ONE_DL
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        rewrite_first_csv_value(run_dir, "hd95_unit", "grid_px")
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(Stage3ReportError, "must be 'native_px'"):
            load_run_scores(run, MANIFEST)

    def test_an_unknown_extra_column_is_still_rejected(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=42,
            rows=build_rows(image_ids, seed=42),
        )
        csv_path = run_dir / "test_per_image_metrics.csv"
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        lines[0] += ",surprise"
        for index in range(1, len(lines)):
            lines[index] += ",1"
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(Stage3ReportError, "surprise"):
            load_run_scores(run, MANIFEST)

    def test_rim_one_dl_output_must_carry_its_context_columns(self) -> None:
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.RIM_ONE_DL,
            seed=42,
            rows=build_rows(
                fold_test_image_ids(MANIFEST, Domain.RIM_ONE_DL), seed=42
            ),
            context_columns=False,
            hd95_unit=HD95_UNIT_NATIVE,
            metric_frame=NATIVE_FRAME,
        )
        self.assertTrue((run_dir / "test_per_image_metrics.csv").is_file())
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(
            Stage3ReportError, "expected .* for rim_one_dl"
        ):
            load_run_scores(run, MANIFEST)

    def test_the_rim_header_is_the_base_schema_plus_context(self) -> None:
        self.assertEqual(
            RIM_PER_IMAGE_FIELDNAMES,
            (
                PER_IMAGE_FIELDNAMES[0],
                *OPTIONAL_PER_IMAGE_FIELDS,
                *PER_IMAGE_FIELDNAMES[1:],
            ),
        )

    def test_a_missing_metric_column_is_rejected(self) -> None:
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=42,
            rows=build_rows(image_ids, seed=42),
        )
        csv_path = run_dir / "test_per_image_metrics.csv"
        rows = [
            line.rsplit(",", 1)[0]
            for line in csv_path.read_text(encoding="utf-8").splitlines()
        ]
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        run = discover_stage3_runs([self.base])[0]

        with self.assertRaisesRegex(
            Stage3ReportError, "expected .* for drishti_gs"
        ):
            load_run_scores(run, MANIFEST)


class Hd95UnitTests(_TempCase):
    """HD95 is measured in different units per domain and must say which."""

    def test_the_unit_is_carried_from_the_run_and_shown(self) -> None:
        root = build_grid(self.base)
        runs, summaries, scores = load_grid(root)

        reports = build_domain_reports(runs, summaries, scores)
        by_domain = {
            report.held_out_domain: report.hd95_unit for report in reports
        }

        self.assertEqual(by_domain[Domain.RIM_ONE_DL], HD95_UNIT_NATIVE)
        self.assertEqual(by_domain[Domain.DRISHTI_GS], HD95_UNIT_GRID)
        table = render_domain_table(reports)
        self.assertIn(f"HD95 unit: {HD95_UNIT_NATIVE}", table)
        self.assertIn(f"HD95 unit: {HD95_UNIT_GRID}", table)

    def test_a_run_without_a_metric_frame_is_refused(self) -> None:
        """An unlabelled HD95 has no unit and must not reach a report."""

        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        run_dir = write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=42,
            rows=build_rows(image_ids, seed=42),
        )
        metrics_path = run_dir / "test_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        del payload["test"]["metric_frame"]
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(Stage3ReportError, "metric_frame"):
            discover_stage3_runs([self.base])

    def test_a_domain_cannot_claim_the_wrong_hd95_unit(self) -> None:
        domain = Domain.DRISHTI_GS
        write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
            hd95_unit=HD95_UNIT_NATIVE,
        )

        with self.assertRaisesRegex(
            Stage3ReportError, "current Stage 3 output requires"
        ):
            discover_stage3_runs([self.base])

    def test_seeds_measured_on_different_grids_cannot_be_pooled(self) -> None:
        """A cell mixing resolutions is not five estimates of one quantity."""

        root = self.base / "mixed"
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        for seed in SEEDS:
            write_run(
                root,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=seed,
                rows=build_rows(image_ids, seed=seed),
                metric_frame=(
                    GRID_FRAME.replace("512", "256") if seed == 46 else GRID_FRAME
                ),
            )
        runs, summaries, scores = load_grid(root)

        with self.assertRaisesRegex(
            Stage3ReportError, "spans more than one metric frame"
        ):
            build_domain_reports(
                runs, summaries, scores, expected_domains=None
            )


class Hd95CommonSubsetTests(_TempCase):
    """The HD95 figure whose five means actually share a denominator."""

    def _shifting(self):
        root = self.base / "shifting"
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        for index, seed in enumerate(SEEDS):
            write_run(
                root,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=seed,
                rows=build_rows(
                    image_ids,
                    seed=seed,
                    degenerate_ids=image_ids[index * 2 : index * 2 + 2],
                ),
            )
        runs, summaries, scores = load_grid(root)
        return image_ids, scores, build_domain_reports(
            runs, summaries, scores, expected_domains=None
        )

    def test_the_common_subset_mean_is_over_the_intersection(self) -> None:
        image_ids, scores, reports = self._shifting()
        disc = next(report for report in reports if report.structure == "disc")

        common = sorted(
            set.intersection(
                *[
                    {
                        score.image_id
                        for score in run_scores
                        if score.structure == "disc"
                        and not statistics_isnan(score.hd95)
                    }
                    for run_scores in scores.values()
                ]
            )
        )
        expected = []
        for identity in sorted(scores, key=lambda key: key.run_seed):
            by_id = {
                score.image_id: score.hd95
                for score in scores[identity]
                if score.structure == "disc"
            }
            expected.append(
                statistics.fmean([by_id[image_id] for image_id in common])
            )

        self.assertEqual(disc.hd95_common_finite_count, len(common))
        self.assertAlmostEqual(
            disc.hd95_common_interval.mean,  # type: ignore[union-attr]
            statistics.fmean(expected),
            places=12,
        )
        self.assertAlmostEqual(
            disc.hd95_common_interval.half_width,  # type: ignore[union-attr]
            T_CRITICAL_4_DF * statistics.stdev(expected) / sqrt(len(expected)),
            places=12,
        )

    def test_a_stable_subset_offers_no_separate_figure(self) -> None:
        root = build_grid(self.base, degenerate=True)
        runs, summaries, scores = load_grid(root)

        reports = build_domain_reports(runs, summaries, scores)

        for report in reports:
            if report.hd95_subset_is_common:
                self.assertNotIn("HD95*", render_domain_table([report]))

    def test_no_common_finite_image_does_not_promise_an_hd95_star_row(self) -> None:
        root = self.base / "no-common"
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        for index, seed in enumerate(SEEDS):
            finite_id = image_ids[index]
            write_run(
                root,
                arm=PLAIN_ARM,
                domain=Domain.DRISHTI_GS,
                seed=seed,
                rows=build_rows(
                    image_ids,
                    seed=seed,
                    degenerate_ids=[
                        image_id for image_id in image_ids if image_id != finite_id
                    ],
                ),
            )
        runs, summaries, scores = load_grid(root)
        reports = build_domain_reports(
            runs, summaries, scores, expected_domains=None
        )

        table = render_domain_table(reports)

        self.assertIn("no image is finite in every seed", table)
        self.assertNotIn("HD95*", table)

    def test_the_common_subset_reaches_the_serialised_report(self) -> None:
        _, _, reports = self._shifting()
        disc = next(report for report in reports if report.structure == "disc")

        payload = disc.as_dict()

        self.assertEqual(payload["hd95_unit"], HD95_UNIT_GRID)
        self.assertIsNotNone(payload["hd95_common_subset"])
        self.assertEqual(
            payload["hd95_common_subset"]["mean"],
            disc.hd95_common_interval.mean,  # type: ignore[union-attr]
        )


class PartialGridTests(_TempCase):
    """A grid that is still filling in must be inspectable, but never quietly."""

    def _ragged(self):
        root = self.base / "ragged"
        for domain, seeds in (
            (Domain.DRISHTI_GS, SEEDS),
            (Domain.REFUGE_ZEISS, SEEDS[:3]),
        ):
            image_ids = fold_test_image_ids(MANIFEST, domain)
            for seed in seeds:
                write_run(
                    root,
                    arm=PLAIN_ARM,
                    domain=domain,
                    seed=seed,
                    rows=build_rows(image_ids, seed=seed),
                )
        return load_grid(root)

    def test_a_ragged_grid_is_refused_by_default(self) -> None:
        runs, summaries, scores = self._ragged()

        with self.assertRaisesRegex(
            Stage3ReportError, "does not use the same seeds"
        ):
            build_domain_reports(
                runs, summaries, scores, expected_domains=None
            )

    def test_a_ragged_grid_can_be_reported_deliberately(self) -> None:
        """The documented opt-out has to actually reach a table."""

        runs, summaries, scores = self._ragged()

        reports = build_domain_reports(
            runs,
            summaries,
            scores,
            expected_seeds=None,
            expected_domains=None,
        )
        by_domain = {
            (report.held_out_domain, report.structure): report
            for report in reports
        }

        self.assertEqual(len(by_domain[(Domain.DRISHTI_GS, "disc")].seeds), 5)
        self.assertEqual(len(by_domain[(Domain.REFUGE_ZEISS, "disc")].seeds), 3)
        render_domain_table(reports)

    def test_one_seed_carries_a_mean_but_never_an_interval(self) -> None:
        root = self.base / "single"
        image_ids = fold_test_image_ids(MANIFEST, Domain.DRISHTI_GS)
        write_run(
            root,
            arm=PLAIN_ARM,
            domain=Domain.DRISHTI_GS,
            seed=42,
            rows=build_rows(image_ids, seed=42),
        )
        runs, summaries, scores = load_grid(root)

        reports = build_domain_reports(
            runs,
            summaries,
            scores,
            expected_seeds=None,
            expected_domains=None,
        )
        disc = next(report for report in reports if report.structure == "disc")

        self.assertIsNone(disc.intervals["dice"])
        self.assertIsNotNone(disc.point_means["dice"])
        table = render_domain_table(reports)
        self.assertIn("no interval exists", table)
        # The mean is still shown; what must never appear is a fabricated zero.
        self.assertIn(f"{disc.point_means['dice']:.4f}", table)

    def test_one_seed_csv_carries_the_mean_but_leaves_interval_fields_blank(
        self,
    ) -> None:
        root = self.base / "single-csv"
        domain = Domain.DRISHTI_GS
        write_run(
            root,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        report = aggregate(
            [root],
            DEFAULT_MANIFEST,
            expected_seeds=None,
            expected_domains=None,
        )
        target = self.base / "partial.csv"

        write_report_csv(report.domain_reports, target)

        with target.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        disc_dice = next(
            row
            for row in rows
            if row["structure"] == "disc" and row["metric"] == "dice"
        )
        self.assertNotEqual(disc_dice["mean"], "")
        self.assertEqual(disc_dice["seed_count"], "1")
        self.assertEqual(disc_dice["std_across_seeds"], "")
        self.assertEqual(disc_dice["ci_low"], "")
        self.assertEqual(disc_dice["ci_high"], "")


class ManifestIdentityTests(_TempCase):
    """Membership matching is not proof the runs used this manifest."""

    def test_a_run_from_another_manifest_is_refused(self) -> None:
        root = build_grid(self.base)
        retag_run(root / "drishti_gs" / "seed_42", "manifest_sha256", "d" * 64)

        with self.assertRaisesRegex(
            Stage3ReportError, "scored against a different manifest"
        ):
            aggregate([root], DEFAULT_MANIFEST)

    def test_a_dirty_working_tree_is_surfaced(self) -> None:
        root = build_grid(self.base)
        for domain in Domain:
            for seed in SEEDS:
                retag_run(
                    root / domain.value / f"seed_{seed}",
                    "git_revision",
                    "c" * 40 + "-dirty",
                )

        report = aggregate([root], DEFAULT_MANIFEST)

        self.assertTrue(
            any("dirty working tree" in warning for warning in report.warnings),
            report.warnings,
        )


class UnequalSeedPairingTests(_TempCase):
    """Pairing a five-seed mean against a three-seed mean is not a fair test."""

    def _lopsided(self):
        build_grid(self.base, arm=PLAIN_ARM)
        build_grid(self.base, arm=SPFILM_ARM, arm_offset=0.02, seeds=SEEDS[:3])
        runs, _summaries, scores = load_grid(self.base)
        return build_paired_substrate(runs, scores)

    def test_unequal_seed_counts_are_refused(self) -> None:
        substrate = self._lopsided()

        with self.assertRaisesRegex(Stage3ReportError, "averaged"):
            paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

    def test_unequal_seed_counts_can_be_compared_deliberately(self) -> None:
        substrate = self._lopsided()

        results = paired_arm_test(
            substrate, PLAIN_ARM, SPFILM_ARM, allow_unequal_seeds=True
        )

        self.assertEqual(len(results), 8)
        self.assertEqual({result.seed_count_a for result in results}, {5})
        self.assertEqual({result.seed_count_b for result in results}, {3})
        # The asymmetry must be visible in the output, not just tolerated.
        self.assertIn("not matched", render_paired_table(results))

    def test_seed_counts_are_compared_domain_by_domain(self) -> None:
        for domain in Domain:
            image_ids = fold_test_image_ids(MANIFEST, domain)
            seeds_a = SEEDS if domain is Domain.DRISHTI_GS else SEEDS[:3]
            seeds_b = SEEDS[:3] if domain is Domain.DRISHTI_GS else SEEDS
            for arm, seeds in ((PLAIN_ARM, seeds_a), (SPFILM_ARM, seeds_b)):
                for seed in seeds:
                    write_run(
                        self.base / arm,
                        arm=arm,
                        domain=domain,
                        seed=seed,
                        rows=build_rows(image_ids, seed=seed),
                    )
        runs, _summaries, scores = load_grid(self.base)
        substrate = build_paired_substrate(runs, scores)

        with self.assertRaisesRegex(
            Stage3ReportError, "different seed counts in these domains"
        ):
            paired_arm_test(substrate, PLAIN_ARM, SPFILM_ARM)

    def test_the_paired_table_also_fits_a_terminal(self) -> None:
        build_grid(self.base, arm=PLAIN_ARM)
        build_grid(self.base, arm=SPFILM_ARM, arm_offset=0.02)
        runs, _summaries, scores = load_grid(self.base)
        results = paired_arm_test(
            build_paired_substrate(runs, scores), PLAIN_ARM, SPFILM_ARM
        )

        table = render_paired_table(results)

        for line in table.splitlines():
            self.assertEqual(line, line.rstrip(), "trailing whitespace")
            self.assertLessEqual(len(line), 80, line)

    def test_matched_arms_state_the_seed_count(self) -> None:
        build_grid(self.base, arm=PLAIN_ARM)
        build_grid(self.base, arm=SPFILM_ARM, arm_offset=0.02)
        runs, _summaries, scores = load_grid(self.base)

        results = paired_arm_test(
            build_paired_substrate(runs, scores), PLAIN_ARM, SPFILM_ARM
        )

        self.assertIn("mean over 5 seeds", render_paired_table(results))


class MarkdownReportTests(_TempCase):
    """The written report must quote the aggregation, never restate it."""

    def _report(self):
        build_grid(self.base, arm=PLAIN_ARM)
        return aggregate([self.base], DEFAULT_MANIFEST)

    def test_every_reported_figure_comes_from_the_aggregation(self) -> None:
        report = self._report()

        text = render_markdown_report(report)

        for cell in report.domain_reports:
            interval = cell.intervals["dice"]
            self.assertIn(f"**{interval.mean:.4f}**", text)
            self.assertIn(
                f"[{interval.low:.4f}, {interval.high:.4f}]", text
            )

    def test_the_house_style_sections_are_all_present(self) -> None:
        text = render_markdown_report(self._report())

        for heading in (
            "**Evidence boundary.**",
            "## 1. Objective",
            "## 2. Dataset and split",
            "## 3. Model and training setup",
            "## 4. Per-epoch results",
            "## 5. Test results",
            "## 6. Findings",
            "## 7. Sanity check against the literature",
            "## 8. Limitations and reproducibility",
            "## 9. Appendix: run inventory",
        ):
            self.assertIn(heading, text)
        for footnote in ("[^manifest]:", "[^runs]:", "[^method]:"):
            self.assertIn(footnote, text)

    def test_the_training_completion_table_is_populated(self) -> None:
        text = render_markdown_report(self._report())

        self.assertIn("Best epoch", text)
        self.assertIn("| 42 | 259 | 300 / 300 | 146 | 1800.0 |", text)

    def test_two_arm_report_does_not_call_every_arm_plain_unet(self) -> None:
        build_grid(self.base, arm=PLAIN_ARM)
        build_grid(self.base, arm=SPFILM_ARM, arm_offset=0.02)
        report = aggregate(
            [self.base],
            DEFAULT_MANIFEST,
            paired_arms=(PLAIN_ARM, SPFILM_ARM),
            paired_method="ttest",
        )

        text = render_markdown_report(report)

        self.assertIn("configured model arms", text)
        self.assertIn(f"### Arm: `{PLAIN_ARM}`", text)
        self.assertIn(f"### Arm: `{SPFILM_ARM}`", text)
        self.assertIn("paired t-test", text)
        self.assertNotIn("ttest signed-rank", text)

    def test_interpretation_is_left_to_the_author(self) -> None:
        """A generated sentence about meaning is the one thing not wanted."""

        text = render_markdown_report(self._report())

        findings = text.split("## 6. Findings")[1].split("## 7.")[0]
        self.assertIn("TODO", findings)

    def test_the_hd95_unit_split_is_stated_as_a_limitation(self) -> None:
        text = render_markdown_report(self._report())

        limitations = text.split("## 8. Limitations")[1]
        self.assertIn("more than one unit", limitations)

    def test_a_single_arm_report_says_it_cannot_compare(self) -> None:
        text = render_markdown_report(self._report())

        self.assertIn("Only one conditioning arm", text)

    def test_one_seed_report_never_claims_a_confidence_interval(self) -> None:
        domain = Domain.DRISHTI_GS
        write_run(
            self.base,
            arm=PLAIN_ARM,
            domain=domain,
            seed=42,
            rows=build_rows(fold_test_image_ids(MANIFEST, domain), seed=42),
        )
        report = aggregate(
            [self.base],
            DEFAULT_MANIFEST,
            expected_seeds=None,
            expected_domains=None,
        )

        text = render_markdown_report(report)

        self.assertIn("one seed, no interval", text)
        self.assertNotIn("0 degrees of freedom", text)

    def test_the_report_is_written_where_asked(self) -> None:
        target = self.base / "reports" / "s3_lodo_report.md"

        written = write_markdown_report(self._report(), target)

        self.assertEqual(written, target)
        self.assertTrue(target.is_file())
        self.assertIn("# Step 3", target.read_text(encoding="utf-8"))


class EndToEndTests(_TempCase):
    def test_the_full_grid_aggregates_and_reports(self) -> None:
        build_grid(self.base, arm=PLAIN_ARM, degenerate=True)

        report = aggregate([self.base], DEFAULT_MANIFEST)

        self.assertEqual(len(report.runs), 20)
        self.assertEqual(len(report.domain_reports), 8)
        self.assertEqual(report.paired_results, ())
        self.assertIn("needs two conditioning arms", report.paired_note)
        self.assertEqual(report.warnings, ())
        payload = report.as_dict()
        self.assertEqual(payload["run_count"], 20)
        self.assertEqual(len(payload["per_domain"]), 8)
        self.assertEqual(payload["paired_test"]["results"], [])

    def test_the_cli_writes_json_csv_and_markdown_from_one_validation_pass(
        self,
    ) -> None:
        root = build_grid(self.base, arm=PLAIN_ARM)
        json_path = self.base / "outputs" / "stage3.json"
        csv_path = self.base / "outputs" / "stage3.csv"
        report_path = self.base / "outputs" / "stage3.md"

        with redirect_stdout(io.StringIO()):
            status = main(
                [
                    "--runs",
                    str(root),
                    "--json-out",
                    str(json_path),
                    "--csv-out",
                    str(csv_path),
                    "--report-out",
                    str(report_path),
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(json_path.read_text())["run_count"], 20)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 24)
        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("**Evidence boundary.**", report_text)
        self.assertIn("## 9. Appendix: run inventory", report_text)

    def test_two_arms_aggregate_and_pair(self) -> None:
        build_grid(self.base, arm=PLAIN_ARM)
        build_grid(self.base, arm=SPFILM_ARM, arm_offset=0.02)

        report = aggregate(
            [self.base], DEFAULT_MANIFEST, paired_arms=(PLAIN_ARM, SPFILM_ARM)
        )

        self.assertEqual(len(report.runs), 40)
        self.assertEqual(len(report.domain_reports), 16)
        self.assertEqual(len(report.paired_results), 8)
        payload = report.as_dict()
        self.assertEqual(sorted(payload["arms"]), sorted([PLAIN_ARM, SPFILM_ARM]))
        self.assertEqual(len(payload["paired_test"]["results"]), 8)
        json.dumps(payload)

    def test_an_arm_spanning_two_configs_is_refused(self) -> None:
        """A seed interval means nothing if the seeds ran different procedures."""

        root = build_grid(self.base)
        retag_run(root / "drishti_gs" / "seed_42", "config_sha256", "d" * 64)

        with self.assertRaisesRegex(
            Stage3ReportError, "spans more than one config"
        ):
            aggregate([root], DEFAULT_MANIFEST)

    def test_an_arm_spanning_two_revisions_is_refused(self) -> None:
        root = build_grid(self.base)
        retag_run(root / "drishti_gs" / "seed_42", "git_revision", "e" * 40)

        with self.assertRaisesRegex(
            Stage3ReportError, "spans more than one git revision"
        ):
            aggregate([root], DEFAULT_MANIFEST)

    def test_warnings_reach_the_serialised_report(self) -> None:
        root = build_grid(self.base)
        for domain in Domain:
            for seed in SEEDS:
                retag_run(
                    root / domain.value / f"seed_{seed}",
                    "git_revision",
                    "unavailable",
                )

        report = aggregate([root], DEFAULT_MANIFEST)

        self.assertTrue(report.warnings)
        self.assertEqual(report.as_dict()["warnings"], list(report.warnings))


if __name__ == "__main__":
    unittest.main()
