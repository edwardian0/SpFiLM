"""Contract tests for the train-on-one, test-on-three aggregator.

The aggregator's job is to refuse to produce a number it cannot justify. Most of
these tests therefore assert that it *raises*: a silently dropped run, a quietly
substituted image, or a stored summary that disagrees with the CSV it came from
would each turn into a wrong mean in a thesis table.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_stage3_1_3 import (  # noqa: E402
    Stage3SingleSourceReportError,
    build_cell_reports,
    build_run,
    discover_runs,
    render_matrix,
    seed_confidence_interval,
    select_scientific_runs,
    verify_run_membership,
    verify_run_summary,
)
from spfilm.lodo import Domain  # noqa: E402
from spfilm.metrics import summarise_per_image_csv  # noqa: E402
from spfilm.single_source import (  # noqa: E402
    SingleSourceManifest,
    write_single_source_manifest,
)
from spfilm.lodo import DomainPartitions, SampleKey  # noqa: E402


SEEDS = (42, 43)
PARENT_SHA = "a" * 64
CONFIG_SHA = "c" * 64


def _partitions() -> tuple[DomainPartitions, ...]:
    """Four tiny domains with a 2/1/2 budget, shaped like the real manifest."""

    partitions = []
    for index, domain in enumerate(sorted(Domain, key=lambda item: item.value)):
        prefix = domain.value[:3]
        partitions.append(
            DomainPartitions(
                domain=domain,
                train=tuple(
                    SampleKey(domain, f"{prefix}_tr{n}") for n in range(2)
                ),
                val=(SampleKey(domain, f"{prefix}_va0"),),
                test=tuple(SampleKey(domain, f"{prefix}_te{n}") for n in range(2)),
            )
        )
    return tuple(partitions)


def _manifest() -> SingleSourceManifest:
    strata = {}
    for partition in _partitions():
        for name in ("train", "val", "test"):
            for key in getattr(partition, name):
                strata[key] = "all"
    return SingleSourceManifest.build(
        PARENT_SHA, _partitions(), 2, 1, 2, strata, 42
    )


def _write_run(
    base: Path,
    manifest: SingleSourceManifest,
    manifest_sha: str,
    source: Domain,
    seed: int,
    *,
    completed: str | None = None,
    dice: float = 0.8,
) -> Path:
    fold = next(f for f in manifest.folds if f.source_domain == source)
    run = base / f"single_s3_{source.value}_seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    by_domain = {}
    for domain, samples in fold.tests:
        csv_path = run / f"test_{domain.value}_per_image_metrics.csv"
        rows = [
            {
                "image_id": sample.sample_id,
                "structure": structure,
                "dice": f"{dice:.12g}",
                "iou": f"{dice / (2 - dice):.12g}",
                "hd95": "10",
                "acc": "0.95",
                "tp": 100,
                "fp": 10,
                "fn": 9,
                "tn": 900,
            }
            for sample in samples
            for structure in ("disc", "cup")
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        block = dict(summarise_per_image_csv(csv_path))
        block["evaluated_sample_count"] = len(samples)
        if domain.value == "rim_one_dl":
            block["hd95_unit"] = "native pixels"
        by_domain[domain.value] = block
    payload = {
        "test_by_domain": by_domain,
        "test_pooled": {"note": "ignored"},
        "single_source": {
            "protocol": "single_source_locked_multi_target_test",
            "arm": "stage3_single_source_plain_unet",
            "source_domain": source.value,
            "target_domains": [d.value for d, _ in fold.tests],
            "run_seed": seed,
            "budget": {"train": 2, "val": 1, "test": 2, "subsample_seed": 42},
            "manifest_sha256": manifest_sha,
            "parent_manifest_sha256": PARENT_SHA,
            "config_sha256": CONFIG_SHA,
            "git_revision": "test",
            "smoke_rehearsal": False,
            "scientific_result": True,
            "completed_at_utc": completed or f"2026-09-05T0{seed - 42}:00:00+00:00",
        },
    }
    (run / "test_metrics.json").write_text(json.dumps(payload, indent=2))
    return run


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True)
        )
        self.manifest = _manifest()
        self.manifest_path = self.directory / "single_source_manifest.json"
        write_single_source_manifest(self.manifest, self.manifest_path)
        import hashlib

        self.manifest_sha = hashlib.sha256(
            self.manifest_path.read_bytes()
        ).hexdigest()
        self.runs_root = self.directory / "runs"
        for source in sorted(Domain, key=lambda item: item.value):
            for seed in SEEDS:
                _write_run(
                    self.runs_root,
                    self.manifest,
                    self.manifest_sha,
                    source,
                    seed,
                )

    def _selected(self):
        return select_scientific_runs(discover_runs([self.runs_root]), SEEDS)


class SeedIntervalTests(unittest.TestCase):
    def test_mean_and_spread_match_hand_computation(self) -> None:
        interval = seed_confidence_interval("dice", (42, 43, 44), (0.8, 0.9, 1.0))
        self.assertAlmostEqual(interval.mean, 0.9)
        self.assertAlmostEqual(interval.std, 0.1)
        self.assertLess(interval.low, interval.mean)
        self.assertGreater(interval.high, interval.mean)

    def test_one_seed_is_refused(self) -> None:
        with self.assertRaises(Stage3SingleSourceReportError):
            seed_confidence_interval("dice", (42,), (0.8,))

    def test_non_finite_is_refused(self) -> None:
        with self.assertRaises(Stage3SingleSourceReportError):
            seed_confidence_interval("dice", (42, 43), (0.8, float("nan")))


class DiscoveryTests(Fixture):
    def test_discovers_every_source_and_seed(self) -> None:
        runs = self._selected()
        self.assertEqual(len(runs), len(Domain) * len(SEEDS))

    def test_bare_test_key_from_the_old_schema_is_refused(self) -> None:
        path = next(self.runs_root.rglob("test_metrics.json"))
        payload = json.loads(path.read_text())
        payload["test"] = {"legacy": True}
        path.write_text(json.dumps(payload))
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            build_run(path)
        self.assertIn("superseded schema", str(caught.exception))

    def test_smoke_rehearsal_is_ignored_not_reported(self) -> None:
        path = next(self.runs_root.rglob("test_metrics.json"))
        payload = json.loads(path.read_text())
        payload["single_source"]["smoke_rehearsal"] = True
        path.write_text(json.dumps(payload))
        self.assertIsNone(build_run(path))

    def test_incomplete_seed_grid_is_refused(self) -> None:
        import shutil

        shutil.rmtree(
            next(
                path
                for path in self.runs_root.iterdir()
                if path.name.endswith("_seed_43")
            )
        )
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            self._selected()
        self.assertIn("missing", str(caught.exception))

    def test_relaunched_run_keeps_only_the_later_completion(self) -> None:
        source = sorted(Domain, key=lambda item: item.value)[0]
        _write_run(
            self.runs_root / "relaunch",
            self.manifest,
            self.manifest_sha,
            source,
            42,
            completed="2026-09-09T00:00:00+00:00",
            dice=0.1,
        )
        runs = self._selected()
        self.assertEqual(len(runs), len(Domain) * len(SEEDS))
        kept = next(
            run
            for run in runs
            if run.identity.source_domain == source and run.identity.run_seed == 42
        )
        self.assertEqual(kept.identity.completed_at_utc, "2026-09-09T00:00:00+00:00")

    def test_runs_from_two_manifests_are_refused(self) -> None:
        path = next(self.runs_root.rglob("test_metrics.json"))
        payload = json.loads(path.read_text())
        payload["single_source"]["manifest_sha256"] = "b" * 64
        path.write_text(json.dumps(payload))
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            self._selected()
        self.assertIn("manifest_sha256", str(caught.exception))

    def test_wrong_hd95_unit_is_refused(self) -> None:
        path = next(
            p
            for p in self.runs_root.rglob("test_metrics.json")
            if "rim_one_dl_seed" not in p.parent.name
        )
        payload = json.loads(path.read_text())
        target = next(
            name
            for name in payload["test_by_domain"]
            if name != "rim_one_dl"
        )
        payload["test_by_domain"][target]["hd95_unit"] = "native pixels"
        path.write_text(json.dumps(payload))
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            build_run(path)
        self.assertIn("HD95", str(caught.exception))


class VerificationTests(Fixture):
    def test_membership_must_match_the_locked_partition(self) -> None:
        run = self._selected()[0]
        target = run.target_domains[0]
        path = run.per_image_csv[target]
        lines = path.read_text().splitlines()
        lines[1] = "SUBSTITUTED" + lines[1][lines[1].index(",") :]
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            verify_run_membership(run, self.manifest)
        self.assertIn("do not match the locked test partition", str(caught.exception))

    def test_membership_passes_on_untouched_runs(self) -> None:
        for run in self._selected():
            counts = verify_run_membership(run, self.manifest)
            self.assertEqual(set(counts), set(run.target_domains))
            self.assertTrue(all(count == 2 for count in counts.values()))

    def test_stored_summary_must_agree_with_its_csv(self) -> None:
        run = self._selected()[0]
        payload = json.loads(run.metrics_path.read_text())
        target = run.target_domains[0].value
        payload["test_by_domain"][target]["disc"]["dice_mean"] += 0.05
        run.metrics_path.write_text(json.dumps(payload))
        reloaded = build_run(run.metrics_path)
        with self.assertRaises(Stage3SingleSourceReportError) as caught:
            verify_run_summary(reloaded)
        self.assertIn("dice_mean", str(caught.exception))


class MatrixTests(Fixture):
    def test_matrix_diagonal_is_empty_and_targets_are_separate(self) -> None:
        runs = self._selected()
        counts = {run.label: verify_run_membership(run, self.manifest) for run in runs}
        recomputed = {run.label: verify_run_summary(run) for run in runs}
        cells = build_cell_reports(runs, recomputed, counts)
        # four sources x three targets x two structures
        self.assertEqual(len(cells), len(Domain) * (len(Domain) - 1) * 2)
        self.assertTrue(
            all(cell.source_domain != cell.target_domain for cell in cells)
        )
        table = render_matrix(cells, "disc")
        for domain in Domain:
            self.assertIn(f"`{domain.value}`", table)
        # one em dash per row: the source domain is never its own target
        self.assertEqual(table.count("—"), len(Domain))

    def test_rim_one_dl_keeps_its_native_frame(self) -> None:
        runs = self._selected()
        counts = {run.label: verify_run_membership(run, self.manifest) for run in runs}
        recomputed = {run.label: verify_run_summary(run) for run in runs}
        cells = build_cell_reports(runs, recomputed, counts)
        for cell in cells:
            expected = (
                "native pixels"
                if cell.target_domain.value == "rim_one_dl"
                else "letterboxed-grid pixels"
            )
            self.assertEqual(cell.hd95_unit, expected)


if __name__ == "__main__":
    unittest.main()
