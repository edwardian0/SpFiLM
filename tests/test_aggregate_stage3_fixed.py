"""Contract tests for the fixed-budget aggregator and its paired significance test.

The brief requires a paired test on per-image Dice over the same test images.
Most of these tests assert that the tool *refuses* when that precondition is not
met: pairing across different image sets, or across arms that averaged different
numbers of seeds, produces a number that looks fine and means nothing.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aggregate_stage3_fixed import (  # noqa: E402
    ALPHA,
    POOLED_ARM,
    FixedLodoReportError,
    Substrate,
    build_fixed_run,
    discover_fixed_runs,
    holm_adjust,
    paired_tests,
    select_fixed_runs,
)
from spfilm.lodo import Domain  # noqa: E402


SEEDS = (42, 43)
MANIFEST_SHA = "a" * 64
HELD_OUT = Domain.DRISHTI_GS
SOURCES = ("refuge_canon_val", "refuge_zeiss", "rim_one_dl")


def _substrate(
    pooled: dict[str, float],
    singles: dict[str, dict[str, float]],
    seed_counts: dict[str, int] | None = None,
) -> Substrate:
    values = {}
    counts = {}
    for image_id, score in pooled.items():
        values[(POOLED_ARM, HELD_OUT, "disc", image_id)] = {"dice": score, "iou": score}
    counts[(POOLED_ARM, HELD_OUT)] = (seed_counts or {}).get(POOLED_ARM, len(SEEDS))
    for arm, scores in singles.items():
        for image_id, score in scores.items():
            values[(arm, HELD_OUT, "disc", image_id)] = {"dice": score, "iou": score}
        counts[(arm, HELD_OUT)] = (seed_counts or {}).get(arm, len(SEEDS))
    return Substrate(seed_counts=counts, values=values)


class HolmTests(unittest.TestCase):
    def test_preserves_input_order(self) -> None:
        adjusted = holm_adjust([0.04, 0.01, 0.03])
        self.assertEqual(len(adjusted), 3)
        # smallest raw p gets multiplied by the full count
        self.assertAlmostEqual(adjusted[1], 0.03)

    def test_is_monotone_non_decreasing_in_rank(self) -> None:
        raw = [0.001, 0.01, 0.02, 0.04]
        adjusted = holm_adjust(raw)
        ordered = [adjusted[i] for i in sorted(range(4), key=lambda i: raw[i])]
        self.assertEqual(ordered, sorted(ordered))

    def test_never_exceeds_one(self) -> None:
        self.assertTrue(all(p <= 1.0 for p in holm_adjust([0.4, 0.5, 0.9])))

    def test_empty_family(self) -> None:
        self.assertEqual(holm_adjust([]), ())

    def test_out_of_range_is_refused(self) -> None:
        with self.assertRaises(FixedLodoReportError):
            holm_adjust([0.5, 1.5])


class PairedTestTests(unittest.TestCase):
    def test_detects_a_known_uniform_shift(self) -> None:
        base = {f"img{i}": 0.50 + i * 0.001 for i in range(40)}
        shifted = {k: v + 0.05 for k, v in base.items()}
        results = paired_tests(_substrate(shifted, {"single_a": base}))
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertAlmostEqual(result.mean_difference, 0.05, places=9)
        self.assertLess(result.p_adjusted, ALPHA)
        self.assertTrue(result.significant)

    def test_identical_arms_are_not_significant(self) -> None:
        base = {f"img{i}": 0.5 + i * 0.001 for i in range(40)}
        results = paired_tests(_substrate(dict(base), {"single_a": base}))
        self.assertAlmostEqual(results[0].mean_difference, 0.0)
        self.assertEqual(results[0].p_value, 1.0)
        self.assertFalse(results[0].significant)

    def test_sign_of_difference_is_pooled_minus_single(self) -> None:
        base = {f"img{i}": 0.6 for i in range(30)}
        worse = {k: v - 0.1 for k, v in base.items()}
        results = paired_tests(_substrate(worse, {"single_a": base}))
        self.assertLess(results[0].mean_difference, 0)

    def test_mismatched_image_sets_are_refused(self) -> None:
        pooled = {f"img{i}": 0.5 for i in range(30)}
        single = {f"img{i}": 0.5 for i in range(29)}
        with self.assertRaises(FixedLodoReportError) as caught:
            paired_tests(_substrate(pooled, {"single_a": single}))
        self.assertIn("identical images", str(caught.exception))

    def test_unequal_seed_counts_are_refused(self) -> None:
        base = {f"img{i}": 0.5 for i in range(30)}
        with self.assertRaises(FixedLodoReportError) as caught:
            paired_tests(
                _substrate(
                    dict(base),
                    {"single_a": base},
                    seed_counts={POOLED_ARM: 5, "single_a": 3},
                )
            )
        self.assertIn("same estimator", str(caught.exception))

    def test_missing_pooled_arm_is_refused(self) -> None:
        substrate = Substrate(
            seed_counts={("single_a", HELD_OUT): 5},
            values={("single_a", HELD_OUT, "disc", "img0"): {"dice": 0.5, "iou": 0.5}},
        )
        with self.assertRaises(FixedLodoReportError):
            paired_tests(substrate)

    def test_unknown_method_is_refused(self) -> None:
        base = {f"img{i}": 0.5 for i in range(30)}
        with self.assertRaises(FixedLodoReportError):
            paired_tests(_substrate(dict(base), {"single_a": base}), method="bootstrap")

    def test_family_size_is_every_source_and_structure(self) -> None:
        base = {f"img{i}": 0.5 + i * 0.001 for i in range(30)}
        singles = {f"single_{s}": dict(base) for s in SOURCES}
        results = paired_tests(_substrate({k: v + 0.02 for k, v in base.items()}, singles))
        self.assertEqual(len(results), len(SOURCES))
        self.assertEqual({r.arm_a for r in results}, set(singles))


def _write_fixed_run(base: Path, domain: Domain, seed: int, completed: str) -> Path:
    run = base / f"fixed_s3_{domain.value}_seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "image_id": f"img{i}",
            "structure": structure,
            "dice": "0.8",
            "iou": "0.66",
            "hd95": "10",
            "acc": "0.9",
            "tp": 10,
            "fp": 1,
            "fn": 1,
            "tn": 90,
        }
        for i in range(5)
        for structure in ("disc", "cup")
    ]
    with (run / "test_per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run / "test_metrics.json").write_text(
        json.dumps(
            {
                "test": {"evaluated_sample_count": 5},
                "fixed_lodo": {
                    "protocol": "leave_one_domain_out_fixed_budget",
                    "arm": "stage3_lodo_fixed_budget_plain_unet",
                    "held_out_domain": domain.value,
                    "source_domains": list(SOURCES),
                    "run_seed": seed,
                    "budget": {
                        "train": 40,
                        "val": 10,
                        "test": 50,
                        "subsample_seed": 42,
                    },
                    "manifest_sha256": MANIFEST_SHA,
                    "completed_at_utc": completed,
                    "smoke_rehearsal": False,
                    "scientific_result": True,
                },
            }
        )
    )
    return run


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        for seed in SEEDS:
            _write_fixed_run(
                self.directory, HELD_OUT, seed, f"2026-09-08T0{seed - 42}:00:00+00:00"
            )

    def test_discovers_and_selects(self) -> None:
        runs = select_fixed_runs(discover_fixed_runs([self.directory]), SEEDS)
        self.assertEqual(len(runs), len(SEEDS))

    def test_smoke_rehearsal_is_ignored(self) -> None:
        path = next(self.directory.rglob("test_metrics.json"))
        payload = json.loads(path.read_text())
        payload["fixed_lodo"]["smoke_rehearsal"] = True
        path.write_text(json.dumps(payload))
        self.assertIsNone(build_fixed_run(path))

    def test_non_scientific_run_is_refused_not_skipped(self) -> None:
        path = next(self.directory.rglob("test_metrics.json"))
        payload = json.loads(path.read_text())
        payload["fixed_lodo"]["scientific_result"] = False
        path.write_text(json.dumps(payload))
        with self.assertRaises(FixedLodoReportError):
            build_fixed_run(path)

    def test_incomplete_grid_is_refused(self) -> None:
        with self.assertRaises(FixedLodoReportError) as caught:
            select_fixed_runs(discover_fixed_runs([self.directory]), (42, 43, 44))
        self.assertIn("seed_44", str(caught.exception))

    def test_relaunched_run_keeps_the_later_completion(self) -> None:
        later = _write_fixed_run(
            self.directory / "relaunch", HELD_OUT, 42, "2026-09-30T00:00:00+00:00"
        )
        self.assertTrue(later.is_dir())
        runs = select_fixed_runs(discover_fixed_runs([self.directory]), SEEDS)
        self.assertEqual(len(runs), len(SEEDS))
        kept = next(r for r in runs if r.run_seed == 42)
        self.assertEqual(kept.completed_at_utc, "2026-09-30T00:00:00+00:00")

    def test_runs_from_two_manifests_are_refused(self) -> None:
        path = next(self.directory.rglob("test_metrics.json"))
        payload = json.loads(path.read_text())
        payload["fixed_lodo"]["manifest_sha256"] = "b" * 64
        path.write_text(json.dumps(payload))
        with self.assertRaises(FixedLodoReportError) as caught:
            select_fixed_runs(discover_fixed_runs([self.directory]), SEEDS)
        self.assertIn("manifest_sha256", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
