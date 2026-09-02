from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from spfilm.engine import (  # noqa: E402
    RESUME_STATE_FILENAME,
    Stage2Config,
    _load_resume_state,
    _resume_fingerprint,
    _save_resume_state,
)


def make_config(**overrides) -> Stage2Config:
    values = {
        "experiment_name": "resume_test",
        "dataset": "refuge",
        "data_root": "datasets/REFUGE",
        "output_dir": "artifacts/resume_test",
        "seed": 42,
        "epochs": 10,
    }
    values.update(overrides)
    return Stage2Config(**values)


class ResumeFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = {"train": 636, "val": 122, "test": 80}

    def test_same_config_and_splits_give_the_same_fingerprint(self) -> None:
        self.assertEqual(
            _resume_fingerprint(make_config(), self.counts),
            _resume_fingerprint(make_config(), self.counts),
        )

    def test_a_changed_config_changes_the_fingerprint(self) -> None:
        self.assertNotEqual(
            _resume_fingerprint(make_config(), self.counts),
            _resume_fingerprint(make_config(seed=43), self.counts),
        )

    def test_changed_split_counts_change_the_fingerprint(self) -> None:
        self.assertNotEqual(
            _resume_fingerprint(make_config(), self.counts),
            _resume_fingerprint(
                make_config(), {"train": 852, "val": 176, "test": 51}
            ),
        )


class ResumeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.path = self.directory / RESUME_STATE_FILENAME
        self.fingerprint = _resume_fingerprint(
            make_config(), {"train": 1, "val": 1, "test": 1}
        )

    def write(self, **overrides) -> None:
        state = {
            "fingerprint": self.fingerprint,
            "epoch": 7,
            "history": [{"epoch": 1.0}],
            "best_val_loss": 0.5,
        }
        state.update(overrides)
        _save_resume_state(self.path, **state)

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertIsNone(
            _load_resume_state(self.directory / "absent.pt", self.fingerprint)
        )

    def test_round_trip_preserves_progress_and_rng(self) -> None:
        self.write()
        state = _load_resume_state(self.path, self.fingerprint)

        self.assertEqual(state["epoch"], 7)
        self.assertEqual(state["history"], [{"epoch": 1.0}])
        self.assertEqual(state["schema_version"], 1)
        for key in ("python", "numpy", "torch"):
            self.assertIn(key, state["rng"])

    def test_no_temporary_file_is_left_behind(self) -> None:
        self.write()

        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            [RESUME_STATE_FILENAME],
        )

    def test_a_fingerprint_mismatch_refuses_to_resume(self) -> None:
        self.write(fingerprint="written-for-a-different-run")

        with self.assertRaisesRegex(
            RuntimeError,
            r"was written for a different config or split",
        ):
            _load_resume_state(self.path, self.fingerprint)

    def test_an_unknown_schema_version_refuses_to_resume(self) -> None:
        self.write()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        payload["schema_version"] = 2
        torch.save(payload, self.path)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Unsupported resume-state schema",
        ):
            _load_resume_state(self.path, self.fingerprint)


class RequireFreshOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        from run_stage3_lodo import _require_fresh_output

        self._require_fresh_output = _require_fresh_output
        self.directory = Path(tempfile.mkdtemp()) / "run"
        self.directory.mkdir()

    def test_an_empty_directory_is_accepted(self) -> None:
        self.assertEqual(
            self._require_fresh_output(self.directory, smoke=False), self.directory
        )

    def test_a_populated_directory_without_resume_state_is_refused(self) -> None:
        (self.directory / "test_metrics.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(
            Exception,
            r"Refusing to overwrite non-empty run directory",
        ):
            self._require_fresh_output(self.directory, smoke=False)

    def test_a_preempted_directory_with_resume_state_is_reentered(self) -> None:
        (self.directory / "history.csv").write_text("epoch\n", encoding="utf-8")
        (self.directory / RESUME_STATE_FILENAME).write_bytes(b"placeholder")

        self.assertEqual(
            self._require_fresh_output(self.directory, smoke=False), self.directory
        )


if __name__ == "__main__":
    unittest.main()
