"""Training-curve rendering from history.csv, mid-run and after."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import plot_training_curves as cli  # noqa: E402
from spfilm.engine import _write_history  # noqa: E402
from spfilm.visualization import (  # noqa: E402
    best_epoch_from_history,
    early_stop_epoch_from_history,
    load_history,
    save_training_curves,
)


def _row(epoch: int, val_loss: float, patience: int, stopped: int = -1) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "train_loss": 1.0 / epoch,
        "val_loss": val_loss,
        "val_disc_dice": 0.5 + 0.01 * epoch,
        "val_cup_dice": 0.3 + 0.01 * epoch,
        "learning_rate": 1e-3 / epoch,
        "epoch_seconds": 4.0,
        "epochs_without_improvement": float(patience),
        "would_have_stopped_at_epoch": float(stopped),
    }


# val loss improves through epoch 3, then plateaus; the rule fires at epoch 5.
HISTORY = [
    _row(1, 0.9, 0),
    _row(2, 0.8, 0),
    _row(3, 0.7, 0),
    _row(4, 0.7, 1),
    _row(5, 0.71, 2, stopped=5),
    _row(6, 0.72, 3, stopped=5),
]
LEGACY_COLUMNS = ("epoch", "train_loss", "val_loss", "val_disc_dice", "val_cup_dice")


class HistoryRoundTripTests(unittest.TestCase):
    def test_load_history_reads_back_what_the_engine_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            _write_history(HISTORY, path)
            self.assertEqual(load_history(path), HISTORY)

    def test_an_empty_history_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text("epoch,train_loss\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no completed epochs"):
                load_history(path)

    def test_a_history_missing_a_curve_column_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text("epoch,train_loss\n1.0,0.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "val_loss"):
                load_history(path)


class EpochMarkerTests(unittest.TestCase):
    def test_best_epoch_is_the_last_one_that_reset_the_patience_counter(self) -> None:
        self.assertEqual(best_epoch_from_history(HISTORY), 3)

    def test_best_epoch_tracks_a_partial_history(self) -> None:
        self.assertEqual(best_epoch_from_history(HISTORY[:2]), 2)

    def test_early_stop_epoch_comes_from_the_latest_row(self) -> None:
        self.assertIsNone(early_stop_epoch_from_history(HISTORY[:4]))
        self.assertEqual(early_stop_epoch_from_history(HISTORY), 5)

    def test_legacy_histories_without_the_columns_give_no_markers(self) -> None:
        legacy = [{k: row[k] for k in LEGACY_COLUMNS} for row in HISTORY]
        self.assertIsNone(best_epoch_from_history(legacy))
        self.assertIsNone(early_stop_epoch_from_history(legacy))
        self.assertIsNone(best_epoch_from_history([]))


class RenderTests(unittest.TestCase):
    def test_renders_a_full_history_with_a_title(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = save_training_curves(HISTORY, Path(directory) / "sub" / "curves.png", title="t")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_renders_a_single_epoch_and_a_legacy_history(self) -> None:
        legacy = [{k: row[k] for k in LEGACY_COLUMNS} for row in HISTORY]
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(save_training_curves(HISTORY[:1], Path(directory) / "one.png").is_file())
            self.assertTrue(save_training_curves(legacy, Path(directory) / "legacy.png").is_file())

    def test_an_empty_history_cannot_be_drawn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_training_curves([], Path(directory) / "curves.png")


class CliTests(unittest.TestCase):
    def _run_dir(self, directory: str, finished: bool) -> Path:
        run = Path(directory) / "allf_s4_seed_42_1"
        run.mkdir()
        _write_history(HISTORY, run / "history.csv")
        if finished:
            (run / "test_metrics.json").write_text(json.dumps({}), encoding="utf-8")
        return run

    def _main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_an_in_progress_run_is_rendered_beside_its_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_dir(directory, finished=False)
            code, out, _ = self._main([str(run)])
            self.assertEqual(code, 0)
            self.assertTrue((run / "training_curves.png").is_file())
            self.assertIn("in progress", out)
            self.assertIn("best_epoch=3", out)
            self.assertIn("early_stop_epoch=5", out)

    def test_a_finished_run_is_labelled_and_can_go_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_dir(directory, finished=True)
            target = Path(directory) / "elsewhere.png"
            code, out, _ = self._main([str(run / "history.csv"), "--out", str(target), "--title", "x"])
            self.assertEqual(code, 0)
            self.assertTrue(target.is_file())
            self.assertFalse((run / "training_curves.png").exists())
            self.assertIn("finished", out)

    def test_out_is_refused_for_several_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_dir(directory, finished=False)
            code, _, err = self._main([str(run), str(run), "--out", "x.png"])
            self.assertEqual(code, 64)
            self.assertIn("single run", err)

    def test_a_missing_history_fails_that_run_but_renders_the_others(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self._run_dir(directory, finished=False)
            code, out, err = self._main([str(Path(directory) / "absent"), str(run)])
            self.assertEqual(code, 2)
            self.assertIn("absent", err)
            self.assertTrue((run / "training_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
