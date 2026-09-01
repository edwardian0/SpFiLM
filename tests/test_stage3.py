from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spfilm.data import FundusRecord  # noqa: E402
from spfilm.lodo import (  # noqa: E402
    Domain,
    DomainPartitions,
    LodoManifest,
    SampleKey,
)
from spfilm.stage3 import (  # noqa: E402
    Stage3ConfigError,
    Stage3DataError,
    Stage3LodoConfig,
    fold_record_splits,
    lodo_manifest_path,
    resolve_manifest_records,
    resolve_project_output,
    select_lodo_smoke_splits,
)


class Stage3ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "configs" / "stage3_lodo.json"
        self.config = Stage3LodoConfig.from_json(self.config_path)

    def test_local_and_create_configs_are_runnable_protocol_configs(self) -> None:
        create_config = Stage3LodoConfig.from_json(
            PROJECT_ROOT / "configs" / "stage3_lodo_create.json"
        )

        for config in (self.config, create_config):
            with self.subTest(config=config):
                self.assertEqual(set(config.held_out_domains), set(Domain))
                self.assertEqual(config.run_seeds, (42, 43, 44, 45, 46))
                self.assertEqual(
                    config.domain_config(Domain.REFUGE_CANON_VAL).adapter,
                    "refuge_canon_val",
                )
                json.dumps(asdict(config))

    def test_manifest_path_is_inside_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = lodo_manifest_path(self.config, root)

        self.assertEqual(
            path,
            root.resolve() / "splits" / "lodo" / "lodo_manifest.json",
        )

    def test_generated_output_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                Stage3ConfigError,
                r"Generated output must remain inside",
            ):
                resolve_project_output(directory, "../outside")

    def test_training_config_keeps_run_seed_separate_from_split_seed(self) -> None:
        config = self.config.training_config(
            Domain.RIM_ONE_DL,
            46,
            "artifacts/stage3_lodo/rim_one_dl/seed_46",
            requested_device="cpu",
        )

        self.assertEqual(config.seed, 46)
        self.assertEqual(config.dataset, "rim_one_dl")
        self.assertEqual(config.rim_manifest, "splits/rim_one_dl.json")
        self.assertEqual(config.requested_device, "cpu")

    def test_stale_blocked_marker_fails_closed(self) -> None:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        payload["domains"]["refuge_canon_val"]["blocked_on"] = "not implemented"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blocked.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Stage3ConfigError, r"remains blocked"):
                Stage3LodoConfig.from_json(path)


class Stage3ResolutionTests(unittest.TestCase):
    @staticmethod
    def _record(domain: Domain, sample_id: str) -> FundusRecord:
        return FundusRecord(
            sample_id=sample_id,
            domain=domain.value,
            image_path=Path(f"/{domain.value}/{sample_id}.png"),
            combined_mask_path=Path(f"/{domain.value}/{sample_id}_mask.png"),
            mask_encoding="unused_in_membership_test",
        )

    def setUp(self) -> None:
        self.partitions = tuple(
            DomainPartitions(
                domain=domain,
                train=(SampleKey(domain, "shared-train"),),
                val=(SampleKey(domain, "shared-val"),),
                test=(SampleKey(domain, "shared-test"),),
            )
            for domain in Domain
        )
        self.manifest = LodoManifest.build(
            self.partitions,
            {domain: 42 for domain in Domain},
        )
        self.records_by_domain = {
            domain: [
                self._record(domain, "shared-train"),
                self._record(domain, "shared-val"),
                self._record(domain, "shared-test"),
            ]
            for domain in Domain
        }

    def test_resolution_uses_domain_and_sample_id_together(self) -> None:
        resolved = resolve_manifest_records(
            self.manifest,
            self.records_by_domain,
        )

        self.assertEqual(len(resolved), 12)
        self.assertEqual(
            resolved[SampleKey(Domain.REFUGE_ZEISS, "shared-train")].domain,
            Domain.REFUGE_ZEISS.value,
        )
        self.assertEqual(
            resolved[SampleKey(Domain.RIM_ONE_DL, "shared-train")].domain,
            Domain.RIM_ONE_DL.value,
        )

    def test_fold_resolution_preserves_locked_lodo_roles(self) -> None:
        resolved = resolve_manifest_records(
            self.manifest,
            self.records_by_domain,
        )
        fold = next(
            item
            for item in self.manifest.folds
            if item.held_out_domain == Domain.DRISHTI_GS
        )

        splits = fold_record_splits(fold, resolved)

        self.assertEqual(len(splits["train"]), 3)
        self.assertEqual(len(splits["val"]), 3)
        self.assertEqual(len(splits["test"]), 1)
        self.assertNotIn(Domain.DRISHTI_GS.value, {r.domain for r in splits["train"]})
        self.assertEqual(
            {record.domain for record in splits["test"]},
            {Domain.DRISHTI_GS.value},
        )

    def test_smoke_selection_keeps_every_represented_domain(self) -> None:
        resolved = resolve_manifest_records(
            self.manifest,
            self.records_by_domain,
        )
        fold = next(
            item
            for item in self.manifest.folds
            if item.held_out_domain == Domain.RIM_ONE_DL
        )
        full_splits = fold_record_splits(fold, resolved)

        smoke_splits = select_lodo_smoke_splits(full_splits)

        self.assertEqual(len(smoke_splits["train"]), 3)
        self.assertEqual(len(smoke_splits["val"]), 3)
        self.assertEqual(len(smoke_splits["test"]), 1)

    def test_resolution_rejects_unlisted_discovery(self) -> None:
        records = dict(self.records_by_domain)
        records[Domain.RIM_ONE_DL] = [
            *records[Domain.RIM_ONE_DL],
            self._record(Domain.RIM_ONE_DL, "unexpected"),
        ]

        with self.assertRaisesRegex(Stage3DataError, r"manifest/discovery mismatch"):
            resolve_manifest_records(self.manifest, records)


if __name__ == "__main__":
    unittest.main()
