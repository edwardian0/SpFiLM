from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_domain_shift import (  # noqa: E402
    GLAUCOMA,
    _diagnosis_caps,
    NON_GLAUCOMA,
    _normalise_diagnosis,
    _read_xlsx_rows,
    diagnosis_of,
    group_by_diagnosis,
)
from spfilm.data import FundusRecord  # noqa: E402


def _record(sample_id: str, **kwargs) -> FundusRecord:
    return FundusRecord(
        sample_id=sample_id,
        domain=kwargs.pop("domain", "refuge_zeiss"),
        image_path=Path(kwargs.pop("image_path", f"/tmp/{sample_id}.jpg")),
        mask_encoding=kwargs.pop("mask_encoding", "combined"),
        **kwargs,
    )


SHEET = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>1</v></c></row>
<row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>0</v></c></row>
</sheetData></worksheet>"""

STRINGS = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="4">
<si><t>ImgName</t></si><si><t>Glaucoma Label</t></si>
<si><t>V0001.jpg</t></si><si><t>V0002.jpg</t></si>
</sst>"""


class NormaliseTests(unittest.TestCase):
    def test_the_three_vocabularies_collapse_to_two_classes(self) -> None:
        for value in ("glaucoma", "Glaucoma", " GLAUCOMATOUS ", "1"):
            self.assertEqual(_normalise_diagnosis(value), GLAUCOMA, value)
        for value in ("normal", "non_glaucoma", "Non-Glaucoma", "healthy", "0"):
            self.assertEqual(_normalise_diagnosis(value), NON_GLAUCOMA, value)

    def test_unrecognised_values_are_not_guessed(self) -> None:
        # RIM-ONE-DL's release-prefixed strata and REFUGE's single-bucket
        # validation stratum must fall through to a real label source, not be
        # pattern-matched into a class.
        for value in (None, "", "refuge_validation400", "r1_normal", "unknown"):
            self.assertIsNone(_normalise_diagnosis(value))


class DiagnosisOfTests(unittest.TestCase):
    def test_diagnosis_class_wins_over_stratum(self) -> None:
        record = _record("a", diagnosis_class="glaucoma", stratum="r1_normal")
        self.assertEqual(diagnosis_of(record, {}), GLAUCOMA)

    def test_stratum_is_used_when_there_is_no_diagnosis_class(self) -> None:
        self.assertEqual(
            diagnosis_of(_record("a", stratum="non_glaucoma"), {}), NON_GLAUCOMA
        )
        self.assertEqual(diagnosis_of(_record("b", stratum="normal"), {}), NON_GLAUCOMA)

    def test_spreadsheet_labels_cover_the_domain_with_no_label_on_the_record(
        self,
    ) -> None:
        record = _record(
            "V0001", stratum="refuge_validation400", image_path="/data/V0001.jpg"
        )
        self.assertIsNone(diagnosis_of(record, {}))
        self.assertEqual(diagnosis_of(record, {"V0001": GLAUCOMA}), GLAUCOMA)

    def test_unlabelled_records_are_reported_not_dropped_silently(self) -> None:
        records = [
            _record("a", stratum="glaucoma"),
            _record("b", stratum="normal"),
            _record("c", stratum="refuge_validation400"),
        ]
        grouped, unlabelled = group_by_diagnosis(records, {})
        self.assertEqual(len(grouped[GLAUCOMA]), 1)
        self.assertEqual(len(grouped[NON_GLAUCOMA]), 1)
        self.assertEqual(unlabelled, 1)


class XlsxTests(unittest.TestCase):
    def test_reads_shared_strings_and_inline_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "labels.xlsx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("xl/sharedStrings.xml", STRINGS)
                archive.writestr("xl/worksheets/sheet1.xml", SHEET)
            rows = _read_xlsx_rows(path)

        self.assertEqual(rows[0], ["ImgName", "Glaucoma Label"])
        self.assertEqual(rows[1], ["V0001.jpg", "1"])
        self.assertEqual(rows[2], ["V0002.jpg", "0"])
        # The label column is what the split keys on, so pin the round trip.
        self.assertEqual(_normalise_diagnosis(rows[1][1]), GLAUCOMA)
        self.assertEqual(_normalise_diagnosis(rows[2][1]), NON_GLAUCOMA)


class BalanceTests(unittest.TestCase):
    # The real shape of the data: REFUGE is 40/360 twice, Drishti 70/31,
    # RIM-ONE-DL 172/313.
    GROUPED = {
        "drishti_gs": {GLAUCOMA: [0] * 70, NON_GLAUCOMA: [0] * 31},
        "refuge_canon_val": {GLAUCOMA: [0] * 40, NON_GLAUCOMA: [0] * 360},
        "refuge_zeiss": {GLAUCOMA: [0] * 40, NON_GLAUCOMA: [0] * 360},
        "rim_one_dl": {GLAUCOMA: [0] * 172, NON_GLAUCOMA: [0] * 313},
    }

    def test_off_passes_the_sample_through_untouched(self) -> None:
        self.assertEqual(
            _diagnosis_caps(self.GROUPED, "off", 100),
            {GLAUCOMA: 100, NON_GLAUCOMA: 100},
        )
        self.assertEqual(
            _diagnosis_caps(self.GROUPED, "off", None),
            {GLAUCOMA: None, NON_GLAUCOMA: None},
        )

    def test_per_class_balances_each_figure_separately(self) -> None:
        self.assertEqual(
            _diagnosis_caps(self.GROUPED, "per-class", None),
            {GLAUCOMA: 40, NON_GLAUCOMA: 31},
        )

    def test_global_uses_one_cap_across_both_classes(self) -> None:
        self.assertEqual(
            _diagnosis_caps(self.GROUPED, "global", None),
            {GLAUCOMA: 31, NON_GLAUCOMA: 31},
        )

    def test_an_explicit_sample_can_only_lower_the_cap(self) -> None:
        # --sample must not be able to inflate a subset past what exists.
        self.assertEqual(
            _diagnosis_caps(self.GROUPED, "per-class", 20),
            {GLAUCOMA: 20, NON_GLAUCOMA: 20},
        )
        self.assertEqual(
            _diagnosis_caps(self.GROUPED, "per-class", 500),
            {GLAUCOMA: 40, NON_GLAUCOMA: 31},
        )

    def test_a_class_absent_from_every_domain_caps_to_nothing(self) -> None:
        empty = {"a": {GLAUCOMA: [0] * 5, NON_GLAUCOMA: []}}
        self.assertEqual(
            _diagnosis_caps(empty, "per-class", None),
            {GLAUCOMA: 5, NON_GLAUCOMA: None},
        )

    def test_domains_missing_a_class_do_not_drag_the_cap_to_zero(self) -> None:
        # An absent subset means "not in that figure", not "cap everyone at 0".
        grouped = dict(self.GROUPED)
        grouped["no_glaucoma_domain"] = {GLAUCOMA: [], NON_GLAUCOMA: [0] * 90}
        self.assertEqual(
            _diagnosis_caps(grouped, "per-class", None),
            {GLAUCOMA: 40, NON_GLAUCOMA: 31},
        )


if __name__ == "__main__":
    unittest.main()
