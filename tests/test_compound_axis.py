"""Unit tests for deterministic compound-axis location."""

from __future__ import annotations

import unittest

from chemtables.matching.compound import (
    CompoundAxis,
    col0_looks_id_like,
    locate_compound_axis,
)
from chemtables.schema.interpret import (
    MissingCompoundRefsError,
    apply_compound_axis,
    interpret_table_schema,
)


class TestLocateCompoundAxis(unittest.TestCase):
    def test_body_axis_fpryz_like(self):
        header = [
            ["Azole/Quinones", "Azole/Quinones", "S. bra", "WT 1"],
        ]
        body = [
            ["Itraconazole", "MIC", "2", "1"],
            ["Q1", "MIC", ">128", "32"],
            ["Q1", "MFC", ">128", "32"],
            ["Q2", "MIC", ">128", "64"],
            ["Q10", "MIC", "128", "128"],
        ]
        refs = [f"Q{i}" for i in range(1, 12)]
        axis = locate_compound_axis(header, body, refs)
        self.assertEqual(axis.layout, "compounds_in_rows")
        self.assertEqual(axis.compound_column, 0)
        self.assertFalse(axis.skip_extract)
        self.assertGreater(axis.body_hits[0], 0)

    def test_header_axis_compounds_in_columns(self):
        header = [
            ["Cell line", "IC50 12g (uM)", "IC50 12h (uM)"],
        ]
        body = [
            ["MCF10A", "n.d.", "n.d."],
            ["MCF-7", "4.4", "4.3"],
        ]
        axis = locate_compound_axis(header, body, ["12g", "12h"])
        self.assertEqual(axis.layout, "compounds_in_columns")
        self.assertIn(axis.compound_column, {1, 2})
        self.assertFalse(axis.skip_extract)
        self.assertGreater(axis.header_hits[axis.compound_column], 0)

    def test_no_hits_col0_id_like(self):
        header = [["Entry", "MIC"]]
        body = [["8a", "1.2"], ["8b", "3.4"], ["8c", "0.5"]]
        axis = locate_compound_axis(header, body, ["Z99", "Z98"])
        self.assertTrue(col0_looks_id_like(body))
        self.assertEqual(axis.layout, "compounds_in_rows")
        self.assertEqual(axis.compound_column, 0)
        self.assertFalse(axis.skip_extract)
        self.assertTrue(any("no compound coreference hits" in w for w in axis.warnings))

    def test_no_hits_skip_extract(self):
        header = [["Assay readout description", "Value"]]
        body = [
            ["cytotoxicity mean across three independent replicates", "12.5"],
            ["viability percentage after seventy-two hours incubation", "11.0"],
            ["selectivity index derived from paired CC50 and EC50", "13.0"],
        ]
        axis = locate_compound_axis(header, body, ["12g", "12h"])
        self.assertFalse(col0_looks_id_like(body))
        self.assertIsNone(axis.layout)
        self.assertIsNone(axis.compound_column)
        self.assertTrue(axis.skip_extract)


class TestApplyCompoundAxis(unittest.TestCase):
    def _base_schema(self, n_cols: int = 4) -> dict:
        columns = []
        for i in range(n_cols):
            columns.append(
                {
                    "column_index": i,
                    "role": "property" if i else "identifier",
                    "property_name": "MIC" if i else None,
                    "context": None,
                    "unit": "μg/mL" if i else None,
                    "footnote_refs": [],
                    "header_path": [f"col{i}"],
                    "target_name": None,
                    "target_type": None,
                }
            )
        return {
            "layout": "compounds_in_columns",  # wrong on purpose
            "assay": {"description": None, "type": None},
            "columns": columns,
        }

    def test_override_wrong_layout_to_rows(self):
        schema = self._base_schema(4)
        # Fake Gemma labeled isolate cols as identifier.
        for entry in schema["columns"]:
            if entry["column_index"] >= 2:
                entry["role"] = "identifier"
        axis = CompoundAxis(
            layout="compounds_in_rows",
            compound_column=0,
            skip_extract=False,
        )
        apply_compound_axis(schema, axis)
        self.assertEqual(schema["layout"], "compounds_in_rows")
        self.assertFalse(schema["skip_extract"])
        by_idx = {c["column_index"]: c for c in schema["columns"]}
        self.assertEqual(by_idx[0]["role"], "identifier")
        self.assertEqual(by_idx[2]["role"], "property")
        self.assertEqual(by_idx[3]["role"], "property")

    def test_skip_extract_clears_roles(self):
        schema = self._base_schema(3)
        axis = CompoundAxis(
            layout=None,
            compound_column=None,
            skip_extract=True,
            warnings=["no axis"],
        )
        apply_compound_axis(schema, axis)
        self.assertIsNone(schema["layout"])
        self.assertTrue(schema["skip_extract"])
        self.assertTrue(all(c["role"] == "other" for c in schema["columns"]))


class TestMissingCompoundRefs(unittest.TestCase):
    def test_interpret_requires_refs(self):
        with self.assertRaises(MissingCompoundRefsError):
            interpret_table_schema(
                "title",
                [["A", "B"]],
                compound_refs=None,
                generate_fn=lambda *a, **k: "{}",
            )

    def test_interpret_rejects_empty_refs(self):
        with self.assertRaises(MissingCompoundRefsError):
            interpret_table_schema(
                "title",
                [["A", "B"]],
                compound_refs=[],
                generate_fn=lambda *a, **k: "{}",
            )


class TestExtractSkip(unittest.TestCase):
    def test_extract_honors_skip_extract(self):
        import csv
        import tempfile
        from pathlib import Path

        from chemtables.measurements.extract import extract_table_measurements

        schema_payload = {
            "title": "Table X",
            "compound_axis": {
                "warnings": ["no reliable compound axis"],
                "skip_extract": True,
            },
            "schema": {
                "layout": None,
                "skip_extract": True,
                "assay": {"description": None, "type": None},
                "columns": [
                    {
                        "column_index": 0,
                        "role": "other",
                        "property_name": None,
                        "context": None,
                        "unit": None,
                        "footnote_refs": [],
                        "header_path": ["A"],
                        "target_name": None,
                        "target_type": None,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "t.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows([["A"], ["1"]])
            result = extract_table_measurements(
                csv_path, schema_payload, ["12g"], "t"
            )
        self.assertEqual(result["records"], [])
        self.assertIsNone(result["layout"])
        self.assertTrue(
            any("skip_extract" in w for w in result["_context_warnings"])
        )


if __name__ == "__main__":
    unittest.main()
