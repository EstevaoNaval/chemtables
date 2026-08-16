"""Unit tests for measurement_parse and extract_table_measurements."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from chemtables.measurements.extract import extract_table_measurements
from chemtables.measurements.parse import UNPARSEABLE, ParsedCell, parse_measurement_cell
from chemtables.schema.interpret import validate_schema
from test_bio_entity_match import cell_line_catalog, memory_catalog


class TestMeasurementParse(unittest.TestCase):
    def test_plain_number(self):
        p = parse_measurement_cell("4.4")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.status, "reported")
        self.assertEqual(p.relation, "=")
        self.assertEqual(p.value, 4.4)
        self.assertIsNone(p.uncertainty)
        self.assertIsNone(p.text_value)

    def test_plus_minus(self):
        p = parse_measurement_cell("4.4 ± 0.9")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.relation, "=")
        self.assertEqual(p.value, 4.4)
        self.assertEqual(p.uncertainty, 0.9)

    def test_inequality_gt(self):
        p = parse_measurement_cell(">100")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.relation, ">")
        self.assertEqual(p.value, 100.0)
        self.assertIsNone(p.text_value)

    def test_inequality_gte(self):
        p = parse_measurement_cell(">= 64")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.relation, ">=")
        self.assertEqual(p.value, 64.0)

    def test_approx(self):
        p = parse_measurement_cell("~5")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.relation, "~")
        self.assertEqual(p.value, 5.0)

    def test_range_en_dash(self):
        p = parse_measurement_cell("3–8")
        assert isinstance(p, ParsedCell)
        self.assertIsNone(p.value)
        self.assertEqual(p.text_value, "3–8")
        self.assertEqual(p.status, "reported")

    def test_range_to(self):
        p = parse_measurement_cell("3 to 8")
        assert isinstance(p, ParsedCell)
        self.assertIsNone(p.value)
        self.assertEqual(p.text_value, "3 to 8")

    def test_nd_text_value(self):
        p = parse_measurement_cell("n.d.")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.status, "not_detected")
        self.assertIsNone(p.value)
        self.assertEqual(p.text_value, "n.d.")

    def test_inactive(self):
        p = parse_measurement_cell("inactive")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.text_value, "inactive")

    def test_hyphen_not_reported(self):
        p = parse_measurement_cell("—")
        assert isinstance(p, ParsedCell)
        self.assertEqual(p.status, "not_reported")
        self.assertIsNone(p.text_value)

    def test_multi_unparseable(self):
        self.assertIs(parse_measurement_cell("1.2; 3.4"), UNPARSEABLE)


class TestExtractTableMeasurements(unittest.TestCase):
    def test_compounds_in_columns_golden(self):
        schema_payload = {
            "title": "Table 2. IC50 values of compounds 12g and 12h for breast cancer cells.",
            "footnotes": {},
            "schema": {
                "layout": "compounds_in_columns",
                "assay": {
                    "description": "IC50 values of compounds for breast cancer cells",
                    "type": "T",
                },
                "columns": [
                    {
                        "column_index": 0,
                        "role": "identifier",
                        "property_name": None,
                        "context": None,
                        "unit": None,
                        "footnote_refs": [],
                        "header_path": ["Cell line"],
                        "target_name": None,
                        "target_type": None,
                    },
                    {
                        "column_index": 1,
                        "role": "property",
                        "property_name": "IC50",
                        "context": None,
                        "unit": "μM",
                        "footnote_refs": [],
                        "header_path": ["IC50 12g (μM)"],
                        "target_name": None,
                        "target_type": None,
                    },
                    {
                        "column_index": 2,
                        "role": "property",
                        "property_name": "IC50",
                        "context": None,
                        "unit": "μM",
                        "footnote_refs": [],
                        "header_path": ["IC50 12h (μM)"],
                        "target_name": None,
                        "target_type": None,
                    },
                ],
            },
        }
        rows = [
            ["Cell line", "IC50 12g (μM)", "IC50 12h (μM)"],
            ["MCF10A", "n.d.", "n.d."],
            ["MCF-7", "4.4 ± 0.9", "4.3 ± 1.1"],
            ["MDA-MB-231", "11.1 ± 2.3", "9.2 ± 2.1"],
        ]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "extracted_table.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)

            result = extract_table_measurements(
                csv_path,
                schema_payload,
                ["12g", "12h"],
                "test_03_like",
                catalog=cell_line_catalog(),
            )

        self.assertIn("_context_warnings", result)
        self.assertIn("records", result)
        self.assertNotIn("ambiguous_cells", result)
        self.assertNotIn("measurements", result)
        self.assertEqual(result["layout"], "compounds_in_columns")
        self.assertEqual(result["unmapped_columns"], [])
        self.assertEqual(result["unmapped_rows"], [])

        records = result["records"]
        self.assertEqual(len(records), 6)

        for rec in records:
            self.assertNotIn("provenance", rec)
            self.assertEqual(rec["compound"]["reference_id"], None)
            self.assertIn(rec["compound"]["name"], {"12g", "12h"})
            self.assertEqual(rec["activity"]["type"], "IC50")
            self.assertEqual(rec["assay"]["type"], "T")

        mcf7 = [
            r
            for r in records
            if r["target"]["name"] == "MCF-7" and r["compound"]["name"] == "12g"
        ]
        self.assertEqual(len(mcf7), 1)
        self.assertEqual(mcf7[0]["activity"]["value"], 4.4)
        self.assertEqual(mcf7[0]["activity"]["relation"], "=")
        self.assertEqual(mcf7[0]["activity"]["comment"], "± 0.9")
        self.assertEqual(mcf7[0]["target"]["type"], "cell_line")

        nd = [
            r
            for r in records
            if r["target"]["name"] == "MCF10A" and r["compound"]["name"] == "12g"
        ]
        self.assertEqual(len(nd), 1)
        self.assertIsNone(nd[0]["activity"]["value"])
        self.assertEqual(nd[0]["activity"]["text_value"], "n.d.")

    def test_skips_unmapped_and_hyphen(self):
        schema_payload = {
            "title": "Table 1. MIC (μg/mL)",
            "schema": {
                "layout": "compounds_in_rows",
                "assay": {"description": "MIC antibacterial assay", "type": "F"},
                "columns": [
                    {
                        "column_index": 0,
                        "role": "identifier",
                        "property_name": None,
                        "context": None,
                        "unit": None,
                        "footnote_refs": [],
                        "header_path": ["Compound"],
                        "target_name": None,
                        "target_type": None,
                    },
                    {
                        "column_index": 1,
                        "role": "property",
                        "property_name": "MIC",
                        "context": "S. aureus",
                        "unit": "μg/mL",
                        "footnote_refs": [],
                        "header_path": ["S. aureus"],
                        "target_name": "S. aureus",
                        "target_type": "organism",
                    },
                ],
            },
        }
        rows = [
            ["Compound", "S. aureus"],
            ["9a", ">64"],
            ["9b", "—"],
            ["Ribavirin", "8"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "t.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            result = extract_table_measurements(
                csv_path, schema_payload, ["9a", "9b"], "mic_test"
            )

        self.assertEqual(len(result["records"]), 1)
        rec = result["records"][0]
        self.assertEqual(rec["compound"]["name"], "9a")
        self.assertEqual(rec["activity"]["relation"], ">")
        self.assertEqual(rec["activity"]["value"], 64.0)
        self.assertEqual(rec["target"]["name"], "S. aureus")
        self.assertEqual(rec["target"]["type"], "organism")
        # 9b hyphen skipped; Ribavirin unmapped
        self.assertEqual(len(result["unmapped_rows"]), 1)
        self.assertEqual(result["unmapped_rows"][0]["cleaned"], "Ribavirin")

    def test_header_path_recovers_prefixed_protein(self):
        catalog = memory_catalog(
            [
                (1, "protein", "SYMBOL", "P00001", "Homo sapiens", 9606),
            ],
            [("SYMBOL", 1)],
        )
        self.addCleanup(catalog.close)
        schema_payload = {
            "title": "HOST_PHRASE assay",
            "schema": {
                "layout": "compounds_in_rows",
                "assay": {"description": "IC50 assay", "type": "F"},
                "columns": [
                    {
                        "column_index": 0,
                        "role": "identifier",
                        "property_name": None,
                        "context": None,
                        "unit": None,
                        "footnote_refs": [],
                        "header_path": ["Compound"],
                        "target_name": None,
                        "target_type": None,
                    },
                    {
                        "column_index": 1,
                        "role": "property",
                        "property_name": "IC50",
                        "context": "READOUT_A host cells",
                        "unit": "μM",
                        "footnote_refs": [],
                        "header_path": ["IC50 mSYMBOL (μM) READOUT_A"],
                        "target_name": "host cells",
                        "target_type": "cell_line",
                    },
                ],
            },
        }
        rows = [
            ["Compound", "IC50 mSYMBOL"],
            ["9a", "1.5 ± 0.03"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "t.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            result = extract_table_measurements(
                csv_path,
                schema_payload,
                ["9a"],
                "header_recover",
                catalog=catalog,
            )

        self.assertEqual(len(result["records"]), 1)
        rec = result["records"][0]
        self.assertEqual(rec["target"]["name"], "mSYMBOL")
        self.assertEqual(rec["target"]["type"], "single_protein")

    def test_chemical_2_like_targets(self):
        catalog = cell_line_catalog()
        self.addCleanup(catalog.close)
        schema_payload = {
            "title": "Table 2",
            "schema": {
                "layout": "compounds_in_rows",
                "assay": {"description": "Table 2", "type": None},
                "columns": [
                    {
                        "column_index": 0,
                        "role": "identifier",
                        "property_name": None,
                        "context": None,
                        "unit": None,
                        "footnote_refs": [],
                        "header_path": ["Compounds"],
                        "target_name": None,
                        "target_type": None,
                    },
                    {
                        "column_index": 1,
                        "role": "property",
                        "property_name": "IC50",
                        "context": "in IEC-6 cells",
                        "unit": "μM",
                        "footnote_refs": [],
                        "header_path": ["['IC _50 in IEC-6 cells ( μ M)']"],
                        "target_name": "IEC-6 cells",
                        "target_type": "cell_line",
                    },
                    {
                        "column_index": 2,
                        "role": "property",
                        "property_name": "IC50",
                        "context": "in T. gondii",
                        "unit": "μM",
                        "footnote_refs": [],
                        "header_path": ["['IC _50 in T. gondii ( μ M)']"],
                        "target_name": "T. gondii",
                        "target_type": "organism",
                    },
                    {
                        "column_index": 3,
                        "role": "property",
                        "property_name": "SI",
                        "context": None,
                        "unit": "unitless",
                        "footnote_refs": [],
                        "header_path": ["SI"],
                        "target_name": None,
                        "target_type": None,
                    },
                ],
            },
        }
        rows = [
            ["Compounds", "IC50 IEC-6", "IC50 T. gondii", "SI"],
            ["16", "416.7", "342.5", "1.2"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "t.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            result = extract_table_measurements(
                csv_path,
                schema_payload,
                ["16"],
                "table_chemical_2_like",
                catalog=catalog,
            )

        records = result["records"]
        self.assertEqual(len(records), 3)
        iec = [r for r in records if r["activity"]["value"] == 416.7]
        gondii = [r for r in records if r["activity"]["value"] == 342.5]
        si = [r for r in records if r["activity"]["type"] == "SI"]
        self.assertEqual(iec[0]["target"]["name"], "IEC-6")
        self.assertEqual(iec[0]["target"]["type"], "cell_line")
        self.assertEqual(gondii[0]["target"]["name"], "T. gondii")
        self.assertEqual(gondii[0]["target"]["type"], "organism")
        self.assertIsNone(si[0]["target"]["name"])
        self.assertIsNone(si[0]["target"]["type"])

    def test_unmatched_cell_line_is_null(self):
        catalog = cell_line_catalog()
        self.addCleanup(catalog.close)
        schema_payload = {
            "title": "Table",
            "schema": {
                "layout": "compounds_in_rows",
                "assay": {"description": None, "type": None},
                "columns": [
                    {
                        "column_index": 0,
                        "role": "identifier",
                        "property_name": None,
                        "context": None,
                        "unit": None,
                        "footnote_refs": [],
                        "header_path": ["Compound"],
                        "target_name": None,
                        "target_type": None,
                    },
                    {
                        "column_index": 1,
                        "role": "property",
                        "property_name": "IC50",
                        "context": "UnknownLine cells",
                        "unit": "μM",
                        "footnote_refs": [],
                        "header_path": ["IC50 UnknownLine cells (μM)"],
                        "target_name": "UnknownLine cells",
                        "target_type": "cell_line",
                    },
                ],
            },
        }
        rows = [
            ["Compound", "IC50"],
            ["9a", "1.5"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "t.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            result = extract_table_measurements(
                csv_path,
                schema_payload,
                ["9a"],
                "unmatched",
                catalog=catalog,
            )

        self.assertEqual(len(result["records"]), 1)
        self.assertIsNone(result["records"][0]["target"]["name"])
        self.assertIsNone(result["records"][0]["target"]["type"])


class TestFlattenHeaderPath(unittest.TestCase):
    def test_nested_header_path_unwrapped(self):
        parsed = {
            "layout": "compounds_in_rows",
            "assay": {"description": None, "type": None},
            "columns": [
                {
                    "column_index": 0,
                    "role": "identifier",
                    "property_name": None,
                    "context": None,
                    "unit": None,
                    "footnote_refs": [],
                    "header_path": ["Compounds"],
                    "target_name": None,
                    "target_type": None,
                },
                {
                    "column_index": 1,
                    "role": "property",
                    "property_name": "IC50",
                    "context": "in T. gondii",
                    "unit": "μM",
                    "footnote_refs": [],
                    "header_path": [["IC _50 in T. gondii ( μ M)"]],
                    "target_name": "T. gondii",
                    "target_type": "organism",
                },
            ],
        }
        schema = validate_schema(parsed, 2)
        self.assertIsNotNone(schema)
        self.assertEqual(
            schema["columns"][1]["header_path"],
            ["IC _50 in T. gondii ( μ M)"],
        )
        self.assertEqual(schema["columns"][1]["target_name"], "T. gondii")


if __name__ == "__main__":
    unittest.main()
