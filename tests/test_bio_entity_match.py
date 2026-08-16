"""Unit tests for bio_entity_match target typing."""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from chemtables.matching.bio_entity import (
    BioEntityCatalog,
    classify_target_type,
    extract_organism_label,
    is_organism_label,
    open_catalog,
)
from chemtables.paths import BIO_ENTITIES_SCHEMA_PATH


def memory_catalog(
    entities: list[tuple],
    aliases: list[tuple[str, int]],
) -> BioEntityCatalog:
    conn = sqlite3.connect(":memory:")
    conn.executescript(BIO_ENTITIES_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.executemany(
        "INSERT INTO entity "
        "(id, type, preferred_name, accession, organism, taxon_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        entities,
    )
    conn.executemany("INSERT INTO alias (name, entity_id) VALUES (?, ?)", aliases)
    conn.commit()
    return BioEntityCatalog(conn, owns_conn=True)


def cell_line_catalog() -> BioEntityCatalog:
    return memory_catalog(
        [
            (1, "cell_line", "HeLa", "CVCL_0030", "Homo sapiens", 9606),
            (2, "cell_line", "MCF-7", "CVCL_0031", "Homo sapiens", 9606),
            (3, "cell_line", "MCF10A", "CVCL_0598", "Homo sapiens", 9606),
            (4, "cell_line", "MDA-MB-231", "CVCL_0062", "Homo sapiens", 9606),
            (5, "cell_line", "IEC-6", "CVCL_0343", "Rattus norvegicus", 10116),
        ],
        [
            ("HeLa", 1),
            ("MCF-7", 2),
            ("MCF10A", 3),
            ("MDA-MB-231", 4),
            ("IEC-6", 5),
        ],
    )


class TestOrganismLabel(unittest.TestCase):
    def test_abbreviated_binomials(self):
        self.assertTrue(is_organism_label("S. aureus"))
        self.assertTrue(is_organism_label("P. aeruginosa"))
        self.assertTrue(is_organism_label("S. pneumoniae"))
        self.assertTrue(is_organism_label("S. simulans"))

    def test_strain_tokens(self):
        self.assertTrue(is_organism_label("ATCC 25922"))
        self.assertTrue(is_organism_label("H37Rv"))
        self.assertTrue(is_organism_label("E. coli strain K12"))

    def test_protein_phrase_not_organism(self):
        self.assertFalse(is_organism_label("Epidermal growth factor receptor"))
        self.assertFalse(is_organism_label("HeLa"))
        self.assertFalse(is_organism_label("MCF-7 cells"))

    def test_classify_without_catalog(self):
        self.assertEqual(classify_target_type("S. aureus"), "organism")
        self.assertEqual(classify_target_type("P. aeruginosa"), "organism")
        self.assertIsNone(classify_target_type("EGFR"))
        self.assertIsNone(classify_target_type("HeLa"))

    def test_extract_organism_span(self):
        self.assertEqual(extract_organism_label("T. gondii"), "T. gondii")
        self.assertEqual(extract_organism_label("in T. gondii"), "T. gondii")
        self.assertEqual(
            extract_organism_label("[' in T. gondii ']"),
            "T. gondii",
        )
        self.assertEqual(extract_organism_label("S. aureus"), "S. aureus")
        self.assertIsNone(extract_organism_label("IEC-6 cells"))
        self.assertIsNone(extract_organism_label("in"))


class TestCatalogLookup(unittest.TestCase):
    def test_hela_case_insensitive(self):
        catalog = cell_line_catalog()
        self.addCleanup(catalog.close)
        self.assertEqual(classify_target_type("HeLa", catalog), "cell_line")
        self.assertEqual(classify_target_type("hela", catalog), "cell_line")

    def test_egfr_human_preferred(self):
        catalog = memory_catalog(
            [
                (1, "protein", "Egfr", "P00533-MOUSE", "Mus musculus", 10090),
                (2, "protein", "EGFR", "P00533", "Homo sapiens", 9606),
            ],
            [("EGFR", 1), ("EGFR", 2)],
        )
        self.addCleanup(catalog.close)
        self.assertEqual(classify_target_type("EGFR", catalog), "single_protein")

    def test_human_preferred_over_lower_id(self):
        catalog = memory_catalog(
            [
                (1, "cell_line", "Mouse X", "CVCL_M", "Mus musculus", 10090),
                (2, "protein", "Human X", "P00001", "Homo sapiens", 9606),
            ],
            [("SHARED", 1), ("SHARED", 2)],
        )
        self.addCleanup(catalog.close)
        self.assertEqual(classify_target_type("SHARED", catalog), "single_protein")

    def test_lowest_id_when_no_human(self):
        catalog = memory_catalog(
            [
                (1, "protein", "Mouse EGFR", "P00533-MOUSE", "Mus musculus", 10090),
                (2, "cell_line", "Rat EGFR", "CVCL_R", "Rattus norvegicus", 10116),
            ],
            [("EGFR", 1), ("EGFR", 2)],
        )
        self.addCleanup(catalog.close)
        self.assertEqual(classify_target_type("EGFR", catalog), "single_protein")

    def test_space_token_fallback(self):
        catalog = cell_line_catalog()
        self.addCleanup(catalog.close)
        self.assertEqual(classify_target_type("MCF-7 cells", catalog), "cell_line")

    def test_full_string_alias(self):
        catalog = memory_catalog(
            [
                (
                    1,
                    "protein",
                    "Epidermal growth factor receptor",
                    "P00533",
                    "Homo sapiens",
                    9606,
                ),
            ],
            [("Epidermal growth factor receptor", 1), ("EGFR", 1)],
        )
        self.addCleanup(catalog.close)
        self.assertEqual(
            classify_target_type("Epidermal growth factor receptor", catalog),
            "single_protein",
        )

    def test_open_missing_file(self):
        self.assertIsNone(open_catalog(Path("no_such_bio_entities.db")))
        self.assertIsNone(open_catalog(None))

    def test_resolve_match_first_letter_strip(self):
        catalog = memory_catalog(
            [
                (1, "protein", "SYMBOL", "P00001", "Homo sapiens", 9606),
                (2, "protein", "foo-bar", "P00002", "Homo sapiens", 9606),
            ],
            [("SYMBOL", 1), ("foo-bar", 2)],
        )
        self.addCleanup(catalog.close)
        self.assertEqual(
            catalog.resolve_match("mSYMBOL cells"),
            ("mSYMBOL", "single_protein"),
        )
        self.assertEqual(
            catalog.resolve_match("mSYMBOL"),
            ("mSYMBOL", "single_protein"),
        )
        self.assertEqual(
            catalog.resolve_match("SYMBOL"),
            ("SYMBOL", "single_protein"),
        )
        self.assertEqual(catalog.resolve_type("mSYMBOL cells"), "single_protein")
        self.assertEqual(
            catalog.resolve_match("xfoo-bar"),
            ("xfoo-bar", "single_protein"),
        )

    def test_stopword_not_matched_as_protein(self):
        catalog = memory_catalog(
            [
                (1, "protein", "IN", "Q9BXR3", "Homo sapiens", 9606),
                (2, "protein", "N", "P00001", "Homo sapiens", 9606),
            ],
            [("IN", 1), ("N", 2)],
        )
        self.addCleanup(catalog.close)
        self.assertIsNone(catalog.resolve_match("in"))
        self.assertIsNone(catalog.resolve_match("IN"))
        self.assertIsNone(classify_target_type("in", catalog))

    def test_stopwords_stripped_before_cell_line_lookup(self):
        catalog = cell_line_catalog()
        self.addCleanup(catalog.close)
        self.assertEqual(
            catalog.resolve_match("in IEC-6 cells"),
            ("IEC-6", "cell_line"),
        )
        self.assertEqual(
            classify_target_type("in IEC-6 cells", catalog),
            "cell_line",
        )
        self.assertEqual(
            classify_target_type("IEC-6 cells", catalog),
            "cell_line",
        )


if __name__ == "__main__":
    unittest.main()
