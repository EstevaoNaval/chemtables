"""Unit tests for merging figure_title blocks above the first table."""

from __future__ import annotations

import unittest

from chemtables.workers.paddleocr_vl.title_merge import merge_titles_above_table


def _block(label: str, content: str) -> dict:
    return {"block_label": label, "block_content": content}


class TestMergeTitlesAboveTable(unittest.TestCase):
    def test_split_table_number_and_caption(self):
        title = merge_titles_above_table(
            [
                _block("figure_title", "Table 2"),
                _block(
                    "figure_title",
                    "In vitro anti-T. gondii activity of indole derivatives.",
                ),
                _block("table", "<table></table>"),
            ]
        )
        self.assertEqual(
            title,
            "Table 2 In vitro anti-T. gondii activity of indole derivatives.",
        )

    def test_two_prose_titles(self):
        title = merge_titles_above_table(
            [
                _block("figure_title", "In vitro activity"),
                _block("figure_title", "of indole derivatives."),
                _block("table", "<table></table>"),
            ]
        )
        self.assertEqual(title, "In vitro activity of indole derivatives.")

    def test_single_complete_title(self):
        title = merge_titles_above_table(
            [
                _block(
                    "figure_title",
                    "Table 2. In vitro susceptibility of isolates.",
                ),
                _block("table", "<table></table>"),
            ]
        )
        self.assertEqual(title, "Table 2. In vitro susceptibility of isolates.")

    def test_title_after_table_only(self):
        title = merge_titles_above_table(
            [
                _block("table", "<table></table>"),
                _block("figure_title", "Table 3. Next crop caption."),
            ]
        )
        self.assertIsNone(title)

    def test_title_above_and_below(self):
        title = merge_titles_above_table(
            [
                _block(
                    "figure_title",
                    "Table 2. In vitro susceptibility of isolates.",
                ),
                _block("table", "<table></table>"),
                _block("vision_footnote", "MIC: minimum inhibitory concentration."),
                _block("figure_title", "Table 3. Geometric means."),
            ]
        )
        self.assertEqual(title, "Table 2. In vitro susceptibility of isolates.")

    def test_no_figure_title(self):
        title = merge_titles_above_table([_block("table", "<table></table>")])
        self.assertIsNone(title)


if __name__ == "__main__":
    unittest.main()
