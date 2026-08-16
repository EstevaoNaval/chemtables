#!/usr/bin/env python3
"""
Header-row-counting task logic.

Pure task logic (table normalization, prompt building, output parsing,
deterministic backoff) -- no model/tokenizer loading here. Generation is
injected via `generate_fn(messages, max_new_tokens=...) -> str` (defaults
to chemtables.gemma_client.default_generate).
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable

MAX_TABLE_ROWS = int(os.getenv("CHEMTABLES_MAX_TABLE_ROWS", "12"))
MAX_NEW_TOKENS = 4

GenerateFn = Callable[..., str]

SYSTEM_PROMPT = """You count header rows in extracted tables.
A header row describes column names, groups, units, data types, or categories.
Header rows are contiguous and always start at row 1.
Data rows begin at the first row that contains actual records.
Return exactly one positive integer: the number of header rows.
Do not explain. Do not output punctuation or any other text."""

HEADER_CUES = re.compile(
    r"\b(unit|units|numeric|categorical|category|type|class|assay|activity|"
    r"toxicity|concentration|dose|value|mean|sd|std|mg|kg|mol|mmol|µm|um|"
    r"yes/no|active/inactive)\b",
    flags=re.IGNORECASE,
)


def normalize_table(table_rows):
    if not isinstance(table_rows, list) or not table_rows:
        raise ValueError("Input must be a non-empty list of rows.")
    if not all(isinstance(row, list) for row in table_rows):
        raise ValueError("Each table row must be a list.")
    return [["" if cell is None else str(cell).strip() for cell in row] for row in table_rows]


def serialize_table(table_rows):
    return json.dumps(table_rows[:MAX_TABLE_ROWS], ensure_ascii=False, separators=(",", ":"))


def build_messages(table_rows):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"TABLE_ROWS:\n{serialize_table(table_rows)}\n\n"
        "HEADER_ROW_COUNT:"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def parse_positive_integer(raw, row_count):
    match = re.search(r"\b([1-9]\d*)\b", raw or "")
    if not match:
        return None
    value = int(match.group(1))
    if value < 1 or value > row_count:
        return None
    return value


def looks_numeric(value: str) -> bool:
    value = value.strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value))


def deterministic_backoff(table_rows) -> int:
    for index, row in enumerate(table_rows):
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        numeric_ratio = sum(looks_numeric(cell) for cell in cells) / len(cells)
        cue_count = sum(bool(HEADER_CUES.search(cell)) for cell in cells)
        if index > 0 and numeric_ratio >= 0.35 and cue_count == 0:
            return index
    return 1


def count_header_rows(
    table_rows,
    generate_fn: GenerateFn | None = None,
):
    if generate_fn is None:
        from chemtables.gemma_client import default_generate

        generate_fn = default_generate

    table_rows = normalize_table(table_rows)
    raw = generate_fn(build_messages(table_rows), max_new_tokens=MAX_NEW_TOKENS)
    count = parse_positive_integer(raw, len(table_rows))
    source = "model"
    if count is None:
        count = deterministic_backoff(table_rows)
        source = "deterministic_backoff"
    return count, raw, source


if __name__ == "__main__":
    from chemtables.gemma_client import GemmaSession

    table_rows = [
        ['Compounds', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity', 'Toxicity'],
        ['Compounds', 'Oral Rat Acute Toxicity (LD50)', 'Oral Rat Chronic Toxicity (LOAEL)', 'Minnow Toxicity', 'HERG I', 'HERG II', 'Hepatotoxicity', 'Toxicological End Points', 'Toxicological End Points', 'Toxicological End Points', 'Toxicological End Points'],
        ['Compounds', '', '', '', '', '', '', 'Immunotoxicity', 'Carcinogenicity', 'Cytotoxicity', 'Mutagenicity'],
        ['Compounds', 'Numeric (mol/kg)', 'Numeric (log mg/kg_bw/day)', 'Numeric (log LC 50)', 'Categorical (Yes/No)', 'Categorical (Yes/No)', 'Categorical (Yes/No)', 'Categorical (Active/Inactive)', 'Categorical (Active/Inactive)', 'Categorical (Active/Inactive)', 'Categorical (Active/Inactive)'],
        ['Itraconazole', '2.966', '0.055', '-4.446', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
        ['Q1', '1.952', '2.322', '-2.223', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
        ['Q2', '2.098', '2.398', '-2.889', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
        ['Q3', '2.245', '2.381', '-2.098', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
        ['Q4', '2.580', '1.549', '-1.846', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes'],
        ['Q5', '2.577', '1.581', '-1.211', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes'],
        ['Q6', '2.972', '1.221', '0.919', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No'],
        ['Q7', '2.965', '2.426', '0.274', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No'],
        ['Q8', '2.711', '2.974', '-6.407', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
        ['Q9', '2.844', '1.724', '-0.550', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No'],
        ['Q10', '2.905', '1.578', '-1.852', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No'],
        ['Q11', '2.247', '2.597', '-2.126', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
        ['H1', '2.596', '1.380', '0.167', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes'],
        ['H2', '2.549', '1.340', '0.644', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No'],
        ['H3', '2.984', '1.238', '-0.396', 'No', 'No', 'No', 'No', 'No', 'No', 'No'],
    ]
    with GemmaSession() as session:
        count, raw, source = count_header_rows(table_rows, generate_fn=session.generate)
    print(f"Count: {count}")
    print(f"Raw: {raw}")
    print(f"Source: {source}")
