#!/usr/bin/env python3
"""
Table header/schema interpretation task logic.

Given a table's title, header rows (already isolated -- e.g. via
count_header_rows), optional sample body rows, and footnotes, ask
an LLM (via injectable generate_fn) to interpret what each column means:
the property/metric being measured, its unit, its experimental context
(assay, organism, condition), and any footnote markers attached to it.

This module contains only task logic (prompt building, output parsing,
deterministic backoff) -- no model/tokenizer loading. Pass
`generate_fn(messages, max_new_tokens=...) -> str` (e.g. GemmaSession.generate).

Programmatic:
    from chemtables.schema.interpret import interpret_table_schema
    schema, raw, source = interpret_table_schema(
        title, header_rows, footnotes, sample_rows=body_rows,
        compound_refs=["REF_A", "REF_B"],
        generate_fn=session.generate,
    )
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text

from chemtables.matching.bio_entity import finalize_target_type
from chemtables.matching.compound import (
    CompoundAxis,
    compound_axis_to_dict,
    locate_compound_axis,
)
from chemtables.schema.header_rows import count_header_rows

GenerateFn = Callable[..., str]

MAX_HEADER_ROWS = int(os.getenv("CHEMTABLES_MAX_HEADER_ROWS", 12))
MAX_SAMPLE_ROWS = int(os.getenv("CHEMTABLES_SCHEMA_MAX_SAMPLE_ROWS", 3))
MAX_NEW_TOKENS = int(os.getenv("CHEMTABLES_SCHEMA_MAX_NEW_TOKENS", 2048))

# Prompt-facing allowed unit families (from Metric-*.csv reference).
DEFAULT_METRIC_UNITS = {
    "IC50": "μM, nM, or M",
    "EC50": "μM, nM, or M",
    "MIC": "μg/mL or mg/L",
    "MBC": "μg/mL or mg/L",
    "MFC": "μg/mL or mg/L",
    "SI": "unitless",
    "ED50": "mg/kg or mg",
    "EE50": "mg/kg or mg",
}

# Single-token post-fill when model leaves unit null (first preferred unit).
DEFAULT_METRIC_UNIT_FALLBACK = {
    "IC50": "μM",
    "EC50": "μM",
    "MIC": "μg/mL",
    "MBC": "μg/mL",
    "MFC": "μg/mL",
    "SI": "unitless",
    "ED50": "mg/kg",
    "EE50": "mg/kg",
}

SYSTEM_PROMPT = """You interpret the header of a scientific data table extracted from a \
chemistry/biology paper. You are given the table TITLE, the HEADER_ROWS \
(one or more contiguous top rows describing the columns), SAMPLE_ROWS \
(a few leading body rows for context only; may be empty), FOOTNOTES \
(markers referenced from the header via plain superscripts, e.g. ^a), \
TITLE_UNIT_HINTS (unit-like tokens pattern-matched out of TITLE/FOOTNOTES; \
see the rules below for how to use them), DEFAULT_METRIC_UNITS \
(canonical metric -> common default unit string), COMPOUND_AXIS \
(deterministic fact: where chemical subject IDs sit in this table), and \
COMPOUND_COREFERENCE_TOKENS (the ID strings themselves — exclude from \
"context" only; do NOT use them to decide layout or identifier roles).

Input text is already plain (LaTeX math converted). Subscripts appear as \
_... (e.g. CC_50, IC_50); superscripts appear as ^... (e.g. IC_50^a, \
leading ^a on a cell).

COMPOUND_AXIS is authoritative and already decided outside this task:
- "layout" is "compounds_in_rows" or "compounds_in_columns" (or null when \
  the axis could not be located — then leave roles as best-effort property \
  semantics only).
- "compound_column" is the 0-based column index of the compound axis when \
  known.
Rules:
- Do NOT invent, override, or contradict COMPOUND_AXIS.layout or \
  compound_column. Do NOT emit a top-level "layout" field (it is ignored).
- Do NOT put any COMPOUND_COREFERENCE_TOKENS string into "context".
- Do NOT treat those tokens as organism, cell line, enzyme, condition, \
  or other experimental complementary text.
- Do NOT invent chemical IDs absent from COMPOUND_COREFERENCE_TOKENS.
- Leave raw header wording in "header_path", but exclude matched \
  coreference tokens from "context".
- When layout is "compounds_in_rows", column compound_column is the \
  chemical identifier column — you may still describe other columns; \
  the pipeline will force that column's role to identifier.
- When layout is "compounds_in_columns", chemical IDs live in property \
  headers (often mixed with metric/unit); the identifier column holds \
  an assay dimension (cell line, organism, isolate, etc.), NOT compounds.

Your job is METRICS and ASSAY semantics only: property_name, unit, \
context, target_name, target_type, footnote_refs, header_path, and \
top-level assay. Classify non-axis columns as "property" or "other" \
(and optionally "identifier" only for the non-compound row-label column \
when layout is compounds_in_columns).

For EACH column (0-indexed, left to right), determine:
- "column_index": integer index of the column.
- "role": one of "identifier", "property", or "other" (see COMPOUND_AXIS \
  rules above; the pipeline may overwrite the compound-axis role).
- "property_name": short canonical name of the measured property/metric \
  (e.g. "MIC", "MFC", "MBC", "EC50", "EE50 (EC50)", "CC50", "SI", "ED50", "IC50"). \
  Use null for identifier/other columns.
- "context": complementary experimental information from the header \
  (and from TITLE/FOOTNOTES when they apply to that column) that is \
  not the property/metric name, not the unit, not reporting metadata \
  (see rule below), and not any COMPOUND_COREFERENCE_TOKENS token. Join \
  multiple pieces with commas if needed. Use null only when none is present.
- "unit": the unit of measurement, if stated or implied (e.g. "uM", \
  "nM", "%", "h", "mg/kg", "μg/mL"). Use null if none.
- "footnote_refs": list of footnote marker strings (letters/numbers) \
  attached to this column via superscript notation in the header. Use \
  an empty list if none.
- "header_path": list of the raw header text(s), top row to bottom row, \
  that apply to this column (skip empty cells from merged/rowspan cells).
- "target_name": for "property" columns only — verbatim biological entity \
  the measurement is against (cell line, organism/strain, protein/enzyme, \
  etc.). Prefer the entity named in THAT COLUMN's own header. Use \
  TITLE/FOOTNOTES only when the column header does not name a biological \
  entity. Use null when none is stated or when the layout is \
  "compounds_in_columns" (row identifiers carry the target). Never invent. \
  Do not assume a naming format for entities.
- "target_type": for "property" columns only — one of \
  "single_protein", "protein_family", "protein_complex", "cell_line", \
  "organism", "dna", "rna", "non_molecular", or null. Classify ONLY from \
  evidence in the table text; when unsure use "organism" or null. Never \
  guess. Use null for identifier/other columns.

Also set top-level "assay" from TITLE and FOOTNOTES (table-wide):
- "description": verbatim or near-verbatim assay/experiment description \
  from the title/caption (and footnotes when they describe the method). \
  Use null only when nothing useful is present.
- "type": coarse category or null — "B" (binding), "F" (functional, e.g. \
  enzyme activity / growth inhibition / antimicrobial), "A" (ADME), \
  "T" (toxicity / cytotoxicity against host/normal cells), "P" \
  (physicochemical), "U" (unclassified), or null when insufficient \
  evidence. Do NOT invent assay details not present in TITLE/FOOTNOTES. \
  Do NOT emit provenance, compound records, page/DOI, or evidence quotes.

Treat _... subscript markers as PART OF a property/context name \
(e.g. "IC_50" / "CC_50" means property_name "IC50" / "CC50" -- canonicalize \
by dropping the underscore). Treat ^... superscript markers as a FOOTNOTE \
REFERENCE only (e.g. "IC_50^a" -> property_name "IC50", footnote_refs ["a"]); \
never put the marker in property_name or context.

Unit of measurement and experimental context for header and data rows are \
often stated in the TITLE/caption or FOOTNOTES rather than in the header \
cells themselves (e.g. a title that states a metric and UNIT_A means every \
property column's unit is UNIT_A even though no header cell says so).

Header cells often mix a metric name, a unit, and complementary text. \
After extracting "property_name" and "unit", and after stripping \
reporting metadata (see rule below) and any COMPOUND_COREFERENCE_TOKENS \
tokens, put EVERY remaining non-empty complementary header token that \
qualifies what was measured into "context" (comma-joined if several). \
Prefer completeness over brevity: do not drop tokens to make "context" \
shorter or prettier. Must-keep categories include organism/species, \
receptor/target/enzyme, cell line, assay readout, strain/isolate, \
condition/timepoint, and any other column-specific qualifier — including \
short or abbreviated tokens that distinguish sibling columns — except \
COMPOUND_COREFERENCE_TOKENS, which must never enter "context".
- Uniqueness: no two "property" columns may share the same \
  ("property_name", "context", "unit") if their "header_path" texts \
  differ. If a draft would collide, missing complementary tokens from \
  that column's own header MUST be added to "context" until the tuples \
  differ; this is required, not optional polish. Prefer non-coreference \
  complementary tokens for that disambiguation.
- TITLE/FOOTNOTE complementary info may apply to columns, but does not \
  excuse dropping complementary text that appears only in a column's \
  own header.
- Do not "simplify" by keeping only a shared complementary phrase when \
  sibling columns also differ elsewhere in the header; include that \
  column's complementary pieces in "context".

Reporting metadata describes HOW a value is reported (statistics, \
uncertainty, replication), not WHAT was assayed. Common forms include \
uncertainty labels often preceded by ± / \\pm (SD, SEM, SE, RSD; \
"mean ± SD", "average ± …"), and replication/sample-size notes \
(n=3, triplicate, duplicate, "independent experiments"). These are \
NOT experimental context and must NOT appear in "context" (nor in \
"property_name" or "unit"). They only indicate that body cells may \
use forms such as value ± error. Strip all reporting metadata from \
the fields; retain only genuine experimental qualifiers (e.g. \
organism, enzyme, cell line, condition).

TITLE_UNIT_HINTS lists unit-like tokens found by simple pattern-matching \
anywhere in TITLE/FOOTNOTES. It is NOISY and may contain false positives \
(units mentioned only as an example, or units for a different metric than \
this table reports). Use it to decide each "property" column's "unit", \
following these rules:
- If TITLE_UNIT_HINTS has exactly one hint, and nothing in the header \
  contradicts it, apply that hint as the "unit" for every "property" \
  column that doesn't already have a more specific unit of its own.
- Do NOT apply a hint that only appears in an illustrative/example clause \
  of the title or footnotes (cues like "e.g.", "for example", "such as", \
  "representative") -- that describes an example, not the table's actual \
  reported values.
- If TITLE_UNIT_HINTS has multiple hints, only assign each one to the \
  property column(s)/metric it actually describes (e.g. one metric in uM, \
  another in ug/mL); never apply one hint to a column it doesn't describe.
- Prefer a unit from the header, TITLE/FOOTNOTES, or TITLE_UNIT_HINTS when \
  it clearly applies.
- If a "property" column still has no unit, and its property_name appears in \
  DEFAULT_METRIC_UNITS, use that entry. When the entry lists several options \
  (e.g. "μM, nM, or M"), treat them as an allowed family and pick one common \
  unit unless context indicates otherwise.
- Leave "unit" null only when the metric is not in DEFAULT_METRIC_UNITS and \
  no textual unit applies; do not guess from unrelated hints.

SAMPLE_ROWS are leading body rows provided only as context. They are NOT \
header rows: never put sample cell text into "header_path". Use them when \
a metric name (e.g. MIC, MFC, IC50) appears in a sample cell rather than \
in the header -- in that case, sibling columns whose headers are only \
organism/isolate/condition labels are still "property" columns for that \
metric, and TITLE/FOOTNOTE unit rules still apply to them. If SAMPLE_ROWS \
is empty, rely on HEADER_ROWS alone.

Examples (abbreviated, illustrating the rules above -- placeholders, not \
literal papers or entity names):
- TITLE states METRIC_A (UNIT_A), TITLE_UNIT_HINTS [UNIT_A], property \
  column with no unit in the header -> that column's "unit" is UNIT_A.
- TITLE mentions UNIT_B only in an "e.g." / "for example" clause, \
  TITLE_UNIT_HINTS [UNIT_B] -> UNIT_B is illustrative, not the table's \
  reported unit -> leave "unit" null (unless a column's own \
  header/footnote states a unit).
- TITLE states METRIC_A (UNIT_A) and METRIC_B (UNIT_B), TITLE_UNIT_HINTS \
  [UNIT_A, UNIT_B] -> assign UNIT_A to the METRIC_A column and UNIT_B to \
  the METRIC_B column; do not swap them or apply both to either column.
- TITLE_UNIT_HINTS [UNIT_A], HEADER cells are isolate labels (ISOLATE_A), \
  SAMPLE_ROWS show METRIC_A / METRIC_B next to numeric values -> those \
  isolate columns are "property" with unit UNIT_A (header_path stays the \
  isolate labels; do not copy the sample metric into header_path).
- HEADER has two property columns with the same metric and unit, e.g. \
  "METRIC_A QUAL_A (UNIT_A)\\nSHARED_PHRASE" and \
  "METRIC_A QUAL_B (UNIT_A)\\nSHARED_PHRASE" -> both get property_name \
  METRIC_A and unit UNIT_A, but different "context" values that each keep \
  that column's complementary text (QUAL_A vs QUAL_B) and SHARED_PHRASE. \
  Do not collapse both contexts to only SHARED_PHRASE.
- HEADER "METRIC_A TARGET_X (UNIT_A) SHARED_PHRASE" vs \
  "METRIC_A TARGET_Y (UNIT_A) SHARED_PHRASE" -> property_name METRIC_A, \
  unit UNIT_A for both; contexts must keep TARGET_X / TARGET_Y and \
  SHARED_PHRASE. Do not drop the middle qualifier tokens and collapse \
  both to only SHARED_PHRASE.
- HEADER "METRIC_A TARGET_X ± SD, UNIT_A^a" -> property_name METRIC_A, \
  context "TARGET_X", unit UNIT_A, footnote_refs ["a"]. Do not put \
  "± SD" or "SD" in context.
- HEADER "METRIC_A ± SEM (n=3), UNIT_A" against TARGET_X -> property_name \
  METRIC_A, context "TARGET_X", unit UNIT_A. Strip "± SEM" and "n=3"; \
  keep the qualifier only.
- If a property header mixes a metric/unit with a COMPOUND_COREFERENCE_TOKENS \
  token, keep the token in "header_path" and leave "context" for true \
  experimental qualifiers only (or null).
- TITLE names HOST_PHRASE and the column header names TARGET_X -> \
  target_name "TARGET_X", not HOST_PHRASE. Set target_type only from \
  evidence in the table text (or "organism"/null).
- HEADER/context with abbreviated binomial "X. species" -> target_name \
  "X. species", target_type "organism".

Return ONLY a JSON object with this exact shape, no explanation, no \
markdown fences (do NOT include "layout"):
{"assay": {"description": ...|null, "type": "B"|"F"|"A"|"T"|"P"|"U"|null}, \
"columns": [{"column_index": 0, "role": "...", "property_name": ..., \
"context": ..., "unit": ..., "footnote_refs": [...], "header_path": [...], \
"target_name": ...|null, "target_type": ...|null}, ...]}
"""

VALID_LAYOUTS = frozenset({"compounds_in_rows", "compounds_in_columns"})
DEFAULT_LAYOUT = "compounds_in_rows"
VALID_ASSAY_TYPES = frozenset({"B", "F", "A", "T", "P", "U"})
VALID_TARGET_TYPES = frozenset(
    {
        "single_protein",
        "protein_family",
        "protein_complex",
        "cell_line",
        "organism",
        "dna",
        "rna",
        "non_molecular",
    }
)
DEFAULT_ASSAY = {"description": None, "type": None}


def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)




def latex_mixed_text_to_plain_text(text: str, math_mode: str = "text") -> str:
    """
    Convert a mixed text/LaTeX string to plain text more safely than sending the
    whole raw string directly to pylatexenc.

    Good for inputs like:
        "$ ^{a} $Concentration ... $ \\pm $ ... $ CC_{50} $ ..."

    Parameters
    ----------
    text : str
        Input string containing prose mixed with inline LaTeX.
    math_mode : str
        Passed to pylatexenc LatexNodes2Text(..., math_mode=...).
        Common values: "text", "with-delimiters", "verbatim", "remove".

    Returns
    -------
    str
        Cleaned plain-text output.
    """

    converter = LatexNodes2Text(math_mode=math_mode)

    s = text

    # Normalize common malformed inline-math spacing from OCR/PDF extraction
    s = re.sub(r'\$\s*\^\{([^}]*)\}\s*\$', r' $ ^{\1} $ ', s)
    s = re.sub(r'\$\s*\\([A-Za-z]+)\s*\$', r' $ \\\1 $ ', s)
    s = re.sub(r'\$\s*([A-Za-z]+(?:_\{[^}]+\})?)\s*\$', r' $ \1 $ ', s)

    # Split into inline math and non-math chunks
    parts = re.split(r'(\$.*?\$)', s)

    out = []
    for part in parts:
        if not part:
            continue

        if part.startswith("$") and part.endswith("$"):
            try:
                converted = converter.latex_to_text(part)
            except Exception:
                converted = part[1:-1]
            out.append(converted)
        else:
            out.append(part)

    result = "".join(out)

    # Whitespace cleanup
    result = re.sub(r'[ \t]+', ' ', result)
    result = re.sub(r'\s+([,.;:])', r'\1', result)
    result = re.sub(r'\n\s+', '\n', result)

    return result.strip()


def plain_cell(text) -> str:
    """Convert one cell/title/footnote string to plain text for the model."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return latex_mixed_text_to_plain_text(text)


def to_model_plain_text(title, header_rows, footnotes, sample_rows=None):
    """Map title/headers/footnotes/samples to plain text for Gemma.

    Footnote dict keys (markers) stay unchanged; only values are converted.
    Call after normalize_footnotes so marker splitting still sees raw LaTeX.
    """
    plain_title = plain_cell(title) if title else ""
    plain_headers = [[plain_cell(cell) for cell in row] for row in (header_rows or [])]
    plain_samples = [[plain_cell(cell) for cell in row] for row in (sample_rows or [])]
    if isinstance(footnotes, dict):
        plain_footnotes = {k: plain_cell(v) for k, v in footnotes.items()}
    else:
        plain_footnotes = {}
    return plain_title, plain_headers, plain_footnotes, plain_samples


def sanitize_row(row: list) -> list:
    for i, cell in enumerate[Any](row):
        if isinstance(cell, str):
            cell = cell.replace(r'\n', ' ')
            cell = re.sub(r'\s+', ' ', cell)
            cell = cell.strip()
        
        if cell is None:
            cell = ""
        
        row[i] = cell
    
    return row

def normalize_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list of rows.")
    if not all(isinstance(row, list) for row in rows):
        raise ValueError("Each row must be a list.")
    return [sanitize_row(row) for row in rows]


normalize_header_rows = normalize_rows


def select_sample_rows(body_rows, n_columns: int, max_rows: int = MAX_SAMPLE_ROWS) -> list:
    """Return up to `max_rows` leading body rows, padded/truncated to n_columns.

    Caller owns the header/body split; this only caps and shapes samples for
    the model payload. Falsy body_rows yields an empty list.
    """
    if not body_rows:
        return []
    normalized = normalize_rows(body_rows)
    samples = []
    for row in normalized[:max_rows]:
        padded = list(row[:n_columns])
        if len(padded) < n_columns:
            padded.extend([""] * (n_columns - len(padded)))
        samples.append(padded)
    return samples


# Distinctive multi-character unit tokens: safe to match bare, anywhere,
# since they're very unlikely to false-positive on ordinary prose (unlike
# bare "g", "M", "h", "s", which collide with abbreviations like "e.g.").
UNIT_HINT_DISTINCTIVE_RE = re.compile(
    r"""(?xi)
    (?:
        \b(?:
            [uµμn]?[mM]?g/(?:m?[lL]|kg|d[lL]|g)   |  # ug/mL, mg/kg, ng/mL, mg/dL, ug/g ...
            [uµμn][mM]?M                           |  # uM, nM, mM (never bare M here)
            m?mol(?:/[lL])?                        |  # mmol, mol/L
            (?:°|deg(?:ree)?s?\s*)[CF]               |  # °C, degrees C
            k[dD]a                                 |  # kDa
            ppm|ppb                                |
            I[uU]                                  |  # IU
            m[mM]Hg                                |
            min(?:utes?)?                          |
            sec(?:onds?)?                          |
            hours?|days?|weeks?
        )\b
        |
        %   # not word-bounded: '%' is non-word on both sides so \b never matches it
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Ambiguous short unit tokens ("g", "M", "h", "s") that only count as a
# unit hint when immediately preceded by a number, e.g. "10 mg" or "24 h",
# to avoid matching stray letters inside words/abbreviations (e.g. "e.g.").
UNIT_HINT_NUMERIC_PREFIXED_RE = re.compile(
    r"(?i)\d\s*([uµμn]?[mM]?g|M|h|s)\b"
)


def _flatten_footnotes_text(footnotes) -> str:
    """Best-effort flattening of a footnotes value (dict, str, or falsy)
    into a single text blob for hint scanning."""
    if not footnotes:
        return ""
    if isinstance(footnotes, dict):
        return " ".join(str(v) for v in footnotes.values())
    return str(footnotes)


def extract_unit_hints(title, footnotes, max_hints: int = 6) -> list:
    """Detect candidate unit-like tokens anywhere in TITLE or FOOTNOTES via
    regex, to be surfaced to the model as hints (never auto-applied).

    This is a noisy, best-effort scan: it may catch false positives (units
    mentioned only as an example, or units for a property the table doesn't
    even have). The model is responsible for deciding, per column, whether
    a hint actually applies -- see the TITLE_UNIT_HINTS rules in
    SYSTEM_PROMPT. Returns a de-duplicated list capped at `max_hints`.
    """
    blob = f"{title or ''} {_flatten_footnotes_text(footnotes)}"
    seen = []

    def add(token):
        token = token.strip()
        if token and token not in seen:
            seen.append(token)

    for match in UNIT_HINT_DISTINCTIVE_RE.finditer(blob):
        add(match.group(0))
        if len(seen) >= max_hints:
            return seen
    for match in UNIT_HINT_NUMERIC_PREFIXED_RE.finditer(blob):
        add(match.group(1))
        if len(seen) >= max_hints:
            return seen
    return seen


FOOTNOTE_MARKER_SPLIT_RE = re.compile(r"\$\s*\^\s*\{?\s*([a-zA-Z0-9]{1,3})\s*\}?\s*\$\s*")


def normalize_footnotes(footnotes):
    if not footnotes:
        return {}
    if isinstance(footnotes, dict):
        return {str(k).strip(): str(v).strip() for k, v in footnotes.items()}
    if isinstance(footnotes, str):
        # Real footnote markers only ever appear as the LaTeX superscript
        # idiom "$ ^{a} $" (as emitted by PaddleOCR-VL's vision_footnote
        # blocks). Split on that pattern specifically so stray alnum chars
        # inside the footnote body (e.g. "$ \mu $", "$ ^{-1} $") are never
        # mistaken for markers.
        text = footnotes.strip()
        parts = FOOTNOTE_MARKER_SPLIT_RE.split(text)
        if len(parts) > 1:
            result = {}
            for i in range(1, len(parts), 2):
                marker = parts[i].strip()
                body = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if marker:
                    result[marker] = body
            if result:
                return result
        return {"raw": text}
    raise ValueError("footnotes must be a dict, a string, or falsy.")


def serialize_payload(
    title,
    header_rows,
    footnotes,
    title_unit_hints=None,
    sample_rows=None,
    default_metric_units=None,
    compound_axis=None,
    compound_coreference_tokens=None,
):
    axis_payload = compound_axis
    if isinstance(compound_axis, CompoundAxis):
        axis_payload = {
            "layout": compound_axis.layout,
            "compound_column": compound_axis.compound_column,
            "skip_extract": compound_axis.skip_extract,
        }
    return json.dumps(
        {
            "title": title or "",
            "header_rows": header_rows[:MAX_HEADER_ROWS],
            "sample_rows": sample_rows or [],
            "footnotes": footnotes or {},
            "title_unit_hints": title_unit_hints or [],
            "default_metric_units": default_metric_units
            if default_metric_units is not None
            else DEFAULT_METRIC_UNITS,
            "compound_axis": axis_payload or {},
            "compound_coreference_tokens": list(compound_coreference_tokens or []),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_messages(
    title,
    header_rows,
    footnotes,
    title_unit_hints=None,
    sample_rows=None,
    default_metric_units=None,
    compound_axis=None,
    compound_coreference_tokens=None,
):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"TABLE_JSON:\n{serialize_payload(title, header_rows, footnotes, title_unit_hints, sample_rows, default_metric_units, compound_axis, compound_coreference_tokens)}\n\n"
        "SCHEMA_JSON:"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def extract_json_object(raw: str):
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Common model slip: trailing commas before a closing bracket/brace.
        repaired = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


# Compound patterns first, then short tokens (SEM before SE/SD).
_REPORTING_METADATA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:mean|average|avg)\s*[±\u00b1]\s*(?:RSD|SEM|SD|SE)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:±|\u00b1|\\pm)\s*(?:RSD|SEM|SD|SE)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:RSD|SEM|SD|SE)\s*(?:±|\u00b1|\\pm)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\(?\s*n\s*=\s*\d+\s*\)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bindependent\s+experiments?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:triplicates?|duplicates?|replicates?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:RSD|SEM|SD|SE)\b", re.IGNORECASE),
    re.compile(r"(?:±|\u00b1|\\pm)"),
)


def strip_reporting_metadata(text: str | None) -> str | None:
    """Remove statistical/replication reporting labels; keep experimental qualifiers.

    Returns None when nothing meaningful remains.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    for pattern in _REPORTING_METADATA_PATTERNS:
        s = pattern.sub(" ", s)
    s = re.sub(r"[|,;:/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -_.,;:/()")
    return s or None


def _flatten_header_path(header_path) -> list[str]:
    """Unwrap one nested list from Gemma header_path; keep non-empty strings."""
    if not isinstance(header_path, list):
        return []
    steps: list[str] = []
    for step in header_path:
        inners = step if isinstance(step, list) else [step]
        for inner in inners:
            if inner is None or isinstance(inner, list):
                continue
            text = str(inner).strip()
            if text:
                steps.append(text)
    return steps


def _normalize_assay(raw) -> dict:
    """Coerce assay object; unknown/missing fields become null."""
    if not isinstance(raw, dict):
        return dict(DEFAULT_ASSAY)
    description = raw.get("description")
    if description is not None:
        description = str(description).strip() or None
    assay_type = raw.get("type")
    if assay_type is not None:
        assay_type = str(assay_type).strip().upper() or None
        if assay_type not in VALID_ASSAY_TYPES:
            assay_type = None
    return {"description": description, "type": assay_type}


def _normalize_target_type(
    raw,
    *,
    role: str,
    target_name: str | None = None,
) -> str | None:
    if role != "property":
        return None
    value: str | None
    if raw is None:
        value = None
    else:
        value = str(raw).strip().lower() or None
        if value == "unknown":
            value = None
        elif value not in VALID_TARGET_TYPES:
            value = None
    return finalize_target_type(target_name, value)


def validate_schema(parsed, n_columns: int):
    if not isinstance(parsed, dict):
        return None
    columns = parsed.get("columns")
    if not isinstance(columns, list) or not columns:
        return None

    layout = parsed.get("layout")
    if layout not in VALID_LAYOUTS:
        # Layout is forced later by apply_compound_axis; placeholder OK.
        layout = DEFAULT_LAYOUT

    assay = _normalize_assay(parsed.get("assay"))

    valid_roles = {"identifier", "property", "other"}
    cleaned = []
    seen_indexes = set()
    for entry in columns:
        if not isinstance(entry, dict):
            return None
        try:
            idx = int(entry.get("column_index"))
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= n_columns or idx in seen_indexes:
            return None
        seen_indexes.add(idx)

        role = entry.get("role")
        if role not in valid_roles:
            role = "other"

        footnote_refs = entry.get("footnote_refs") or []
        if not isinstance(footnote_refs, list):
            footnote_refs = []
        footnote_refs = [str(ref).strip() for ref in footnote_refs if str(ref).strip()]

        header_path = _flatten_header_path(entry.get("header_path"))

        target_name = None
        if role == "property":
            raw_target = entry.get("target_name")
            if raw_target is not None:
                target_name = str(raw_target).strip() or None
        target_type = _normalize_target_type(
            entry.get("target_type"),
            role=role,
            target_name=target_name,
        )

        cleaned.append(
            {
                "column_index": idx,
                "role": role,
                "property_name": entry.get("property_name") or None,
                "context": strip_reporting_metadata(entry.get("context") or None),
                "unit": entry.get("unit") or None,
                "footnote_refs": footnote_refs,
                "header_path": header_path,
                "target_name": target_name,
                "target_type": target_type,
            }
        )

    if len(cleaned) != n_columns:
        return None

    cleaned.sort(key=lambda item: item["column_index"])
    return {"layout": layout, "assay": assay, "columns": cleaned}


def _normalize_property_lookup_key(property_name: str) -> str:
    """Canonicalize property_name for DEFAULT_METRIC_UNIT_FALLBACK lookup."""
    return property_name.replace("_", "").strip().upper()


def apply_default_units(schema: dict, fallback: dict | None = None) -> dict:
    """Fill null property units from DEFAULT_METRIC_UNIT_FALLBACK (in place)."""
    if fallback is None:
        fallback = DEFAULT_METRIC_UNIT_FALLBACK
    if not schema or not fallback:
        return schema
    lookup = {_normalize_property_lookup_key(k): v for k, v in fallback.items()}
    for entry in schema.get("columns", []):
        if entry.get("role") != "property":
            continue
        if entry.get("unit"):
            continue
        name = entry.get("property_name")
        if not name:
            continue
        fill = lookup.get(_normalize_property_lookup_key(str(name)))
        if fill:
            entry["unit"] = fill
    return schema


def apply_compound_axis(schema: dict, axis: CompoundAxis) -> dict:
    """Force layout and compound-axis roles from deterministic locator (in place)."""
    if not schema or not isinstance(schema.get("columns"), list):
        return schema

    columns = schema["columns"]
    n_cols = len(columns)

    if axis.skip_extract or axis.layout is None or axis.compound_column is None:
        schema["layout"] = None
        schema["skip_extract"] = True
        for entry in columns:
            entry["role"] = "other"
            entry["property_name"] = None
            entry["target_name"] = None
            entry["target_type"] = None
        return schema

    schema["skip_extract"] = False
    compound_col = int(axis.compound_column)
    if compound_col < 0 or compound_col >= n_cols:
        schema["layout"] = None
        schema["skip_extract"] = True
        for entry in columns:
            entry["role"] = "other"
        return schema

    schema["layout"] = axis.layout

    if axis.layout == "compounds_in_rows":
        for entry in columns:
            idx = entry["column_index"]
            if idx == compound_col:
                entry["role"] = "identifier"
                entry["property_name"] = None
                entry["target_name"] = None
                entry["target_type"] = None
            elif entry.get("role") == "identifier":
                # Only the compound axis column is the chemical identifier.
                entry["role"] = "property"
        return schema

    if axis.layout == "compounds_in_columns":
        # Compound-bearing header column stays property (IDs in header text).
        compound_entry = columns[compound_col]
        if compound_entry.get("role") != "property":
            compound_entry["role"] = "property"

        # Ensure exactly one non-compound identifier column for row labels.
        id_candidates = [
            e for e in columns
            if e["column_index"] != compound_col and e.get("role") == "identifier"
        ]
        if not id_candidates:
            # Prefer col0 if it is not the compound column.
            prefer = 0 if compound_col != 0 else (1 if n_cols > 1 else None)
            if prefer is not None:
                for entry in columns:
                    if entry["column_index"] == prefer:
                        entry["role"] = "identifier"
                        entry["property_name"] = None
                        entry["target_name"] = None
                        entry["target_type"] = None
                        break
        else:
            # Keep the first identifier; demote extras that aren't needed.
            keep_idx = min(e["column_index"] for e in id_candidates)
            for entry in columns:
                if entry["column_index"] == compound_col:
                    continue
                if entry.get("role") == "identifier" and entry["column_index"] != keep_idx:
                    entry["role"] = "property"

        # Never treat the compound header column as identifier.
        columns[compound_col]["role"] = "property"
        return schema

    return schema


FOOTNOTE_MARK_RE = re.compile(r"\^\s*\{?\s*([a-zA-Z0-9])\s*\}?")
SUBSCRIPT_RE = re.compile(r"_\s*\{?\s*([a-zA-Z0-9]+)\s*\}?")
DOLLAR_RE = re.compile(r"\$")


def deterministic_backoff(header_rows, n_columns: int):
    """Rule-based fallback: builds a schema straight from header text,
    without any semantic interpretation (no property/unit/context split).
    Guarantees a valid, well-formed schema even if the model output is
    unusable, at the cost of no semantic enrichment.
    """
    columns = []
    for col_idx in range(n_columns):
        header_path = []
        footnote_refs = set()
        for row in header_rows:
            if col_idx >= len(row):
                continue
            cell = row[col_idx].strip()
            if not cell:
                continue
            for mark in FOOTNOTE_MARK_RE.findall(cell):
                footnote_refs.add(mark)
            cleaned_cell = FOOTNOTE_MARK_RE.sub("", cell)
            cleaned_cell = DOLLAR_RE.sub("", cleaned_cell).strip()
            if cleaned_cell and (not header_path or header_path[-1] != cleaned_cell):
                header_path.append(cleaned_cell)

        role = "identifier" if col_idx == 0 else ("property" if header_path else "other")
        columns.append(
            {
                "column_index": col_idx,
                "role": role,
                "property_name": header_path[-1] if header_path and role == "property" else None,
                "context": header_path[0] if len(header_path) > 1 else None,
                "unit": None,
                "footnote_refs": sorted(footnote_refs),
                "header_path": header_path,
                "target_name": None,
                "target_type": None,
            }
        )
    return {
        "layout": DEFAULT_LAYOUT,
        "assay": dict(DEFAULT_ASSAY),
        "skip_extract": False,
        "columns": columns,
    }


class MissingCompoundRefsError(ValueError):
    """Raised when table schema interpretation is attempted without corefs."""


def interpret_table_schema(
    title,
    header_rows,
    footnotes=None,
    sample_rows=None,
    body_rows=None,
    compound_refs: list[str] | None = None,
    generate_fn: GenerateFn | None = None,
):
    """Interpret a table's header into a per-column semantic schema.

    Args:
        title: table title/caption string, or None/"".
        header_rows: list of rows (each a list of cell strings) already
            isolated as header rows (e.g. via count_header_rows).
        footnotes: dict of {marker: text}, a raw footnote string, or None.
        sample_rows: optional leading body rows for context when metric
            names live in data cells rather than headers; capped internally.
        body_rows: full body rows for compound-axis location (defaults to
            sample_rows / uncapped body when provided via sample_rows arg
            historically). Prefer passing full body separately.
        compound_refs: required chemical coreference IDs for this table.
        generate_fn: callable(messages, max_new_tokens=...) -> str;
            defaults to chemtables.gemma_client.default_generate.

    Returns:
        (schema, raw_output, source, axis) where schema is a dict
        {"layout": ..., "columns": [...], "skip_extract": bool}, raw_output
        is the model's raw text, source is "model" or "deterministic_backoff",
        and axis is the CompoundAxis used.
    """
    compound_coreferences = [
        str(ref).strip()
        for ref in (compound_refs or [])
        if ref is not None and str(ref).strip()
    ]
    if not compound_coreferences:
        raise MissingCompoundRefsError(
            "compound_refs required for table schema extraction"
        )

    if generate_fn is None:
        from chemtables.gemma_client import default_generate

        generate_fn = default_generate

    header_rows = normalize_rows(header_rows)
    footnotes = normalize_footnotes(footnotes)
    n_columns = max(len(row) for row in header_rows)

    # Full body for axis location; capped samples for the model prompt.
    body_for_axis = body_rows if body_rows is not None else sample_rows
    if body_for_axis:
        body_for_axis = normalize_rows(body_for_axis)
    else:
        body_for_axis = []

    sample_rows = select_sample_rows(sample_rows if sample_rows is not None else body_for_axis, n_columns)
    title, header_rows, footnotes, sample_rows = to_model_plain_text(
        title, header_rows, footnotes, sample_rows
    )
    body_plain = [[plain_cell(cell) for cell in row] for row in body_for_axis]

    title_unit_hints = extract_unit_hints(title, footnotes)
    axis = locate_compound_axis(header_rows, body_plain, compound_coreferences)

    raw = generate_fn(
        build_messages(
            title,
            header_rows,
            footnotes,
            title_unit_hints,
            sample_rows,
            DEFAULT_METRIC_UNITS,
            axis,
            compound_coreferences,
        ),
        max_new_tokens=MAX_NEW_TOKENS,
    )
    parsed = extract_json_object(raw)
    schema = validate_schema(parsed, n_columns)
    source = "model"
    if schema is None:
        schema = deterministic_backoff(header_rows, n_columns)
        source = "deterministic_backoff"
    else:
        apply_default_units(schema)

    apply_compound_axis(schema, axis)
    return schema, raw, source, axis


def detect_header_rows(
    grid: list,
    min_header_rows: int = 1,
    generate_fn: GenerateFn | None = None,
) -> int:
    """
    Detects how many leading rows in `grid` are header rows.
    Returns the number of header rows (>= min_header_rows).
    """
    count, _raw, _source = count_header_rows(grid, generate_fn=generate_fn)
    return max(count, min_header_rows)


def build_columns(header_rows: list, n_id_cols: int = 0,
                   id_col_names: list = None,
                   sep: str = " | ") -> list:
    """
    Builds flat column names from possibly multi-row headers.
    - header_rows: list of rows (already expanded grid rows) belonging to the header.
    - n_id_cols: number of leading columns treated as identifiers
      (e.g. group/metric) rather than data columns.
    - id_col_names: optional explicit names for the identifier columns.
    - sep: separator used to join multi-level header labels.
    """
    if not header_rows:
        raise ValueError("header_rows is empty")

    n_cols = len(header_rows[0])
    columns = []

    for col_idx in range(n_cols):
        levels = []
        for row in header_rows:
            val = row[col_idx].strip()
            if val and (not levels or levels[-1] != val):
                levels.append(val)
        columns.append(sep.join(levels) if levels else f"col_{col_idx}")

    if n_id_cols:
        default_ids = id_col_names or [f"id_{i}" for i in range(n_id_cols)]
        for i in range(min(n_id_cols, len(columns))):
            columns[i] = default_ids[i] if i < len(default_ids) else columns[i]

    return columns


def build_columns_from_schema(schema: dict, fallback_columns: list, sep: str = " | ") -> list:
    """Render column names from an interpret_table_schema() result,
    falling back to fallback_columns (raw joined header text) per-column
    when semantic info is missing.
    """
    by_index = {c["column_index"]: c for c in schema.get("columns", [])}
    names = []
    for idx, fallback in enumerate(fallback_columns):
        entry = by_index.get(idx)
        if not entry:
            names.append(fallback)
            continue

        role = entry.get("role")
        if role == "identifier":
            name = entry.get("property_name") or fallback
        elif role == "property":
            parts = [p for p in (entry.get("context"), entry.get("property_name")) if p]
            name = sep.join(parts) if parts else fallback
            if entry.get("unit"):
                name = f"{name} ({entry['unit']})"
        else:
            name = fallback

        if entry.get("footnote_refs"):
            name = f"{name} [{','.join(entry['footnote_refs'])}]"
        names.append(name or fallback)

    seen = {}
    deduped = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        deduped.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return deduped

def write_schema_outputs(
    grid: list,
    title,
    footnotes,
    output_dir: Path,
    generate_fn: GenerateFn,
    compound_refs: list[str] | None = None,
) -> dict:
    """Header detect + schema interpret + CSV/JSON write. Shared by CLI and pipeline."""
    if not compound_refs:
        raise MissingCompoundRefsError(
            "compound_refs required for table schema extraction"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_header = detect_header_rows(grid, generate_fn=generate_fn)
    header_rows_auto = grid[:n_header]
    data_rows_auto = grid[n_header:]
    columns_auto = build_columns(header_rows_auto)
    sample_rows_used = select_sample_rows(
        data_rows_auto, max(len(row) for row in header_rows_auto)
    )
    schema, raw_schema, schema_source, axis = interpret_table_schema(
        title,
        header_rows_auto,
        footnotes,
        sample_rows=data_rows_auto,
        body_rows=data_rows_auto,
        compound_refs=compound_refs,
        generate_fn=generate_fn,
    )
    print(f"Schema interpretation source: {schema_source}")
    print(
        f"Compound axis: layout={axis.layout} "
        f"column={axis.compound_column} skip_extract={axis.skip_extract}"
    )
    if axis.warnings:
        for warning in axis.warnings:
            print(f"Compound axis warning: {warning}")

    schema_path = output_dir / "table_schema.json"
    with open(schema_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "title": title,
                "footnotes": footnotes,
                "sample_rows": sample_rows_used,
                "compound_refs": list(compound_refs or []),
                "compound_axis": compound_axis_to_dict(axis),
                "source": schema_source,
                "raw_model_output": raw_schema,
                "schema": schema,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    columns_semantic = build_columns_from_schema(schema, columns_auto)
    print(f"Header rows detected: {n_header}")
    print(f"Columns (raw header): {columns_auto}")
    print(f"Columns (semantic): {columns_semantic}")

    # Body cells: same LaTeX→plain path as headers (CSV must not keep $...$ math).
    data_rows_plain = [
        [plain_cell(cell) for cell in row] for row in data_rows_auto
    ]
    df = pd.DataFrame(data_rows_plain, columns=columns_semantic)
    print(f"Extracted table:\n{df}")
    csv_path = output_dir / "extracted_table.csv"
    df.to_csv(csv_path, index=False)
    return {
        "n_header": n_header,
        "schema_path": schema_path,
        "csv_path": csv_path,
        "schema_source": schema_source,
        "compound_axis": axis,
        "skip_extract": bool(schema.get("skip_extract")),
    }


if __name__ == "__main__":
    from chemtables.gemma_client import GemmaSession

    output_dir = Path("output")
    grid = [
        [
            "Cell line $ ^{a} $",
            "IC $ _{50} $ 12g ( $ \\mu $M)",
            "IC $ _{50} $ 12h ( $ \\mu $M)",
            "Doxorubicin ( $ \\mu $M)",
        ],
        ["MCF10A", "n.d.", "n.d.", "n.d."],
        ["MCF-7", "4.4  $ \\pm $ 0.9", "4.3  $ \\pm $ 1.1", "1.1  $ \\pm $ 0.3"],
        ["MDA-MB-231", "11.1  $ \\pm $ 2.3", "9.2  $ \\pm $ 2.1", "n.d."],
    ]
    title = (
        "IC $ _{50} $ values of compounds 12g and 12h for breast cancer cells after 24 h."
    )
    footnotes = (
        " $ ^{a} $ Cells were treated at  $ \\sim $ 60 % confluence. n.d. = not determined. "
        "IC $ _{50} $ values are mean of three independent experiments (n = 3, mean  $ \\pm $ SEM)."
    )

    with GemmaSession() as session:
        write_schema_outputs(
            grid,
            title,
            footnotes,
            output_dir,
            session.generate,
            compound_refs=["12g", "12h"],
        )
