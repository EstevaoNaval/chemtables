#!/usr/bin/env python3
"""Stage 4: extract bioactivity records from flattened CSV + schema."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from chemtables.matching.bio_entity import (
    DEFAULT_BIO_ENTITIES_DB,
    BioEntityCatalog,
    extract_organism_label,
    open_catalog,
    resolve_target_label,
)
from chemtables.matching.compound import build_coref_set, match_compound_id, normalize_row_label
from chemtables.measurements.parse import UNPARSEABLE, ParsedCell, parse_measurement_cell
from chemtables.schema.interpret import plain_cell, strip_reporting_metadata

ALLOWED_METRICS = {
    "IC50",
    "EC50",
    "CC50",
    "MIC",
    "MBC",
    "MFC",
    "SI",
    "ED50",
    "EE50",
    "MBC/MIC",
}

UNITLESS_METRICS = {"SI", "MBC/MIC"}

# Metrics where a per-measurement biological target is meaningful.
BIO_TARGET_METRICS = frozenset(
    {"EC50", "IC50", "CC50", "MIC", "MBC", "MFC", "ED50", "EE50"}
)

DEFAULT_METRIC_UNITS = {
    "IC50": "μM",
    "EC50": "μM",
    "CC50": "μM",
    "MIC": "μg/mL",
    "MBC": "μg/mL",
    "MFC": "μg/mL",
    "SI": "unitless",
    "ED50": "mg/kg",
    "EE50": "mg/kg",
    "MBC/MIC": "unitless",
}

VALID_RELATIONS = frozenset({"=", "<", ">", "<=", ">=", "~"})
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
VALID_ASSAY_TYPES = frozenset({"B", "F", "A", "T", "P", "U"})

# Longer tokens first so MBC/MIC wins over MBC or MIC.
_METRIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bMBC\s*/\s*MIC\b", re.IGNORECASE), "MBC/MIC"),
    (
        re.compile(
            r"\bCC(?:\s*_\s*\{\{50\}\}|\s*_\s*\{50\}|\s*_?\s*50|₅₀)(?!\w)",
            re.IGNORECASE,
        ),
        "CC50",
    ),
    (
        re.compile(
            r"\bIC(?:\s*_\s*\{\{50\}\}|\s*_\s*\{50\}|\s*_?\s*50|₅₀)(?!\w)",
            re.IGNORECASE,
        ),
        "IC50",
    ),
    (
        re.compile(
            r"\bEC(?:\s*_\s*\{\{50\}\}|\s*_\s*\{50\}|\s*_?\s*50|₅₀)(?!\w)",
            re.IGNORECASE,
        ),
        "EC50",
    ),
    (
        re.compile(
            r"\bED(?:\s*_\s*\{\{50\}\}|\s*_\s*\{50\}|\s*_?\s*50|₅₀)(?!\w)",
            re.IGNORECASE,
        ),
        "ED50",
    ),
    (
        re.compile(
            r"\bEE(?:\s*_\s*\{\{50\}\}|\s*_\s*\{50\}|\s*_?\s*50|₅₀)(?!\w)",
            re.IGNORECASE,
        ),
        "EE50",
    ),
    (re.compile(r"\bMBC\b", re.IGNORECASE), "MBC"),
    (re.compile(r"\bMFC\b", re.IGNORECASE), "MFC"),
    (re.compile(r"\bMIC\b", re.IGNORECASE), "MIC"),
    (re.compile(r"\bSI\b", re.IGNORECASE), "SI"),
]

_UNIT_TOKEN_RE = re.compile(
    r"(?:"
    r"\([^)]*\)|"
    r"\b(?:μg|ug|µg)\s*/\s*mL\b|"
    r"\bmg\s*/\s*(?:L|kg|mL)\b|"
    r"\b(?:μM|uM|µM|nM|mM|M)\b|"
    r"\bunitless\b|"
    r"\\+mu\b|"
    r"\bmu\b"
    r")",
    re.IGNORECASE,
)
_FOOTNOTE_MARK_RE = re.compile(
    r"(?:\^[{\s]*[a-zA-Z0-9]+[}\s]*|\[[a-zA-Z0-9]+]|[\^*\u2020\u2021])"
)
_WS_RE = re.compile(r"\s+")


def canonicalize_metric(name: str | None) -> str | None:
    """Map schema property_name to an allowed metric_type, or None."""
    if not name:
        return None
    raw = str(name).strip()
    if not raw:
        return None
    if raw in ALLOWED_METRICS:
        return raw

    compact = re.sub(r"[\s_]+", "", raw)
    compact = compact.replace("{{50}}", "50").replace("{50}", "50").replace("₅₀", "50")
    upper = compact.upper()
    aliases = {
        "IC50": "IC50",
        "EC50": "EC50",
        "CC50": "CC50",
        "MIC": "MIC",
        "MBC": "MBC",
        "MFC": "MFC",
        "SI": "SI",
        "ED50": "ED50",
        "EE50": "EE50",
        "MBC/MIC": "MBC/MIC",
        "MBCMIC": "MBC/MIC",
    }
    if upper in aliases:
        return aliases[upper]

    for pattern, metric in _METRIC_PATTERNS:
        if pattern.search(raw):
            return metric
    return None


def resolve_unit(metric_type: str, schema_unit: str | None) -> str | None:
    if schema_unit:
        unit = str(schema_unit).strip()
        if unit:
            return unit
    if metric_type in UNITLESS_METRICS:
        return "unitless"
    return DEFAULT_METRIC_UNITS.get(metric_type)


def _strip_metric_tokens(text: str) -> str:
    """Replace IC50/MIC/… tokens with spaces. Leave the rest."""
    s = text or ""
    for pattern, _metric in _METRIC_PATTERNS:
        s = pattern.sub(" ", s)
    return s


def _strip_unit_tokens(text: str) -> str:
    """Replace unit tokens (μM, μg/mL, parenthetical units, …) with spaces."""
    return _UNIT_TOKEN_RE.sub(" ", text or "")


def _header_bio_residual(text: str) -> str:
    """Header phrase after metric, unit, footnote, and reporting crumbs."""
    s = plain_cell(text) if text else ""
    if not s:
        return ""
    s = _strip_metric_tokens(s)
    s = _strip_unit_tokens(s)
    s = _FOOTNOTE_MARK_RE.sub(" ", s)
    s = re.sub(r"[{}\\]", " ", s)
    s = re.sub(r"[|,;:]+", " ", s)
    s = _WS_RE.sub(" ", s).strip(" -_/.")
    cleaned = strip_reporting_metadata(s)
    return cleaned or ""


def _header_path_residuals(header_path: list | None) -> list[str]:
    """Bio residuals from each header_path step, unique."""
    residuals: list[str] = []
    seen: set[str] = set()
    for step in header_path or []:
        if step is None:
            continue
        text = str(step).strip()
        if not text:
            continue
        residual = _header_bio_residual(text)
        if residual and residual not in seen:
            seen.add(residual)
            residuals.append(residual)
    return residuals


def extract_biological_system_residual(
    property_name: str | None,
    header_path: list | None,
) -> str | None:
    """Peel metric/unit from property_name / header_path; keep bio residual."""
    if property_name:
        residual = _header_bio_residual(str(property_name))
        if residual:
            return residual
    residuals = _header_path_residuals(header_path)
    return residuals[0] if residuals else None


def _target_candidates(entry: dict) -> list[str]:
    """header_path residuals, then context, then target_name. Unique, order kept."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        if not raw:
            return
        text = str(raw).strip()
        if not text or text in seen:
            return
        seen.add(text)
        candidates.append(text)

    for residual in _header_path_residuals(entry.get("header_path")):
        add(residual)
    context = entry.get("context")
    if context:
        add(_header_bio_residual(str(context)))
    name = entry.get("target_name")
    if name is not None:
        add(str(name).strip() or None)
    return candidates


def resolve_target_from_schema(
    metric_type: str,
    entry: dict,
    catalog: BioEntityCatalog | None = None,
) -> dict[str, Any]:
    """Build target dict from schema column fields + header/context lookup."""
    if metric_type not in BIO_TARGET_METRICS:
        return {"name": None, "type": None}

    schema_name = entry.get("target_name")
    if schema_name is not None:
        schema_name = str(schema_name).strip() or None

    extracted = extract_organism_label(schema_name)
    if extracted:
        return {"name": extracted, "type": "organism"}

    candidates = _target_candidates(entry)
    for cand in candidates:
        extracted = extract_organism_label(cand)
        if extracted:
            return {"name": extracted, "type": "organism"}

    if catalog is not None:
        for cand in candidates:
            match = catalog.resolve_match(cand)
            if match:
                return {"name": match[0], "type": match[1]}

    return {"name": None, "type": None}


def resolve_assay(schema: dict, title: Any) -> dict[str, Any]:
    """Assay from schema top-level; fallback description = plain title."""
    raw = schema.get("assay") if isinstance(schema.get("assay"), dict) else {}
    description = raw.get("description")
    if description is not None:
        description = str(description).strip() or None
    if not description and title:
        description = plain_cell(title) or None
    assay_type = raw.get("type")
    if assay_type is not None:
        assay_type = str(assay_type).strip().upper() or None
        if assay_type not in VALID_ASSAY_TYPES:
            assay_type = None
    return {"description": description, "type": assay_type}


def build_column_map(
    schema: dict,
    catalog: BioEntityCatalog | None = None,
) -> dict[int, dict[str, Any]]:
    """Map column_index → role info for identifier / measurement columns."""
    column_map: dict[int, dict[str, Any]] = {}
    for entry in schema.get("columns") or []:
        try:
            idx = int(entry["column_index"])
        except (KeyError, TypeError, ValueError):
            continue
        role = entry.get("role")
        if role == "identifier":
            column_map[idx] = {"role": "identifier"}
            continue
        if role != "property":
            continue
        property_name = entry.get("property_name")
        metric = canonicalize_metric(property_name)
        if not metric or metric not in ALLOWED_METRICS:
            continue
        unit = resolve_unit(metric, entry.get("unit"))
        target = resolve_target_from_schema(metric, entry, catalog)
        column_map[idx] = {
            "role": "measurement",
            "metric_type": metric,
            "unit": unit,
            "target": target,
            "schema_entry": entry,
        }
    return column_map


def find_identifier_index(column_map: dict[int, dict[str, Any]], n_cols: int) -> int | None:
    for idx, info in sorted(column_map.items()):
        if info.get("role") == "identifier":
            return idx
    return 0 if n_cols else None


def _activity_comment(parsed: ParsedCell) -> str | None:
    if parsed.uncertainty is None:
        return None
    return f"± {parsed.uncertainty}"


def _parsed_to_activity(
    parsed: ParsedCell,
    *,
    metric_type: str,
    unit: str | None,
) -> dict[str, Any] | None:
    """Map ParsedCell → activity object, or None if cell should be skipped."""
    if parsed.status == "not_reported":
        return None

    relation = parsed.relation
    value = parsed.value
    text_value = parsed.text_value

    if parsed.status == "not_detected":
        if not text_value:
            return None
        relation = relation or "="
        activity = {
            "type": metric_type,
            "relation": relation,
            "value": None,
            "units": unit,
            "text_value": text_value,
            "comment": _activity_comment(parsed),
        }
    elif value is not None:
        activity = {
            "type": metric_type,
            "relation": relation or "=",
            "value": value,
            "units": unit,
            "text_value": None,
            "comment": _activity_comment(parsed),
        }
    elif text_value:
        activity = {
            "type": metric_type,
            "relation": relation or "=",
            "value": None,
            "units": unit,
            "text_value": text_value,
            "comment": _activity_comment(parsed),
        }
    else:
        return None

    if metric_type in UNITLESS_METRICS and activity.get("units") not in (None, "unitless"):
        activity["units"] = "unitless"
    elif value is not None and activity.get("units") is None and metric_type not in UNITLESS_METRICS:
        # Numeric concentration without units — keep null; validation may skip.
        pass

    return activity


def validate_record(record: dict) -> bool:
    compound = record.get("compound")
    if not isinstance(compound, dict) or not compound.get("name"):
        return False
    activity = record.get("activity")
    if not isinstance(activity, dict):
        return False
    if activity.get("type") not in ALLOWED_METRICS:
        return False
    if activity.get("relation") not in VALID_RELATIONS:
        return False
    if activity.get("value") is None and not activity.get("text_value"):
        return False
    if activity.get("value") is not None and activity.get("units") is None:
        if activity["type"] not in UNITLESS_METRICS:
            return False
    target = record.get("target")
    if not isinstance(target, dict):
        return False
    ttype = target.get("type")
    if ttype is not None and ttype not in VALID_TARGET_TYPES:
        return False
    assay = record.get("assay")
    if not isinstance(assay, dict):
        return False
    atype = assay.get("type")
    if atype is not None and atype not in VALID_ASSAY_TYPES:
        return False
    return True


def _schema_layout(schema: dict) -> str | None:
    if schema.get("skip_extract"):
        return None
    layout = schema.get("layout")
    if layout in {"compounds_in_rows", "compounds_in_columns"}:
        return layout
    if layout is None:
        return None
    return "compounds_in_rows"


def _column_haystack(entry: dict) -> str:
    """Join schema fields used to find a compound ref in a column header."""
    parts: list[str] = []
    context = entry.get("context")
    if context:
        parts.append(str(context))
    for step in entry.get("header_path") or []:
        if step is None:
            continue
        text = str(step).strip()
        if text:
            parts.append(text)
    property_name = entry.get("property_name")
    if property_name:
        parts.append(str(property_name))
    return " ".join(parts)


def _schema_entries_by_index(schema: dict) -> dict[int, dict]:
    by_idx: dict[int, dict] = {}
    for entry in schema.get("columns") or []:
        try:
            idx = int(entry["column_index"])
        except (KeyError, TypeError, ValueError):
            continue
        by_idx[idx] = entry
    return by_idx


def _bind_columns_to_compounds(
    column_map: dict[int, dict[str, Any]],
    schema_entries: dict[int, dict],
    coref_map: dict[str, str],
) -> tuple[dict[int, str], list[dict]]:
    """Map measurement column_index → compound name; list unmapped columns."""
    bound: dict[int, str] = {}
    unmapped_columns: list[dict] = []
    for col_idx, info in sorted(column_map.items()):
        if info.get("role") != "measurement":
            continue
        entry = schema_entries.get(col_idx) or {}
        haystack = _column_haystack(entry)
        compound_name = match_compound_id(haystack, coref_map)
        if compound_name is None:
            unmapped_columns.append(
                {
                    "column_index": col_idx,
                    "haystack": haystack,
                    "reason": "not_in_compound_refs",
                }
            )
            continue
        bound[col_idx] = compound_name
    return bound, unmapped_columns


def _make_record(
    *,
    compound_name: str,
    activity: dict,
    target: dict,
    assay: dict,
) -> dict | None:
    record = {
        "compound": {"name": compound_name, "reference_id": None},
        "activity": activity,
        "target": target,
        "assay": assay,
    }
    if not validate_record(record):
        return None
    return record


def _append_cell_record(
    *,
    records: list[dict],
    cell_text: str,
    compound_name: str,
    metric_type: str,
    unit: str | None,
    target: dict,
    assay: dict,
) -> None:
    parsed = parse_measurement_cell(cell_text)
    if parsed is UNPARSEABLE:
        return
    assert isinstance(parsed, ParsedCell)
    activity = _parsed_to_activity(parsed, metric_type=metric_type, unit=unit)
    if activity is None:
        return
    record = _make_record(
        compound_name=compound_name,
        activity=activity,
        target=target,
        assay=assay,
    )
    if record is not None:
        records.append(record)


def load_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with open(csv_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    header = rows[0]
    body = rows[1:]
    return header, body


def extract_table_measurements(
    csv_path: Path,
    schema_payload: dict,
    compound_refs: list[str],
    table_id: str,
    *,
    bio_entities_db: Path | None = DEFAULT_BIO_ENTITIES_DB,
    catalog: BioEntityCatalog | None = None,
) -> dict:
    """Build table-level bioactivity envelope from CSV + schema + corefs."""
    schema = schema_payload.get("schema") or schema_payload
    if "columns" not in schema:
        raise ValueError("schema payload missing columns")

    context_warnings: list[str] = []
    axis_meta = schema_payload.get("compound_axis") or {}
    if isinstance(axis_meta, dict):
        for warning in axis_meta.get("warnings") or []:
            if warning:
                context_warnings.append(str(warning))

    if not compound_refs:
        context_warnings.append("compound_refs empty; skipping measurement extract")
        return {
            "_context_warnings": context_warnings,
            "table_id": table_id,
            "layout": None,
            "unmapped_rows": [],
            "unmapped_columns": [],
            "records": [],
        }

    layout = _schema_layout(schema)
    if schema.get("skip_extract") or layout is None:
        context_warnings.append(
            "skip_extract: no reliable compound axis; no measurement records"
        )
        return {
            "_context_warnings": context_warnings,
            "table_id": table_id,
            "layout": None,
            "unmapped_rows": [],
            "unmapped_columns": [],
            "records": [],
        }

    owned_catalog = False
    if catalog is None and bio_entities_db is not None:
        catalog = open_catalog(bio_entities_db)
        owned_catalog = catalog is not None

    try:
        return _extract_table_measurements(
            csv_path,
            schema_payload,
            schema,
            compound_refs,
            table_id,
            layout,
            context_warnings,
            catalog,
        )
    finally:
        if owned_catalog and catalog is not None:
            catalog.close()


def _extract_table_measurements(
    csv_path: Path,
    schema_payload: dict,
    schema: dict,
    compound_refs: list[str],
    table_id: str,
    layout: str,
    context_warnings: list[str],
    catalog: BioEntityCatalog | None,
) -> dict:
    _, body = load_csv_rows(csv_path)
    title = schema_payload.get("title")
    assay = resolve_assay(schema, title)
    column_map = build_column_map(schema, catalog)
    coref_map = build_coref_set(compound_refs)
    schema_entries = _schema_entries_by_index(schema)

    records: list[dict] = []
    unmapped_rows: list[dict] = []
    unmapped_columns: list[dict] = []

    column_compounds: dict[int, str] | None = None
    if layout == "compounds_in_columns":
        column_compounds, unmapped_columns = _bind_columns_to_compounds(
            column_map, schema_entries, coref_map
        )

    for row_idx, row in enumerate(body):
        plain_row = [plain_cell(cell) for cell in row]
        n_cols = len(plain_row)
        id_idx = find_identifier_index(column_map, n_cols)
        if id_idx is None or id_idx >= n_cols:
            unmapped_rows.append(
                {
                    "row_index": row_idx,
                    "raw": None,
                    "cleaned": None,
                    "reason": "no_identifier_column",
                }
            )
            continue

        raw_label = plain_row[id_idx]
        cleaned = normalize_row_label(raw_label)

        if layout == "compounds_in_columns":
            assert column_compounds is not None
            target = resolve_target_label(cleaned, catalog)
            for col_idx, compound_name in sorted(column_compounds.items()):
                if col_idx >= n_cols:
                    continue
                info = column_map[col_idx]
                _append_cell_record(
                    records=records,
                    cell_text=plain_row[col_idx],
                    compound_name=compound_name,
                    metric_type=info["metric_type"],
                    unit=info["unit"],
                    target=target,
                    assay=assay,
                )
            continue

        compound_name = match_compound_id(raw_label, coref_map)
        if compound_name is None:
            unmapped_rows.append(
                {
                    "row_index": row_idx,
                    "raw": raw_label,
                    "cleaned": cleaned,
                    "reason": "not_in_compound_refs",
                }
            )
            continue

        for col_idx, info in sorted(column_map.items()):
            if info.get("role") != "measurement":
                continue
            if col_idx >= n_cols:
                continue
            _append_cell_record(
                records=records,
                cell_text=plain_row[col_idx],
                compound_name=compound_name,
                metric_type=info["metric_type"],
                unit=info["unit"],
                target=info["target"],
                assay=assay,
            )

    return {
        "_context_warnings": context_warnings,
        "table_id": table_id,
        "layout": layout,
        "unmapped_rows": unmapped_rows,
        "unmapped_columns": unmapped_columns,
        "records": records,
    }


def write_measurement_outputs(
    output_dir: Path,
    compound_refs: list[str],
    table_id: str | None = None,
    bio_entities_db: Path | None = DEFAULT_BIO_ENTITIES_DB,
) -> Path:
    output_dir = Path(output_dir)
    csv_path = output_dir / "extracted_table.csv"
    schema_path = output_dir / "table_schema.json"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)

    with open(schema_path, encoding="utf-8") as handle:
        schema_payload = json.load(handle)

    tid = table_id or output_dir.name
    result = extract_table_measurements(
        csv_path,
        schema_payload,
        compound_refs,
        tid,
        bio_entities_db=bio_entities_db,
    )
    out_path = output_dir / "measurements.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract bioactivity records from extracted_table.csv + table_schema.json"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing extracted_table.csv and table_schema.json",
    )
    parser.add_argument(
        "--compound-refs",
        type=Path,
        required=True,
        help="JSON file with a list of compound coreference strings",
    )
    parser.add_argument(
        "--table-id",
        default=None,
        help="Optional table_id (default: output directory name)",
    )
    parser.add_argument(
        "--bio-entities",
        type=Path,
        default=DEFAULT_BIO_ENTITIES_DB,
        help=(
            "SQLite bio_entities.db (default: ./data/bio_entities.db, "
            "override via CHEMTABLES_BIO_ENTITIES_DB)"
        ),
    )
    args = parser.parse_args()

    with open(args.compound_refs, encoding="utf-8") as handle:
        compound_refs = json.load(handle)
    if not isinstance(compound_refs, list) or not all(
        isinstance(item, str) for item in compound_refs
    ):
        raise SystemExit("compound-refs must be a JSON array of strings")

    path = write_measurement_outputs(
        args.output_dir,
        compound_refs,
        args.table_id,
        bio_entities_db=args.bio_entities,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
