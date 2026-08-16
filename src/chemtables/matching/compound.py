"""Match table cells to compound coreferences; locate compound axis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Superficial footnote junk only — never peel letters that belong to IDs (9b, 10a).
_MARKER_RE = (
    r"(?:[\^*\u2020\u2021\u00a7\u00b6]+|"
    r"\^[{\s]*[a-zA-Z0-9]+[}\s]*|"
    r"\[[a-zA-Z0-9]+\])"
)
_FOOTNOTE_LEAD_RE = re.compile(rf"^{_MARKER_RE}")
_FOOTNOTE_TRAIL_RE = re.compile(rf"{_MARKER_RE}$")
_WS_RE = re.compile(r"\s+")
_PURE_NUMBER_RE = re.compile(
    r"^[<>≤≥~≈]?\s*[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*$"
)
_VALUE_LIKE_RE = re.compile(
    r"^(?:nd|n\.?\s*d\.?|n\.?\s*a\.?|n/?a|na|[-–—−]+|\.)$",
    re.IGNORECASE,
)


def normalize_row_label(cell: str) -> str:
    """Strip whitespace and superficial footnote markers from an ID cell."""
    text = _WS_RE.sub(" ", (cell or "").strip())
    if not text:
        return ""
    for _ in range(3):
        nxt = _FOOTNOTE_LEAD_RE.sub("", text)
        nxt = _FOOTNOTE_TRAIL_RE.sub("", nxt).strip()
        nxt = _WS_RE.sub(" ", nxt).strip()
        if nxt == text:
            break
        text = nxt
    return text


def build_coref_set(compound_refs: list[str]) -> dict[str, str]:
    """Map normalized label → canonical coref string from the supplied list."""
    mapping: dict[str, str] = {}
    for ref in compound_refs:
        if ref is None:
            continue
        canonical = str(ref).strip()
        if not canonical:
            continue
        key = normalize_row_label(canonical)
        if key:
            mapping[key] = canonical
    return mapping


def _ref_contained(haystack: str, needle: str) -> bool:
    """True if needle appears in haystack with alphanumeric boundaries."""
    if not needle:
        return False
    pattern = (
        rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
    )
    return re.search(pattern, haystack, flags=re.IGNORECASE) is not None


def match_compound_id(cell: str, coref_map: dict[str, str]) -> str | None:
    """Longest bounded containment match of cleaned text against coref map.

    Exact and substring hits both qualify when the ref is bounded by
    non-alphanumeric edges (or string ends). Among hits, the longest
    normalized key wins; equal-length conflicts return None.
    """
    cleaned = normalize_row_label(cell)
    if not cleaned or not coref_map:
        return None

    # Exact first (case-sensitive on normalized keys as built).
    exact = coref_map.get(cleaned)
    if exact is not None:
        return exact

    best_len = -1
    best_canonical: str | None = None
    tied = False
    for key, canonical in coref_map.items():
        if not _ref_contained(cleaned, key):
            continue
        key_len = len(key)
        if key_len > best_len:
            best_len = key_len
            best_canonical = canonical
            tied = False
        elif key_len == best_len:
            if best_canonical is not None and canonical != best_canonical:
                tied = True

    if tied:
        return None
    return best_canonical


@dataclass
class CompoundAxis:
    layout: str | None  # compounds_in_rows | compounds_in_columns | None
    compound_column: int | None
    header_hits: dict[int, int] = field(default_factory=dict)
    body_hits: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skip_extract: bool = False


def _n_columns(header_rows: list, body_rows: list) -> int:
    widths = [len(row) for row in (header_rows or []) if row]
    widths.extend(len(row) for row in (body_rows or []) if row)
    return max(widths) if widths else 0


def _score_columns(
    rows: list,
    n_cols: int,
    coref_map: dict[str, str],
) -> dict[int, int]:
    hits = {i: 0 for i in range(n_cols)}
    if not rows or not coref_map:
        return hits
    for row in rows:
        if not row:
            continue
        for col_idx in range(min(n_cols, len(row))):
            cell = row[col_idx]
            if cell is None or not str(cell).strip():
                continue
            if match_compound_id(str(cell), coref_map) is not None:
                hits[col_idx] += 1
    return hits


def _best_column(hits: dict[int, int]) -> tuple[int | None, int]:
    """Return (column_index, score) for the single best column; None if all zero."""
    best_col: int | None = None
    best_score = 0
    for col_idx, score in sorted(hits.items()):
        if score > best_score:
            best_score = score
            best_col = col_idx
    if best_score <= 0:
        return None, 0
    return best_col, best_score


def _is_id_like_label(cell: str) -> bool:
    text = normalize_row_label(cell)
    if not text:
        return False
    if len(text) > 12:
        return False
    if _PURE_NUMBER_RE.match(text):
        return False
    if _VALUE_LIKE_RE.match(text):
        return False
    # Reject obvious metric tokens that appear in body (MIC/MFC/…).
    if text.upper() in {"MIC", "MFC", "MBC", "SI", "IC50", "EC50", "CC50", "ED50", "EE50"}:
        return False
    return True


def col0_looks_id_like(body_rows: list) -> bool:
    """True when ≥50% of non-empty body col0 cells look like short ID labels."""
    if not body_rows:
        return False
    non_empty = 0
    id_like = 0
    for row in body_rows:
        if not row:
            continue
        cell = row[0] if len(row) > 0 else ""
        if cell is None or not str(cell).strip():
            continue
        non_empty += 1
        if _is_id_like_label(str(cell)):
            id_like += 1
    if non_empty == 0:
        return False
    return (id_like / non_empty) >= 0.5


def locate_compound_axis(
    header_rows: list,
    body_rows: list,
    compound_refs: list[str],
) -> CompoundAxis:
    """Locate where compound corefs sit; decide layout + single axis column."""
    coref_map = build_coref_set(compound_refs)
    n_cols = _n_columns(header_rows, body_rows)
    if n_cols == 0:
        return CompoundAxis(
            layout=None,
            compound_column=None,
            warnings=["empty table grid; cannot locate compound axis"],
            skip_extract=True,
        )

    header_hits = _score_columns(header_rows or [], n_cols, coref_map)
    body_hits = _score_columns(body_rows or [], n_cols, coref_map)
    header_col, header_best = _best_column(header_hits)
    body_col, body_best = _best_column(body_hits)

    if body_best > header_best:
        return CompoundAxis(
            layout="compounds_in_rows",
            compound_column=body_col,
            header_hits=header_hits,
            body_hits=body_hits,
            skip_extract=False,
        )

    if header_best > body_best:
        return CompoundAxis(
            layout="compounds_in_columns",
            compound_column=header_col,
            header_hits=header_hits,
            body_hits=body_hits,
            skip_extract=False,
        )

    # Tie or both zero → no-hit / ambiguous handling.
    warnings: list[str] = []
    if body_best == 0 and header_best == 0:
        warnings.append(
            "no compound coreference hits in header or body; "
            "cannot locate compound axis from refs"
        )
    else:
        warnings.append(
            f"tied header/body coref scores (header={header_best}, "
            f"body={body_best}); falling back to no-hit handling"
        )

    if col0_looks_id_like(body_rows or []):
        warnings.append(
            "col0 looks ID-like; defaulting to compounds_in_rows with compound_column=0"
        )
        return CompoundAxis(
            layout="compounds_in_rows",
            compound_column=0,
            header_hits=header_hits,
            body_hits=body_hits,
            warnings=warnings,
            skip_extract=False,
        )

    warnings.append("skip measurement extract: no reliable compound axis")
    return CompoundAxis(
        layout=None,
        compound_column=None,
        header_hits=header_hits,
        body_hits=body_hits,
        warnings=warnings,
        skip_extract=True,
    )


def compound_axis_to_dict(axis: CompoundAxis) -> dict:
    """JSON-serializable form for prompts / schema payloads."""
    return {
        "layout": axis.layout,
        "compound_column": axis.compound_column,
        "header_hits": {str(k): v for k, v in sorted(axis.header_hits.items())},
        "body_hits": {str(k): v for k, v in sorted(axis.body_hits.items())},
        "warnings": list(axis.warnings),
        "skip_extract": axis.skip_extract,
    }
