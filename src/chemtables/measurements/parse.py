"""Deterministic parse of a single measurement cell."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Status = Literal["reported", "not_detected", "not_reported", "unparseable"]
Relation = Literal["=", "<", ">", "<=", ">=", "~"]

# Sentinel for unparseable / multi-value cells (callers should skip).
UNPARSEABLE = object()
# Backward-compatible alias.
AMBIGUOUS = UNPARSEABLE

_WS_RE = re.compile(r"\s+")
# Unicode / ASCII plus-minus variants seen in OCR.
_PM_RE = re.compile(
    r"^(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"(?:±|\+/-|\+/\-|[+]\s*/\s*[-]|\\pm)\s*"
    r"(?P<unc>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
_NUM_RE = re.compile(
    r"^(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
_RELATION_RE = re.compile(
    r"^(?P<rel><=|>=|≤|≥|<|>|~|≈)\s*"
    r"(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)
_RANGE_RE = re.compile(
    r"^(?P<a>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"(?:[-–—−]|to)\s*"
    r"(?P<b>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)
_NOT_DETECTED_RE = re.compile(
    r"^(?:nd|n\.?\s*d\.?|not\s*detected|no\s*detected\s*activity|"
    r"inactive|no\s*activity)$",
    re.IGNORECASE,
)
# Empty / dash / n.a. — no measurable result stated (skip at extract).
_NOT_REPORTED_RE = re.compile(
    r"^(?:[-–—−]+|\.|n\.?\s*a\.?|n/?a|not\s*reported)$",
    re.IGNORECASE,
)
# Qualitative "not determined" style — emit as text_value.
_QUALITATIVE_RE = re.compile(
    r"^(?:not\s*determined)$",
    re.IGNORECASE,
)
# Multiple numeric-like chunks separated by ; or / (multi-assay in one cell).
_MULTI_SEP_RE = re.compile(r"[;/]")
_NUM_TOKEN_RE = re.compile(r"[<>≤≥~≈]?\s*\d+(?:\.\d+)?")

_REL_NORMALIZE = {
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "≤": "<=",
    "≥": ">=",
    "~": "~",
    "≈": "~",
}


@dataclass(frozen=True)
class ParsedCell:
    value: float | None
    uncertainty: float | None
    status: Status
    relation: Relation | None = None
    text_value: str | None = None


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def looks_multi_value(text: str) -> bool:
    """True when cell likely holds more than one measurement."""
    parts = [p.strip() for p in _MULTI_SEP_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return False
    hits = sum(
        1
        for p in parts
        if _NUM_TOKEN_RE.search(p)
        or _NOT_DETECTED_RE.match(p)
        or _QUALITATIVE_RE.match(p)
    )
    return hits >= 2


def parse_measurement_cell(text: str) -> ParsedCell | object:
    """Parse one cell. Returns ParsedCell or UNPARSEABLE."""
    raw = text if text is not None else ""
    s = _norm(raw)
    if not s:
        return ParsedCell(
            value=None,
            uncertainty=None,
            status="not_reported",
            relation=None,
            text_value=None,
        )

    if looks_multi_value(s):
        return UNPARSEABLE

    if _NOT_DETECTED_RE.match(s):
        return ParsedCell(
            value=None,
            uncertainty=None,
            status="not_detected",
            relation=None,
            text_value=s,
        )

    if _QUALITATIVE_RE.match(s):
        return ParsedCell(
            value=None,
            uncertainty=None,
            status="not_detected",
            relation=None,
            text_value=s,
        )

    if _NOT_REPORTED_RE.match(s):
        return ParsedCell(
            value=None,
            uncertainty=None,
            status="not_reported",
            relation=None,
            text_value=None,
        )

    m = _RELATION_RE.match(s)
    if m:
        rel = _REL_NORMALIZE.get(m.group("rel"))
        if rel is None:
            return UNPARSEABLE
        return ParsedCell(
            value=float(m.group("value")),
            uncertainty=None,
            status="reported",
            relation=rel,  # type: ignore[arg-type]
            text_value=None,
        )

    m = _RANGE_RE.match(s)
    if m:
        return ParsedCell(
            value=None,
            uncertainty=None,
            status="reported",
            relation=None,
            text_value=s,
        )

    m = _PM_RE.match(s)
    if m:
        return ParsedCell(
            value=float(m.group("value")),
            uncertainty=float(m.group("unc")),
            status="reported",
            relation="=",
            text_value=None,
        )

    m = _NUM_RE.match(s)
    if m:
        return ParsedCell(
            value=float(m.group("value")),
            uncertainty=None,
            status="reported",
            relation="=",
            text_value=None,
        )

    return UNPARSEABLE
