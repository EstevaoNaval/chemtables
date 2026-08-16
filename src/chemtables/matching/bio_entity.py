"""Match table target labels to bio_entities.db (protein / cell_line)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from chemtables.paths import STOPWORDS_PATH, default_bio_entities_db

DEFAULT_BIO_ENTITIES_DB = default_bio_entities_db()
HUMAN_TAXON_ID = 9606

_TYPE_MAP = {
    "protein": "single_protein",
    "cell_line": "cell_line",
}

PROTEIN_TARGET_TYPES = frozenset(
    {
        "single_protein",
        "protein_family",
        "protein_complex",
    }
)

_STRAIN_RE = re.compile(
    r"\b(?:ATCC|NCTC|DSMZ|H37Rv|strain|spp\.?)\b",
    re.IGNORECASE,
)
# Case-sensitive abbreviated binomial: S. aureus, P. aeruginosa.
_ABBREV_ORGANISM_RE = re.compile(r"\b[A-Z]\.\s*[a-z]+\b")

_LOOKUP_SQL = """
SELECT e.type
FROM alias AS a
JOIN entity AS e ON e.id = a.entity_id
WHERE a.name = ? COLLATE NOCASE
ORDER BY
  CASE
    WHEN e.taxon_id = ? THEN 0
    WHEN e.organism = 'Homo sapiens' COLLATE NOCASE THEN 1
    WHEN e.organism = 'Human' COLLATE NOCASE THEN 1
    WHEN e.organism LIKE 'Homo sapiens%' COLLATE NOCASE THEN 2
    ELSE 3
  END,
  e.id
LIMIT 1
"""

_STOPWORDS: frozenset[str] | None = None


def load_stopwords() -> frozenset[str]:
    """English stopwords for catalog lookup only. Lazy, casefolded."""
    global _STOPWORDS
    if _STOPWORDS is None:
        if STOPWORDS_PATH.is_file():
            _STOPWORDS = frozenset(
                line.strip().casefold()
                for line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        else:
            _STOPWORDS = frozenset()
    return _STOPWORDS


def _is_stopword(token: str) -> bool:
    return token.casefold() in load_stopwords()


def _content_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token and not _is_stopword(token)]


def extract_organism_label(name: str | None) -> str | None:
    """Binomial or strain span from a phrase, or None."""
    if not name:
        return None
    text = str(name).strip()
    if not text:
        return None
    abbrev = _ABBREV_ORGANISM_RE.search(text)
    if abbrev:
        return abbrev.group(0)
    if _STRAIN_RE.search(text):
        return text
    return None


def is_organism_label(name: str | None) -> bool:
    """True for abbreviated/full binomials and strain / culture-collection tokens."""
    return extract_organism_label(name) is not None


class BioEntityCatalog:
    """Case-insensitive alias lookup against an open SQLite connection."""

    def __init__(self, conn: sqlite3.Connection, *, owns_conn: bool = True) -> None:
        self.conn = conn
        self._owns_conn = owns_conn
        self._ensure_nocase_index()

    def _ensure_nocase_index(self) -> None:
        try:
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alias_name_nocase "
                "ON alias(name COLLATE NOCASE)"
            )
        except sqlite3.OperationalError:
            pass

    def close(self) -> None:
        if self._owns_conn:
            self.conn.close()

    def _lookup_type(self, query: str) -> str | None:
        row = self.conn.execute(_LOOKUP_SQL, (query, HUMAN_TAXON_ID)).fetchone()
        if not row:
            return None
        return _TYPE_MAP.get(row[0])

    def _lookup_token(self, token: str) -> str | None:
        if not token or _is_stopword(token):
            return None
        hit = self._lookup_type(token)
        if hit:
            return hit
        if len(token) > 1:
            stripped = token[1:]
            if stripped and not _is_stopword(stripped):
                return self._lookup_type(stripped)
        return None

    def resolve_match(self, name: str) -> tuple[str, str] | None:
        """Return (matched table token, type) on hit, else None.

        Drops English stopwords before lookup. Tries the remaining phrase,
        then whitespace tokens. On miss, retries without the first character.
        The returned name is the token that matched, not the full phrase.
        """
        text = (name or "").strip()
        if not text:
            return None
        tokens = _content_tokens(text)
        if not tokens:
            return None
        joined = " ".join(tokens)
        hit = self._lookup_token(joined)
        if hit:
            return joined, hit
        if len(tokens) == 1:
            return None
        for token in tokens:
            hit = self._lookup_token(token)
            if hit:
                return token, hit
        return None

    def resolve_type(self, name: str) -> str | None:
        """Map a label to single_protein / cell_line, or None on miss."""
        match = self.resolve_match(name)
        return match[1] if match else None


def open_catalog(path: Path | str | None) -> BioEntityCatalog | None:
    """Open bio_entities.db; None when path missing or file absent."""
    if path is None:
        return None
    db_path = Path(path)
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(db_path)
    return BioEntityCatalog(conn, owns_conn=True)


def finalize_target_type(
    name: str | None,
    target_type: str | None,
) -> str | None:
    """Keep protein/cell_line when set; otherwise organism if a name exists."""
    if not name or not str(name).strip():
        return None
    if target_type in PROTEIN_TARGET_TYPES or target_type == "cell_line":
        return target_type
    return "organism"


def resolve_target_label(
    name: str | None,
    catalog: BioEntityCatalog | None = None,
) -> dict[str, str | None]:
    """Organism span, else catalog protein/cell_line, else nulls."""
    text = str(name).strip() if name else ""
    if not text:
        return {"name": None, "type": None}
    extracted = extract_organism_label(text)
    if extracted:
        return {"name": extracted, "type": "organism"}
    if catalog is not None:
        match = catalog.resolve_match(text)
        if match:
            return {"name": match[0], "type": match[1]}
    return {"name": None, "type": None}


def classify_target_type(
    name: str | None,
    catalog: BioEntityCatalog | None = None,
) -> str | None:
    """Protein/cell_line from catalog; organism from regex; else None."""
    return resolve_target_label(name, catalog)["type"]
