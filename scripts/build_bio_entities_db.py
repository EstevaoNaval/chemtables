"""Build data/bio_entities.db from UniProtKB reviewed TSV + Cellosaurus.

Requires an editable install of chemtables (`pip install -e .`). Place
source files first:
    data/sources/uniprotkb_reviewed.tsv
    data/sources/cellosaurus.txt
(see README.md for where to obtain them). Run with:
    python scripts/build_bio_entities_db.py
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

from chemtables.paths import BIO_ENTITIES_SCHEMA_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "bio_entities.db"
SCHEMA_PATH = BIO_ENTITIES_SCHEMA_PATH
UNIPROT_PATH = DATA_DIR / "sources" / "uniprotkb_reviewed.tsv"
CELLOSAURUS_PATH = DATA_DIR / "sources" / "cellosaurus.txt"

BATCH_SIZE = 8000
_EC_RE = re.compile(r"^EC\s+", re.IGNORECASE)
_OX_RE = re.compile(r"NCBI_TaxID=(\d+);\s*!\s*(.+)$")
_OX_TAXON_ONLY_RE = re.compile(r"NCBI_TaxID=(\d+)")


def recommended_name(protein_name: str) -> str:
    idx = protein_name.find(" (")
    if idx == -1:
        return protein_name.strip()
    return protein_name[:idx].strip()


def parenthetical_aliases(protein_name: str) -> list[str]:
    names: list[str] = []
    i = 0
    n = len(protein_name)
    while i < n:
        if protein_name[i] == "(":
            depth = 1
            j = i + 1
            while j < n and depth:
                if protein_name[j] == "(":
                    depth += 1
                elif protein_name[j] == ")":
                    depth -= 1
                j += 1
            inner = protein_name[i + 1 : j - 1].strip()
            if inner and not _EC_RE.match(inner):
                names.append(inner)
            i = j
        else:
            i += 1
    return names


def split_semicolon(text: str) -> list[str]:
    return [part.strip() for part in text.split(";") if part.strip()]


def split_space(text: str) -> list[str]:
    return [part for part in text.split() if part]


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def unique_aliases(*groups: str | None | list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        if group is None:
            continue
        values = group if isinstance(group, list) else [group]
        for raw in values:
            name = (raw or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def parse_ox(value: str) -> tuple[int | None, str | None]:
    match = _OX_RE.search(value)
    if match:
        return int(match.group(1)), match.group(2).strip()
    taxon = _OX_TAXON_ONLY_RE.search(value)
    if taxon:
        return int(taxon.group(1)), None
    return None, None


class Loader:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.cur = conn.cursor()
        self.next_id = 1
        self.entity_rows: list[tuple] = []
        self.alias_rows: list[tuple[str, int]] = []
        self.seen_accessions: set[str] = set()
        self.skipped_no_key = 0
        self.skipped_dup_accession = 0
        self.n_protein = 0
        self.n_cell_line = 0
        self.n_alias = 0

    def add_entity(
        self,
        entity_type: str,
        preferred_name: str,
        accession: str,
        organism: str | None,
        taxon_id: int | None,
        aliases: list[str],
    ) -> None:
        preferred_name = (preferred_name or "").strip()
        accession = (accession or "").strip()
        if not preferred_name or not accession:
            self.skipped_no_key += 1
            return
        if accession in self.seen_accessions:
            self.skipped_dup_accession += 1
            return
        self.seen_accessions.add(accession)
        organism = (organism or "").strip() or None
        entity_id = self.next_id
        self.next_id += 1
        self.entity_rows.append(
            (entity_id, entity_type, preferred_name, accession, organism, taxon_id)
        )
        if entity_type == "protein":
            self.n_protein += 1
        else:
            self.n_cell_line += 1
        for name in aliases:
            self.alias_rows.append((name, entity_id))
            self.n_alias += 1
        if len(self.entity_rows) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if self.entity_rows:
            self.cur.executemany(
                "INSERT INTO entity "
                "(id, type, preferred_name, accession, organism, taxon_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                self.entity_rows,
            )
            self.entity_rows.clear()
        if self.alias_rows:
            self.cur.executemany(
                "INSERT INTO alias (name, entity_id) VALUES (?, ?)",
                self.alias_rows,
            )
            self.alias_rows.clear()


def load_uniprot(loader: Loader, path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            protein_name = (row.get("protein_name") or "").strip()
            accession = (row.get("accession") or "").strip()
            if not protein_name or not accession:
                loader.skipped_no_key += 1
                continue
            preferred = recommended_name(protein_name)
            gene_primary = (row.get("gene_primary") or "").strip() or None
            aliases = unique_aliases(
                preferred,
                protein_name if protein_name != preferred else None,
                parenthetical_aliases(protein_name),
                gene_primary,
                split_space(row.get("gene_synonym") or ""),
                split_semicolon(row.get("ec") or ""),
            )
            loader.add_entity(
                "protein",
                preferred,
                accession,
                (row.get("organism_name") or "").strip() or None,
                parse_int(row.get("organism_id")),
                aliases,
            )


def _flush_cellosaurus_entry(loader: Loader, entry: dict) -> None:
    preferred = (entry.get("ID") or "").strip()
    accession = (entry.get("AC") or "").strip()
    ox_list: list[tuple[int | None, str | None]] = entry.get("OX") or []
    taxon_id = None
    organisms: list[str] = []
    for tax, org in ox_list:
        if taxon_id is None and tax is not None:
            taxon_id = tax
        if org:
            organisms.append(org)
    organism = "; ".join(organisms) if organisms else None
    aliases = unique_aliases(
        preferred,
        split_semicolon(entry.get("SY") or ""),
        split_semicolon(entry.get("AS") or ""),
    )
    loader.add_entity("cell_line", preferred, accession, organism, taxon_id, aliases)


def load_cellosaurus(loader: Loader, path: Path) -> None:
    entry: dict | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if raw.startswith("ID   "):
                if entry is not None:
                    _flush_cellosaurus_entry(loader, entry)
                entry = {"ID": raw[5:].rstrip("\n"), "OX": []}
                continue
            if entry is None:
                continue
            if raw.startswith("//"):
                _flush_cellosaurus_entry(loader, entry)
                entry = None
                continue
            if len(raw) < 5:
                continue
            code = raw[:2]
            value = raw[5:].rstrip("\n")
            if code == "AC":
                entry["AC"] = value
            elif code == "SY":
                entry["SY"] = value
            elif code == "AS":
                entry["AS"] = value
            elif code == "OX":
                entry["OX"].append(parse_ox(value))
        if entry is not None:
            _flush_cellosaurus_entry(loader, entry)


def apply_schema(conn: sqlite3.Connection) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)


def validate(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    egfr = cur.execute(
        "SELECT id, type, preferred_name, accession, organism, taxon_id "
        "FROM entity WHERE accession = ?",
        ("P00533",),
    ).fetchone()
    hela = cur.execute(
        "SELECT id, type, preferred_name, accession, organism, taxon_id "
        "FROM entity WHERE accession = ?",
        ("CVCL_0030",),
    ).fetchone()
    fk = cur.execute("PRAGMA foreign_key_check").fetchall()
    print("--- validation ---")
    print("P00533:", egfr)
    if egfr:
        names = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM alias WHERE entity_id = ? ORDER BY name",
                (egfr[0],),
            )
        ]
        print("P00533 aliases:", names)
    print("CVCL_0030:", hela)
    if hela:
        names = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM alias WHERE entity_id = ? ORDER BY name",
                (hela[0],),
            )
        ]
        print("CVCL_0030 aliases:", names)
    print("foreign_key_check:", fk)


def main() -> int:
    for required in (SCHEMA_PATH, UNIPROT_PATH, CELLOSAURUS_PATH):
        if not required.exists():
            print(f"missing source: {required}", file=sys.stderr)
            return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if DB_PATH.exists():
        DB_PATH.unlink()
    for sidecar in (DATA_DIR / "bio_entities.db-wal", DATA_DIR / "bio_entities.db-shm"):
        if sidecar.exists():
            sidecar.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        apply_schema(conn)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")

        loader = Loader(conn)
        print("loading UniProt...")
        load_uniprot(loader, UNIPROT_PATH)
        print("loading Cellosaurus...")
        load_cellosaurus(loader, CELLOSAURUS_PATH)
        loader.flush()
        conn.commit()

        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.commit()

        elapsed = time.perf_counter() - started
        print("--- counts ---")
        print(f"protein: {loader.n_protein}")
        print(f"cell_line: {loader.n_cell_line}")
        print(f"alias: {loader.n_alias}")
        print(f"skipped_no_key: {loader.skipped_no_key}")
        print(f"skipped_dup_accession: {loader.skipped_dup_accession}")
        print(f"elapsed_s: {elapsed:.1f}")
        validate(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
