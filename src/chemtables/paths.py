"""Filesystem locations for chemtables package data and runtime resources."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent

DATA_DIR = PACKAGE_DIR / "data"
STOPWORDS_PATH = DATA_DIR / "stopwords-en.txt"
BIO_ENTITIES_SCHEMA_PATH = DATA_DIR / "bio_entities.sql"

BIO_ENTITIES_DB_ENV_VAR = "CHEMTABLES_BIO_ENTITIES_DB"


def default_bio_entities_db() -> Path:
    """Resolve bio_entities.db: env var override, else ./data/bio_entities.db.

    The database (built from UniProtKB + Cellosaurus) is a large runtime
    artifact and is never bundled with the package. When absent, target and
    protein/cell-line matching is silently disabled (see
    chemtables.matching.bio_entity.open_catalog).
    """
    override = os.environ.get(BIO_ENTITIES_DB_ENV_VAR)
    if override:
        return Path(override)
    return Path("data") / "bio_entities.db"


def worker_pythonpath() -> str:
    """Directory to prepend to a worker subprocess's PYTHONPATH.

    Isolated conda envs (paddle, ort) never have chemtables installed; this
    lets `conda run -n <env> python -m chemtables.workers.<name>` import the
    package as plain files regardless of install method (editable src/
    checkout or a regular site-packages install).
    """
    return str(SRC_DIR)
