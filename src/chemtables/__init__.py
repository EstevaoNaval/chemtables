"""chemtables: optional GPU-backed pipeline for extracting bioactivity tables from PDFs.

Public API is intentionally small and lazily loaded so importing a submodule
(e.g. `chemtables.workers.paddleocr_vl`, run from an isolated conda env) never
pulls in wrapper-only dependencies such as pandas or pylatexenc.
"""

from __future__ import annotations

__all__ = [
    "TableExtractionConfig",
    "TableResult",
    "extract_tables",
    "environment_ready",
]

__version__ = "0.1.0"

_PUBLIC_NAMES = frozenset(__all__)


def __getattr__(name: str):
    if name in _PUBLIC_NAMES:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
