"""Merge figure_title blocks that sit above the first table."""

from __future__ import annotations

from typing import Any


def merge_titles_above_table(blocks: list[dict[str, Any]]) -> str | None:
    """Join figure_title contents before the first table with a single space."""
    table_index = next(
        (i for i, block in enumerate(blocks) if block.get("block_label") == "table"),
        None,
    )
    if table_index is None:
        return None

    parts = []
    for block in blocks[:table_index]:
        if block.get("block_label") not in ["figure_title", "text"]:
            continue
        content = str(block.get("block_content") or "").strip()
        if content:
            parts.append(content)
    if not parts:
        return None
    return " ".join(parts)
