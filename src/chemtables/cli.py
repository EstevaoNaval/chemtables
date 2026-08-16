"""Argparse front-end over chemtables.api.

Prefer the Python API directly for programmatic use:
`from chemtables import extract_tables`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chemtables.api import TableExtractionConfig, extract_tables
from chemtables.pipeline import list_pngs


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch PNG -> table detect -> PaddleOCR-VL -> table schema "
            "-> measurements pipeline."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("images"),
        help="Directory of PNG images (non-recursive)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Root output directory (default: output)",
    )
    parser.add_argument("--paddle-env", default=TableExtractionConfig.paddle_env)
    parser.add_argument("--ort-env", default=TableExtractionConfig.ort_env)
    parser.add_argument(
        "--conda",
        default=TableExtractionConfig.conda,
        help="conda executable (default: CONDA_EXE or 'conda')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip images that already have schema outputs "
            "(and measurements.json when --compound-refs is set). "
            "Without --compound-refs, schema/measurements are skipped anyway."
        ),
    )
    parser.add_argument(
        "--write-table-detection",
        action="store_true",
        help="Write output/<stem>/table_detection.json (default: off)",
    )
    parser.add_argument(
        "--compound-refs",
        type=Path,
        default=None,
        help=(
            "JSON array of compound coreference strings. Required for "
            "schema interpretation + measurement extraction. When omitted, "
            "those stages are skipped."
        ),
    )
    parser.add_argument(
        "--bio-entities-db",
        type=Path,
        default=None,
        help="SQLite bio_entities.db (default: ./data/bio_entities.db)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    compound_refs: list[str] | None = None
    if args.compound_refs is not None:
        refs_path = args.compound_refs.resolve()
        if not refs_path.is_file():
            raise SystemExit(f"compound-refs not found: {refs_path}")
        with open(refs_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise SystemExit("compound-refs must be a JSON array of strings")
        compound_refs = loaded

    images = list_pngs(input_dir)
    if not images:
        raise SystemExit(f"No PNG files in {input_dir}")

    print(f"[chemtables] {len(images)} image(s) in {input_dir}")
    if compound_refs is None:
        print("[chemtables] schema/measurements skipped (no --compound-refs)")

    config = TableExtractionConfig(
        paddle_env=args.paddle_env,
        ort_env=args.ort_env,
        conda=args.conda,
        skip_existing=args.skip_existing,
        write_table_detection=args.write_table_detection,
        bio_entities_db=args.bio_entities_db,
    )
    results = extract_tables(
        images,
        compound_refs=compound_refs,
        output_dir=args.output_dir,
        config=config,
    )

    ok = sum(1 for r in results if r.status == "ok")
    skipped_no_table = sum(1 for r in results if r.status == "no_table")
    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        print(f"[chemtables] failed: {r.image.name}: {r.error}", file=sys.stderr)

    print(
        f"[chemtables] finished: {ok} ok, {len(failed)} failed, "
        f"{skipped_no_table} skipped_no_table"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
