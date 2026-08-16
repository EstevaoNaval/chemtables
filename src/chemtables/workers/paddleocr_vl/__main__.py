"""CLI: PNG → extracted_table.json (conda run -n paddle python -m chemtables.workers.paddleocr_vl)."""

from __future__ import annotations

import argparse
from pathlib import Path

from chemtables.workers.paddleocr_vl.extract import extract_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract table grid/title/footnotes from a PNG via PaddleOCR-VL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input PNG")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for extracted_table.json and paddleocr artifacts",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    extract_table(input_path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
