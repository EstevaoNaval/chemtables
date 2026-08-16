"""CLI: PNG dir → table detection JSON (conda run -n paddle python -m chemtables.workers.table_detection)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chemtables.workers.table_detection.detect import (
    DEFAULT_THRESHOLD,
    detect_images,
    write_detection_files,
)


def list_pngs(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect tables in PNG images via PaddleX layout models."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of PNG images (non-recursive)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Detection score threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PaddleX device (default: gpu:0 if CUDA else cpu)",
    )
    parser.add_argument(
        "--write-detection",
        action="store_true",
        help="Write output/<stem>/table_detection.json per image (default: off)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory (required with --write-detection)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Optional PNG basename to process (repeatable; default: all in --input-dir)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    if args.write_detection and args.output_dir is None:
        raise SystemExit("--output-dir is required when --write-detection is set")

    images = list_pngs(input_dir)
    if args.include:
        wanted = set(args.include)
        images = [p for p in images if p.name in wanted]
        missing = wanted - {p.name for p in images}
        if missing:
            raise SystemExit(f"Include image(s) not found in {input_dir}: {sorted(missing)}")
    if not images:
        raise SystemExit(f"No PNG files in {input_dir}")

    print(f"[table_detection] {len(images)} image(s) device={args.device or 'auto'}", file=sys.stderr)

    results = detect_images(images, threshold=args.threshold, device=args.device)

    if args.write_detection:
        write_detection_files(
            results,
            args.output_dir.resolve(),
            image_paths=images,
        )
        print(
            f"[table_detection] wrote table_detection.json under {args.output_dir}",
            file=sys.stderr,
        )

    # Single JSON line for orchestrator (logs go to stderr).
    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)

    n_table = sum(1 for r in results if r["has_table"])
    print(
        f"[table_detection] done: {n_table}/{len(results)} with table",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
