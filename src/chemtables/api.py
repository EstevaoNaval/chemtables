"""Public API surface for chemtables.

Consumers should only ever import from here:

    from chemtables import TableExtractionConfig, extract_tables

    results = extract_tables(
        images=["images/table_chemical_6.png"],
        compound_refs=["1a", "2b"],
        output_dir="output",
    )
    for result in results:
        print(result.status, result.schema, result.measurements)

Everything else in this package is an implementation detail: which GPU
libraries are used, how workers are spawned, and how conda environments are
selected are not part of the contract and may change without notice.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from chemtables import pipeline
from chemtables.gemma_client import DEFAULT_CONDA, DEFAULT_ORT_ENV
from chemtables.paths import default_bio_entities_db

DEFAULT_PADDLE_ENV = pipeline.DEFAULT_PADDLE_ENV


@dataclass
class TableExtractionConfig:
    """Optional overrides for `extract_tables`.

    Defaults work out of the box once the `paddle` and `ort` conda
    environments exist (see envs/environment.paddle.yml and
    envs/environment.ort.yml).
    """

    paddle_env: str = DEFAULT_PADDLE_ENV
    ort_env: str = DEFAULT_ORT_ENV
    conda: str = DEFAULT_CONDA
    skip_existing: bool = False
    write_table_detection: bool = False
    bio_entities_db: str | Path | None = None
    verbose: bool = True


@dataclass
class TableResult:
    """Outcome of running the pipeline on a single input image."""

    image: Path
    status: str  # "ok" | "no_table" | "skipped" | "failed"
    output_dir: Path
    schema: dict | None = None
    measurements: dict | None = None
    error: str | None = None


def _conda_envs(conda: str) -> set[str]:
    try:
        completed = subprocess.run(
            [conda, "env", "list"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    names = set()
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0])
    return names


def environment_ready(config: TableExtractionConfig | None = None) -> bool:
    """True when `conda` and the paddle/ort environments needed by
    `extract_tables` appear to be available.

    Callers can use this to degrade gracefully instead of letting
    `extract_tables` raise partway through a run.
    """
    config = config or TableExtractionConfig()
    if shutil.which(config.conda) is None:
        return False
    envs = _conda_envs(config.conda)
    return config.paddle_env in envs and config.ort_env in envs


def extract_tables(
    images: list[str | Path],
    *,
    compound_refs: list[str] | None = None,
    output_dir: str | Path = "output",
    config: TableExtractionConfig | None = None,
) -> list[TableResult]:
    """Detect, OCR, and interpret bioactivity tables in `images`.

    Runs table detection, PaddleOCR-VL extraction, LLM-assisted schema
    interpretation, and deterministic measurement extraction, in that order.

    Args:
        images: paths to PNG images. All must live in the same directory.
        compound_refs: chemical coreference strings (e.g. compound IDs) used
            to locate the compound axis and bind measurements to compounds.
            When omitted, only detection + OCR run; schema and measurement
            stages are skipped for every image (status "skipped").
        output_dir: root directory for per-image output subdirectories.
        config: optional `TableExtractionConfig` overrides.

    Returns:
        One `TableResult` per input image, in the same order as `images`.

    Raises:
        ValueError: if `images` span more than one directory.
        RuntimeError: if the table-detection worker subprocess fails outright
            (e.g. the paddle conda environment doesn't exist).
    """
    config = config or TableExtractionConfig()
    image_paths = [Path(image).resolve() for image in images]
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    bio_entities_db = (
        Path(config.bio_entities_db)
        if config.bio_entities_db is not None
        else default_bio_entities_db()
    )
    settings = pipeline.PipelineSettings(
        paddle_env=config.paddle_env,
        ort_env=config.ort_env,
        conda=config.conda,
        skip_existing=config.skip_existing,
        write_table_detection=config.write_table_detection,
        bio_entities_db=bio_entities_db,
        verbose=config.verbose,
    )

    image_results = pipeline.run(image_paths, output_root, compound_refs, settings)
    return [
        TableResult(
            image=r.image,
            status=r.status,
            output_dir=r.output_dir,
            schema=r.schema,
            measurements=r.measurements,
            error=r.error,
        )
        for r in image_results
    ]
