"""Internal stage orchestration: detect -> PaddleOCR-VL -> schema -> measurements.

Runs each GPU-heavy stage as a `conda run -n <env>` subprocess against an
isolated environment (see envs/), and glues the stages together with plain
files under the per-image output directory. Not part of the public API --
see `chemtables.api` for the supported surface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from chemtables.gemma_client import DEFAULT_CONDA, DEFAULT_ORT_ENV, GemmaSession
from chemtables.measurements.extract import write_measurement_outputs
from chemtables.paths import default_bio_entities_db, worker_pythonpath
from chemtables.schema.interpret import write_schema_outputs

DEFAULT_PADDLE_ENV = os.environ.get("CHEMTABLES_PADDLE_ENV", "paddle")


@dataclass
class PipelineSettings:
    """Knobs for a pipeline run; every field has a working default."""

    paddle_env: str = DEFAULT_PADDLE_ENV
    ort_env: str = DEFAULT_ORT_ENV
    conda: str = DEFAULT_CONDA
    skip_existing: bool = False
    write_table_detection: bool = False
    bio_entities_db: Path | None = None
    verbose: bool = True


@dataclass
class ImageResult:
    """Outcome of running the pipeline on a single image (internal shape;
    `chemtables.api.TableResult` is the public equivalent)."""

    image: Path
    status: str  # "ok" | "no_table" | "skipped" | "failed"
    output_dir: Path
    schema: dict | None = None
    measurements: dict | None = None
    error: str | None = None


def _log(settings: PipelineSettings, message: str) -> None:
    if settings.verbose:
        print(f"[chemtables] {message}", file=sys.stderr)


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (worker_pythonpath(), env.get("PYTHONPATH")) if part
    )
    return env


def list_pngs(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"
    )


def _parse_detection_stdout(stdout: str) -> list[dict]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        results = payload.get("results")
        if isinstance(results, list):
            return results
    raise ValueError("No table-detection JSON results found in worker stdout")


def _run_table_detection(
    input_dir: Path,
    output_root: Path,
    images: list[Path],
    settings: PipelineSettings,
) -> list[dict]:
    cmd = [
        settings.conda,
        "run",
        "-n",
        settings.paddle_env,
        "--no-capture-output",
        "python",
        "-m",
        "chemtables.workers.table_detection",
        "--input-dir",
        str(input_dir),
    ]
    for image in images:
        cmd.extend(["--include", image.name])
    if settings.write_table_detection:
        cmd.extend(["--write-detection", "--output-dir", str(output_root)])

    _log(settings, f"table detect: {len(images)} image(s)")
    completed = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", env=_worker_env()
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        if completed.stdout:
            sys.stdout.write(completed.stdout)
        raise RuntimeError(f"table detection failed with exit code {completed.returncode}")
    return _parse_detection_stdout(completed.stdout or "")


def _run_paddle_extract(image: Path, output_dir: Path, settings: PipelineSettings) -> int:
    cmd = [
        settings.conda,
        "run",
        "-n",
        settings.paddle_env,
        "--no-capture-output",
        "python",
        "-m",
        "chemtables.workers.paddleocr_vl",
        "--input",
        str(image),
        "--output-dir",
        str(output_dir),
    ]
    _log(settings, f"paddle: {image.name}")
    completed = subprocess.run(cmd, env=_worker_env())
    return completed.returncode


def _run_schema(
    image: Path,
    output_dir: Path,
    *,
    generate,
    compound_refs: list[str],
) -> tuple[bool, str | None]:
    handoff = output_dir / "extracted_table.json"
    if not handoff.is_file():
        return False, f"missing handoff: {handoff}"

    with open(handoff, encoding="utf-8") as handle:
        payload = json.load(handle)

    grid = payload.get("grid")
    if not grid:
        return False, f"empty grid: {image.name}"

    try:
        write_schema_outputs(
            grid,
            payload.get("title"),
            payload.get("footnotes"),
            output_dir,
            generate,
            compound_refs=compound_refs,
        )
    except Exception as exc:
        return False, f"schema failed: {image.name}: {exc}"
    return True, None


def _run_measurements(
    image: Path,
    output_dir: Path,
    compound_refs: list[str],
    settings: PipelineSettings,
) -> tuple[bool, str | None]:
    try:
        write_measurement_outputs(
            output_dir,
            compound_refs,
            table_id=output_dir.name,
            bio_entities_db=settings.bio_entities_db,
        )
    except Exception as exc:
        return False, f"measurements failed: {image.name}: {exc}"
    return True, None


def _has_existing_outputs(output_dir: Path, *, require_measurements: bool) -> bool:
    base = (
        (output_dir / "extracted_table.json").is_file()
        and (output_dir / "table_schema.json").is_file()
        and (output_dir / "extracted_table.csv").is_file()
    )
    if not require_measurements:
        return base
    return base and (output_dir / "measurements.json").is_file()


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run(
    images: list[Path],
    output_root: Path,
    compound_refs: list[str] | None,
    settings: PipelineSettings,
) -> list[ImageResult]:
    """Run detect -> extract -> schema -> measurements over `images`.

    All `images` must live in the same directory (the underlying detection
    worker batches by input directory). Returns one `ImageResult` per image,
    in the same order as `images`.
    """
    if not images:
        return []

    input_dirs = {image.parent for image in images}
    if len(input_dirs) != 1:
        raise ValueError("all images must live in the same directory")
    input_dir = next(iter(input_dirs))

    if settings.bio_entities_db is None:
        settings.bio_entities_db = default_bio_entities_db()

    results: dict[str, ImageResult] = {}
    pending: list[Path] = []
    for image in images:
        stem_out = output_root / image.stem
        if settings.skip_existing and _has_existing_outputs(
            stem_out, require_measurements=compound_refs is not None
        ):
            _log(settings, f"skip existing: {image.name}")
            results[image.name] = ImageResult(
                image=image,
                status="ok",
                output_dir=stem_out,
                schema=_load_json(stem_out / "table_schema.json"),
                measurements=_load_json(stem_out / "measurements.json"),
            )
        else:
            pending.append(image)

    # Stage 1: table detect pending images.
    has_table: set[str] = set()
    if pending:
        detect_results = _run_table_detection(input_dir, output_root, pending, settings)
        for item in detect_results:
            name = item.get("image")
            if item.get("has_table"):
                has_table.add(name)
            else:
                _log(settings, f"no table: {name}")
                results[name] = ImageResult(
                    image=input_dir / name,
                    status="no_table",
                    output_dir=output_root / Path(name).stem,
                )

    table_images = [p for p in pending if p.name in has_table]

    # Stage 2: PaddleOCR-VL extraction for images with tables.
    handoff_ready: list[Path] = []
    for image in table_images:
        stem_out = output_root / image.stem
        stem_out.mkdir(parents=True, exist_ok=True)
        handoff = stem_out / "extracted_table.json"

        if settings.skip_existing and handoff.is_file() and not (
            stem_out / "table_schema.json"
        ).is_file():
            handoff_ready.append(image)
            continue
        if settings.skip_existing and _has_existing_outputs(
            stem_out, require_measurements=compound_refs is not None
        ):
            results[image.name] = ImageResult(
                image=image,
                status="ok",
                output_dir=stem_out,
                schema=_load_json(stem_out / "table_schema.json"),
                measurements=_load_json(stem_out / "measurements.json"),
            )
            continue

        code = _run_paddle_extract(image, stem_out, settings)
        if code != 0 or not handoff.is_file():
            results[image.name] = ImageResult(
                image=image,
                status="failed",
                output_dir=stem_out,
                error=f"paddle extraction failed (exit={code})",
            )
            continue
        handoff_ready.append(image)

    if compound_refs is None:
        for image in handoff_ready:
            results[image.name] = ImageResult(
                image=image,
                status="skipped",
                output_dir=output_root / image.stem,
                error="compound_refs not provided; schema/measurements skipped",
            )
        return [results[image.name] for image in images if image.name in results]

    # Stage 3: Gemma schema interpretation.
    need_schema: list[Path] = []
    for image in handoff_ready:
        stem_out = output_root / image.stem
        if settings.skip_existing and (stem_out / "table_schema.json").is_file():
            results[image.name] = ImageResult(
                image=image,
                status="ok",
                output_dir=stem_out,
                schema=_load_json(stem_out / "table_schema.json"),
            )
            continue
        need_schema.append(image)

    if need_schema:
        try:
            with GemmaSession(ort_env=settings.ort_env, conda=settings.conda) as session:
                for image in need_schema:
                    stem_out = output_root / image.stem
                    ok, error = _run_schema(
                        image,
                        stem_out,
                        generate=session.generate,
                        compound_refs=compound_refs,
                    )
                    if ok:
                        results[image.name] = ImageResult(
                            image=image,
                            status="ok",
                            output_dir=stem_out,
                            schema=_load_json(stem_out / "table_schema.json"),
                        )
                    else:
                        results[image.name] = ImageResult(
                            image=image, status="failed", output_dir=stem_out, error=error
                        )
        except Exception as exc:
            for image in need_schema:
                results[image.name] = ImageResult(
                    image=image,
                    status="failed",
                    output_dir=output_root / image.stem,
                    error=f"gemma session failed: {exc}",
                )

    # Stage 4: measurements, for every image that reached a clean schema.
    for image in images:
        existing = results.get(image.name)
        if existing is None or existing.status != "ok":
            continue
        stem_out = existing.output_dir
        if not (
            (stem_out / "extracted_table.csv").is_file()
            and (stem_out / "table_schema.json").is_file()
        ):
            continue

        schema_payload = existing.schema or {}
        schema_obj = schema_payload.get("schema") or {}
        axis = schema_payload.get("compound_axis") or {}
        if schema_obj.get("skip_extract") or axis.get("skip_extract"):
            _log(settings, f"skip measurements (no compound axis): {image.name}")
            write_measurement_outputs(
                stem_out,
                compound_refs,
                table_id=stem_out.name,
                bio_entities_db=settings.bio_entities_db,
            )
            existing.measurements = _load_json(stem_out / "measurements.json")
            continue

        if settings.skip_existing and (stem_out / "measurements.json").is_file():
            existing.measurements = _load_json(stem_out / "measurements.json")
            continue

        ok, error = _run_measurements(image, stem_out, compound_refs, settings)
        if ok:
            existing.measurements = _load_json(stem_out / "measurements.json")
        else:
            existing.status = "failed"
            existing.error = error

    return [results[image.name] for image in images if image.name in results]
