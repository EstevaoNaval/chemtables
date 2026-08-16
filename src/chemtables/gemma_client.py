"""Stdlib client for chemtables.workers.ort_gemma's JSONL generate session."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from chemtables.paths import worker_pythonpath

DEFAULT_ORT_ENV = os.environ.get("CHEMTABLES_ORT_ENV", "ort")
DEFAULT_CONDA = os.environ.get("CONDA_EXE") or "conda"


GenerateFn = Callable[..., str]


class GemmaSessionError(RuntimeError):
    pass


def _worker_env() -> dict[str, str]:
    """Inherit the caller's environment; prepend chemtables to PYTHONPATH so
    the target conda env's interpreter can import chemtables.workers.* even
    though chemtables itself is never installed there."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (worker_pythonpath(), env.get("PYTHONPATH")) if part
    )
    return env


class GemmaSession:
    """One subprocess, model loaded once; sequential generate() calls share it."""

    def __init__(
        self,
        ort_env: str = DEFAULT_ORT_ENV,
        conda: str = DEFAULT_CONDA,
        cwd: Path | None = None,
    ):
        self.ort_env = ort_env
        self.conda = conda
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._proc: subprocess.Popen | None = None
        self._responses: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    def __enter__(self) -> "GemmaSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [
            self.conda,
            "run",
            "-n",
            self.ort_env,
            "--no-capture-output",
            "python",
            "-m",
            "chemtables.workers.ort_gemma",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            cwd=str(self.cwd),
            env=_worker_env(),
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[gemma_client] skipping non-JSON stdout: {line}", file=sys.stderr)
                continue
            self._responses.put(payload)
        self._responses.put(None)

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int = 4) -> str:
        if self._proc is None or self._proc.stdin is None:
            raise GemmaSessionError("Session not started")

        request = {"messages": messages, "max_new_tokens": max_new_tokens}
        try:
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise GemmaSessionError("Worker stdin closed") from exc

        payload = self._responses.get()
        if payload is None:
            code = self._proc.poll()
            raise GemmaSessionError(f"Worker closed stdout unexpectedly (exit={code})")
        if "error" in payload:
            raise GemmaSessionError(payload["error"])
        if "text" not in payload:
            raise GemmaSessionError(f"Unexpected worker response: {payload}")
        return payload["text"]

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        self._proc = None


def default_generate(messages: list[dict[str, Any]], max_new_tokens: int = 4) -> str:
    """One-shot generate (loads model per call). Prefer GemmaSession for batches."""
    with GemmaSession() as session:
        return session.generate(messages, max_new_tokens=max_new_tokens)
