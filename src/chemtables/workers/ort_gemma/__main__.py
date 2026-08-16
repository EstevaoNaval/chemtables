"""JSONL generate server for Gemma ONNX.

Protocol (stdin/stdout, one JSON object per line):
  request:  {"messages": [...], "max_new_tokens": 4}
  response: {"text": "..."}

EOF on stdin ends the process. Logs go to stderr only.
Model loads on first request (avoids pipe deadlock during startup).
"""

from __future__ import annotations

import json
import sys

from chemtables.workers.ort_gemma.gemma_onnx_runtime import (
    DEFAULT_MAX_NEW_TOKENS,
    get_runtime,
)


def main() -> None:
    runtime = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"invalid JSON: {exc}"}), flush=True)
            continue

        messages = request.get("messages")
        if not isinstance(messages, list):
            print(json.dumps({"error": "messages must be a list"}), flush=True)
            continue

        max_new_tokens = int(request.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        try:
            if runtime is None:
                runtime = get_runtime()
            text = runtime.generate(messages, max_new_tokens=max_new_tokens)
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
