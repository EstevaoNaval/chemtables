"""Detect tables in PNGs via lightweight PaddleX layout models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from paddlex import create_model
from paddlex.utils.device import get_default_device

PRIMARY_MODEL = "PicoDet_layout_1x"
FALLBACK_MODEL = "PP-DocLayout-S"
DEFAULT_THRESHOLD = 0.40
# PicoDet_layout_1x uses "table"; PP-DocLayout-S uses "Table".
TABLE_LABEL = "table"


def resolve_device(device: Optional[str] = None) -> str:
    """Prefer CUDA GPU when available; else CPU (PaddleX default helper)."""
    if device:
        return device
    return get_default_device()


def _as_dict(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    if hasattr(item, "json"):
        data = item.json
        if isinstance(data, dict):
            return data
    return {}


def _boxes_from_payload(data: dict) -> List[dict]:
    if "boxes" in data:
        return list(data.get("boxes") or [])
    res = data.get("res")
    if isinstance(res, dict):
        return list(res.get("boxes") or [])
    return []


def normalize_boxes(result: Any) -> List[dict]:
    """Flatten Table/layout boxes from PaddleX predict output."""
    boxes: List[dict] = []
    if isinstance(result, dict):
        return _boxes_from_payload(result)

    for item in result:
        boxes.extend(_boxes_from_payload(_as_dict(item)))
    return boxes


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_table_label(label: Any) -> bool:
    return isinstance(label, str) and label.lower() == TABLE_LABEL


def table_boxes(boxes: Iterable[dict]) -> List[dict]:
    return [_jsonable(b) for b in boxes if _is_table_label(b.get("label"))]


class TableDetector:
    """Load layout models once; reuse across images."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.device = resolve_device(device)
        self._models: dict[str, Any] = {}

    def _get_model(self, model_name: str) -> Any:
        if model_name not in self._models:
            self._models[model_name] = create_model(model_name, device=self.device)
        return self._models[model_name]

    def detect_table_boxes(self, image_path: str | Path, model_name: str) -> List[dict]:
        model = self._get_model(model_name)
        result = model.predict(str(image_path), threshold=self.threshold)
        return table_boxes(normalize_boxes(result))

    def has_table(self, image_path: str | Path) -> tuple[bool, List[dict], str]:
        path = str(image_path)
        for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
            boxes = self.detect_table_boxes(path, model_name)
            if boxes:
                return True, boxes, model_name
        return False, [], ""


def detect_images(
    image_paths: List[Path],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: Optional[str] = None,
) -> list[dict]:
    detector = TableDetector(threshold=threshold, device=device)
    results: list[dict] = []
    for image_path in image_paths:
        found, boxes, model_name = detector.has_table(image_path)
        results.append(
            {
                "image": image_path.name,
                "has_table": found,
                "model": model_name,
                "boxes": boxes,
                "device": detector.device,
            }
        )
    return results


def write_detection_files(
    results: list[dict],
    output_root: Path,
    *,
    image_paths: List[Path],
) -> None:
    by_name = {p.name: p for p in image_paths}
    for item in results:
        stem = Path(item["image"]).stem
        out_dir = output_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_image": str(by_name.get(item["image"], item["image"])),
            "has_table": item["has_table"],
            "model": item["model"],
            "device": item.get("device"),
            "boxes": item["boxes"],
        }
        path = out_dir / "table_detection.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
