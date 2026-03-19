# app/services/yolo_detection.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

_IMPORT_ERROR: Exception | None = None
_LOAD_ERROR: Exception | None = None
_YOLO = None
_DEVICE: str = "cpu"
_YOLO_WEIGHTS: Path | None = None


@dataclass(frozen=True)
class YoloDetection:
    cls: int
    label: Literal["upper", "lower"]
    confidence: float
    # xyxy in absolute image coordinates
    x1: int
    y1: int
    x2: int
    y2: int


try:
    import torch
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e
else:
    try:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _REPO_ROOT = Path(__file__).resolve().parents[2]
        env_path = (os.getenv("YOLO_MODEL_PATH") or "").strip()
        weight_candidates: list[Path] = []
        if env_path:
            weight_candidates.append(Path(env_path))
        weight_candidates.extend(
            [
                _REPO_ROOT / "best_stage2_FINAL.pt",
                _REPO_ROOT / "best.pt",
            ]
        )

        weights = next((candidate for candidate in weight_candidates if candidate.exists()), None)
        if weights is None:
            raise FileNotFoundError(
                "YOLO weights not found. Checked YOLO_MODEL_PATH, best_stage2_FINAL.pt, and best.pt in the repo root."
            )

        _YOLO_WEIGHTS = weights
        _YOLO = YOLO(str(weights))
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = e


def detect_upper_lower(
    image_path: str,
    *,
    conf: float = 0.35,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 10,
) -> list[YoloDetection]:
    """Run YOLOv8 on an image and return upper/lower detections.

    Expected model classes:
      - 0: upper
      - 1: lower

    If your model's class ordering differs, set YOLO_CLASS_MAP, e.g.:
      YOLO_CLASS_MAP="upper=1,lower=0"
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "ultralytics/torch not available; install ultralytics to enable YOLO detection."
        ) from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("YOLO model failed to load.") from _LOAD_ERROR
    if _YOLO is None:
        raise RuntimeError("YOLO model is not initialized.")

    cls_map: dict[int, str] = {0: "upper", 1: "lower"}
    raw_map = (os.getenv("YOLO_CLASS_MAP") or "").strip()
    if raw_map:
        try:
            mapping: dict[str, int] = {}
            for part in raw_map.split(","):
                k, v = part.split("=")
                mapping[k.strip()] = int(v.strip())
            cls_map = {mapping["upper"]: "upper", mapping["lower"]: "lower"}
        except Exception:
            # Ignore invalid overrides
            pass

    def _load_source(path: str):
        # Some OpenCV builds struggle with AVIF/WEBP decode; PIL decode keeps detection stable.
        ext = Path(path).suffix.lower()
        source_mode = (os.getenv("YOLO_SOURCE_MODE") or "auto").strip().lower()
        if source_mode == "path":
            return path
        if source_mode == "pil" or ext in {".avif", ".webp"}:
            with Image.open(path) as im:
                return np.array(im.convert("RGB"), copy=True)
        return path

    source = _load_source(image_path)
    results = _YOLO.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
        device=_DEVICE,
    )

    if not results:
        return []

    r0 = results[0]
    if getattr(r0, "boxes", None) is None or r0.boxes is None:
        return []

    boxes = r0.boxes

    try:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
    except Exception:
        return []

    detections: list[YoloDetection] = []
    for (x1, y1, x2, y2), c, score in zip(xyxy, cls, confs):
        label = cls_map.get(int(c))
        if label not in ("upper", "lower"):
            continue
        detections.append(
            YoloDetection(
                cls=int(c),
                label=label,  # type: ignore[arg-type]
                confidence=float(score),
                x1=int(round(float(x1))),
                y1=int(round(float(y1))),
                x2=int(round(float(x2))),
                y2=int(round(float(y2))),
            )
        )

    return detections


def _yolo_device() -> str:
    """For diagnostics/logging."""
    return _DEVICE
