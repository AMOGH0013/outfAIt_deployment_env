from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
TOP_KEYWORDS = ("shirt", "tshirt", "tee", "top", "kurta", "hoodie", "sweater", "blouse", "upper")
BOTTOM_KEYWORDS = ("pant", "pants", "jean", "jeans", "trouser", "trousers", "short", "shorts", "skirt", "lower")
ALLOWED_LABELS = {"upper_only", "lower_only", "full_outfit", "unknown"}


@dataclass(frozen=True)
class ImageCase:
    path: Path
    width: int
    height: int
    weak_label: str
    weak_label_source: str
    foreground_ratio: float
    silhouette: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _load_rgb(path: Path, max_side: int = 960) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            width = max(1, int(round(width * scale)))
            height = max(1, int(round(height * scale)))
            im = im.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.array(im, copy=True)


def _estimate_foreground_mask(rgb: np.ndarray) -> np.ndarray | None:
    height, width = rgb.shape[:2]
    patch = max(12, min(28, min(height, width) // 10))
    if patch <= 0:
        return None

    tl = rgb[:patch, :patch, :].reshape(-1, 3)
    tr = rgb[:patch, -patch:, :].reshape(-1, 3)
    bl = rgb[-patch:, :patch, :].reshape(-1, 3)
    br = rgb[-patch:, -patch:, :].reshape(-1, 3)
    corners = np.concatenate([tl, tr, bl, br], axis=0).astype(np.float32)
    if float(corners.std(axis=0).mean()) >= 34.0:
        return None

    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
    fg = dist > 26.0
    if float(fg.mean()) < 0.01:
        return None

    if cv2 is not None:
        fg_u8 = fg.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
        fg = fg_u8 > 0
    return fg


def _component_count(mask: np.ndarray) -> int:
    if mask.size == 0 or not bool(mask.any()):
        return 0
    if cv2 is not None:
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        count = 0
        for idx in range(1, n_labels):
            if int(stats[idx, cv2.CC_STAT_AREA]) >= 24:
                count += 1
        return count

    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if area >= 24:
                count += 1
    return count


def _silhouette_features(mask: np.ndarray | None, height: int, width: int) -> dict[str, float]:
    if mask is None or mask.size == 0 or not bool(mask.any()):
        return {
            "fg_ratio": 0.0,
            "top_margin": 1.0,
            "bottom_extent": 0.0,
            "height_frac": 0.0,
            "top_fill": 0.0,
            "mid_fill": 0.0,
            "bottom_fill": 0.0,
            "bottom_components": 0.0,
            "bottom_mid_ratio": 0.0,
        }

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    crop = mask[y1 : y2 + 1, x1 : x2 + 1]
    row_fill = crop.mean(axis=1)
    n_rows = len(row_fill)
    top = row_fill[: max(1, int(n_rows * 0.30))]
    mid = row_fill[int(n_rows * 0.35) : max(int(n_rows * 0.65), int(n_rows * 0.35) + 1)]
    bottom = row_fill[int(n_rows * 0.70) :]
    bottom_band = crop[int(crop.shape[0] * 0.68) :, :]

    top_fill = float(top.mean()) if top.size else 0.0
    mid_fill = float(mid.mean()) if mid.size else 0.0
    bottom_fill = float(bottom.mean()) if bottom.size else 0.0
    bottom_components = float(_component_count(bottom_band))

    return {
        "fg_ratio": float(mask.mean()),
        "top_margin": float(y1 / max(1, height)),
        "bottom_extent": float((y2 + 1) / max(1, height)),
        "height_frac": float((y2 - y1 + 1) / max(1, height)),
        "top_fill": top_fill,
        "mid_fill": mid_fill,
        "bottom_fill": bottom_fill,
        "bottom_components": bottom_components,
        "bottom_mid_ratio": float(bottom_fill / max(mid_fill, 1e-6)),
    }


def _filename_label(name: str) -> tuple[str | None, str]:
    lower = name.lower()
    if any(token in lower for token in TOP_KEYWORDS):
        return "upper_only", "filename"
    if any(token in lower for token in BOTTOM_KEYWORDS):
        return "lower_only", "filename"
    return None, "none"


def _normalize_manifest_label(value: str | None) -> str | None:
    if not value:
        return None
    v = str(value).strip().lower()
    aliases = {
        "upper": "upper_only",
        "top": "upper_only",
        "upper_only": "upper_only",
        "lower": "lower_only",
        "bottom": "lower_only",
        "lower_only": "lower_only",
        "full": "full_outfit",
        "full_body": "full_outfit",
        "full_outfit": "full_outfit",
        "unknown": "unknown",
        "unsure": "unknown",
    }
    norm = aliases.get(v)
    if norm in ALLOWED_LABELS:
        return norm
    return None


def _load_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        if not isinstance(key, str):
            continue
        lbl = _normalize_manifest_label(str(val) if val is not None else None)
        if lbl:
            out[key.strip()] = lbl
    return out


def _silhouette_label(features: dict[str, float]) -> tuple[str, str]:
    if features["fg_ratio"] <= 0.01:
        return "unknown", "silhouette_empty"

    top_margin = features["top_margin"]
    bottom_extent = features["bottom_extent"]
    height_frac = features["height_frac"]
    top_fill = features["top_fill"]
    mid_fill = features["mid_fill"]
    bottom_fill = features["bottom_fill"]
    bottom_components = int(round(features["bottom_components"]))
    bottom_mid_ratio = features["bottom_mid_ratio"]

    if (
        top_margin <= 0.14
        and bottom_extent >= 0.82
        and height_frac >= 0.58
        and (bottom_components >= 2 or bottom_mid_ratio <= 0.86)
    ):
        return "full_outfit", "silhouette_full"

    if bottom_extent <= 0.82 and top_fill >= bottom_fill * 1.08:
        return "upper_only", "silhouette_upper"

    if top_margin >= 0.10 and (bottom_components >= 2 or bottom_fill >= max(top_fill * 1.05, mid_fill * 0.92)):
        return "lower_only", "silhouette_lower"

    if top_margin <= 0.08 and bottom_extent <= 0.78:
        return "upper_only", "silhouette_upper_soft"

    if top_margin >= 0.18 and bottom_extent >= 0.78:
        return "lower_only", "silhouette_lower_soft"

    return "unknown", "silhouette_unknown"


def collect_image_cases(dataset_dir: Path, label_manifest: Path | None = None) -> list[ImageCase]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    manifest_map = _load_manifest(label_manifest)

    cases: list[ImageCase] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rgb = _load_rgb(path)
        height, width = rgb.shape[:2]
        fg_mask = _estimate_foreground_mask(rgb)
        features = _silhouette_features(fg_mask, height=height, width=width)

        manifest_label = manifest_map.get(path.name)
        if manifest_label:
            weak_label = manifest_label
            weak_source = "manifest"
        else:
            file_label, file_source = _filename_label(path.name)
            sil_label, sil_source = _silhouette_label(features)
            weak_label = file_label or sil_label
            weak_source = file_source if file_label else sil_source

        cases.append(
            ImageCase(
                path=path,
                width=width,
                height=height,
                weak_label=weak_label,
                weak_label_source=weak_source,
                foreground_ratio=features["fg_ratio"],
                silhouette=features,
            )
        )

    return cases
