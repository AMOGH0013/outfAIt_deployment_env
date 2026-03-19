from __future__ import annotations

import colorsys
import json
import math
import shutil
import tempfile
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from app.services.color_extraction import (
    _filter_garment_pixels_bgr,
    _lab_to_color_name,
    _load_image_bgr,
    _load_mask_bool,
    _rgb_array_to_lab,
    _shrink_mask_for_color,
    extract_dominant_colors,
)
from app.services.sam2_segmentation import load_image_rgb, save_mask_png, segment_bbox_sam2
from app.services.sam_segmentation import segment_bbox_sam1
from app.services.yolo_detection import YoloDetection, detect_upper_lower
from evaluation.internal_eval.dataset import ImageCase

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


PALETTE_SWATCHES = {
    "hot pink": "#ff4fa3",
    "baby pink": "#f8c8dc",
    "dusty rose": "#c98793",
    "pink": "#ef6ca7",
    "burgundy": "#7a1730",
    "maroon": "#7b2435",
    "red": "#cf243b",
    "orange": "#ef7d22",
    "yellow": "#d7b227",
    "olive": "#667a2f",
    "forest green": "#1f6c47",
    "green": "#2ea86f",
    "teal": "#1c8b83",
    "powder blue": "#a6c8ef",
    "sky blue": "#72b6ef",
    "royal blue": "#345edb",
    "indigo": "#433a8b",
    "navy": "#1d3557",
    "blue": "#3a7be0",
    "lavender": "#b7a7e5",
    "purple": "#7850b2",
    "beige": "#d8c29a",
    "tan": "#bc9668",
    "brown": "#7b5433",
    "off white": "#f3eee1",
    "white": "#f7f7f4",
    "charcoal": "#3b4148",
    "dark gray": "#585e67",
    "light gray": "#bfc5cd",
    "gray": "#8f97a1",
    "black": "#13161b",
}


@dataclass(frozen=True)
class ColorProfile:
    label: str
    centroid_lab: tuple[float, float, float]
    palette: list[str]
    confidence: float


def _safe_name(path: Path) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)
    return safe[:80] or "image"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    arr_a = np.array(a, dtype=np.float32)
    arr_b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(arr_a - arr_b))


def _best_detections(detections: list[YoloDetection]) -> dict[str, YoloDetection]:
    best: dict[str, YoloDetection] = {}
    for det in detections:
        current = best.get(det.label)
        if current is None or float(det.confidence) > float(current.confidence):
            best[det.label] = det
    return best


def _detection_status(case: ImageCase, detections: list[YoloDetection]) -> tuple[bool, str]:
    labels = {det.label for det in detections}
    if not detections:
        return False, "missed_detection"

    if case.weak_label == "upper_only":
        if "upper" in labels and "lower" not in labels:
            return True, "ok"
        if "upper" not in labels:
            return False, "missed_detection"
        return False, "wrong_class"

    if case.weak_label == "lower_only":
        if "lower" in labels and "upper" not in labels:
            return True, "ok"
        if "lower" not in labels:
            return False, "missed_detection"
        return False, "wrong_class"

    if case.weak_label == "full_outfit":
        return ("upper" in labels and "lower" in labels), ("ok" if {"upper", "lower"} <= labels else "missed_detection")

    return True, "ok"


def _component_count(mask: np.ndarray) -> tuple[int, float]:
    if mask.size == 0 or not bool(mask.any()):
        return 0, 0.0
    if cv2 is not None:
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        areas = []
        for idx in range(1, n_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= 12:
                areas.append(area)
        if not areas:
            return 0, 0.0
        total = float(sum(areas))
        return len(areas), float(max(areas) / max(total, 1.0))

    visited = np.zeros_like(mask, dtype=bool)
    areas: list[int] = []
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
            if area >= 12:
                areas.append(area)
    if not areas:
        return 0, 0.0
    total = float(sum(areas))
    return len(areas), float(max(areas) / max(total, 1.0))


def _mask_quality(mask_bool: np.ndarray, bbox: tuple[int, int, int, int], sam1_mask: np.ndarray | None) -> dict[str, float]:
    x1, y1, x2, y2 = bbox
    box_mask = mask_bool[y1:y2, x1:x2]
    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)
    box_area = float(box_h * box_w)
    area = float(box_mask.sum())
    area_ratio = area / max(box_area, 1.0)

    edge_band_px = max(1, int(round(min(box_h, box_w) * 0.04)))
    edge_band = np.zeros_like(box_mask, dtype=bool)
    edge_band[:edge_band_px, :] = True
    edge_band[-edge_band_px:, :] = True
    edge_band[:, :edge_band_px] = True
    edge_band[:, -edge_band_px:] = True
    edge_touch_ratio = float(box_mask[edge_band].mean()) if bool(edge_band.any()) else 0.0

    component_count, largest_component_ratio = _component_count(box_mask)

    agreement_iou = 0.0
    if sam1_mask is not None:
        sam1_box = sam1_mask[y1:y2, x1:x2]
        inter = float(np.logical_and(box_mask, sam1_box).sum())
        union = float(np.logical_or(box_mask, sam1_box).sum())
        agreement_iou = inter / max(union, 1.0)

    quality_score = 1.0
    if area_ratio < 0.08:
        quality_score -= 0.35
    elif area_ratio > 0.92:
        quality_score -= 0.25
    if edge_touch_ratio > 0.40:
        quality_score -= 0.20
    if component_count > 3:
        quality_score -= 0.15
    if largest_component_ratio < 0.78:
        quality_score -= 0.15
    if sam1_mask is not None and agreement_iou < 0.40:
        quality_score -= 0.15

    return {
        "area_ratio": round(area_ratio, 4),
        "edge_touch_ratio": round(edge_touch_ratio, 4),
        "component_count": float(component_count),
        "largest_component_ratio": round(largest_component_ratio, 4),
        "sam2_sam1_iou": round(agreement_iou, 4),
        "quality_score": round(max(0.0, min(1.0, quality_score)), 4),
    }


def _color_pixels(image_path: str, mask_path: str, seed: int, sample_size: int = 5000, max_side: int = 256) -> np.ndarray:
    image_bgr = _load_image_bgr(image_path, max_side=max_side)
    height, width = image_bgr.shape[:2]
    mask_bool = _load_mask_bool(mask_path, size_hw=(height, width))
    selection = _shrink_mask_for_color(mask_bool)
    pixels = image_bgr.reshape(-1, 3)[selection.reshape(-1)]
    if pixels.shape[0] < 200:
        pixels = image_bgr.reshape(-1, 3)
    pixels = _filter_garment_pixels_bgr(pixels)
    if pixels.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    if pixels.shape[0] > sample_size:
        idx = rng.choice(pixels.shape[0], size=sample_size, replace=False)
        pixels = pixels[idx]
    return pixels[:, ::-1].astype(np.float32)


def _lab_profile(image_path: str, mask_path: str, seed: int, k: int = 5) -> ColorProfile:
    pixels_rgb = _color_pixels(image_path, mask_path, seed=seed)
    if pixels_rgb.shape[0] == 0:
        return ColorProfile(label="unknown", centroid_lab=(0.0, 0.0, 0.0), palette=[], confidence=0.0)

    lab = _rgb_array_to_lab(pixels_rgb)
    k_eff = min(k, int(lab.shape[0]))
    model = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = model.fit_predict(lab)
    counts = np.bincount(labels, minlength=k_eff).astype(np.float32)
    order = np.argsort(-counts)
    primary_idx = int(order[0])
    center = tuple(float(x) for x in model.cluster_centers_[primary_idx])
    primary_label, _best_internal, _second_internal, _gap, _conf = _lab_to_color_name(center)

    total = float(counts.sum()) if float(counts.sum()) > 0 else 1.0
    palette: list[str] = []
    for idx in order.tolist():
        frac = float(counts[idx] / total)
        if frac < 0.08:
            continue
        label, _best_internal, _second_internal, _gap, _conf = _lab_to_color_name(
            tuple(float(x) for x in model.cluster_centers_[idx])
        )
        if label not in palette:
            palette.append(label)
    return ColorProfile(
        label=primary_label,
        centroid_lab=center,
        palette=palette,
        confidence=round(float(counts[primary_idx] / total), 4),
    )


def _hsv_name(h: float, s: float, v: float) -> str:
    if v >= 0.92 and s < 0.10:
        return "white"
    if v >= 0.82 and s < 0.16:
        return "off white"
    if v <= 0.16:
        return "black"
    if s < 0.14 and v <= 0.35:
        return "charcoal"
    if s < 0.18 and v < 0.58:
        return "dark gray"
    if s < 0.18 and v >= 0.75:
        return "light gray"
    if s < 0.20:
        return "gray"

    deg = (h % 1.0) * 360.0
    if deg < 12 or deg >= 345:
        return "red"
    if deg < 28:
        return "orange"
    if deg < 55:
        return "yellow"
    if deg < 90:
        return "olive"
    if deg < 150:
        return "green"
    if deg < 190:
        return "teal"
    if deg < 235:
        return "blue"
    if deg < 270:
        return "indigo"
    if deg < 305:
        return "purple"
    if deg < 338:
        if s < 0.35 and v > 0.82:
            return "baby pink"
        return "hot pink" if s >= 0.55 else "pink"
    return "red"


def _hsv_profile(image_path: str, mask_path: str, seed: int, k: int = 5) -> ColorProfile:
    pixels_rgb = _color_pixels(image_path, mask_path, seed=seed)
    if pixels_rgb.shape[0] == 0:
        return ColorProfile(label="unknown", centroid_lab=(0.0, 0.0, 0.0), palette=[], confidence=0.0)

    rgb = pixels_rgb.astype(np.float32) / 255.0
    if cv2 is not None:
        hsv = cv2.cvtColor((rgb.reshape(-1, 1, 3) * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        hue = hsv[:, 0] / 179.0
        sat = hsv[:, 1] / 255.0
        val = hsv[:, 2] / 255.0
    else:
        hue = np.zeros((rgb.shape[0],), dtype=np.float32)
        val = np.max(rgb, axis=1)
        maxc = np.max(rgb, axis=1)
        minc = np.min(rgb, axis=1)
        delta = maxc - minc
        sat = np.where(maxc > 0, delta / np.maximum(maxc, 1e-6), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = (maxc - rgb[:, 0]) / np.maximum(delta, 1e-6)
            gc = (maxc - rgb[:, 1]) / np.maximum(delta, 1e-6)
            bc = (maxc - rgb[:, 2]) / np.maximum(delta, 1e-6)
            hue = np.select(
                [rgb[:, 0] == maxc, rgb[:, 1] == maxc],
                [bc - gc, 2.0 + rc - bc],
                default=4.0 + gc - rc,
            )
            hue = (hue / 6.0) % 1.0

    features = np.stack([np.cos(2.0 * math.pi * hue), np.sin(2.0 * math.pi * hue), sat, val], axis=1)
    k_eff = min(k, int(features.shape[0]))
    model = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = model.fit_predict(features)
    counts = np.bincount(labels, minlength=k_eff).astype(np.float32)
    primary_idx = int(np.argmax(counts))
    cluster_mask = labels == primary_idx
    hue_mean = float(np.mod(np.angle(np.mean(np.exp(1j * 2.0 * math.pi * hue[cluster_mask]))) / (2.0 * math.pi), 1.0))
    sat_mean = float(np.mean(sat[cluster_mask]))
    val_mean = float(np.mean(val[cluster_mask]))
    rgb_centroid = np.array(
        tuple(int(round(c * 255.0)) for c in colorsys.hsv_to_rgb(hue_mean, sat_mean, val_mean)),
        dtype=np.float32,
    )
    centroid_lab = tuple(float(x) for x in _rgb_array_to_lab(rgb_centroid.reshape(1, 3))[0])
    primary_label = _hsv_name(hue_mean, sat_mean, val_mean)

    total = float(counts.sum()) if float(counts.sum()) > 0 else 1.0
    palette = [primary_label]
    return ColorProfile(
        label=primary_label,
        centroid_lab=centroid_lab,
        palette=palette,
        confidence=round(float(counts[primary_idx] / total), 4),
    )


def _edge_leakage(image_path: str, mask_path: str) -> dict[str, float]:
    image_bgr = _load_image_bgr(image_path, max_side=256)
    height, width = image_bgr.shape[:2]
    mask = _load_mask_bool(mask_path, size_hw=(height, width))
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]
    mask = mask.astype(bool)
    if not bool(mask.any()):
        return {"interior_var": 0.0, "edge_var": 0.0, "edge_delta_e": 0.0, "leakage_score": 0.0}

    if cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=2) > 0
    else:
        eroded = mask.copy()
    eroded = np.asarray(eroded)
    eroded = np.squeeze(eroded)
    if eroded.ndim > 2:
        eroded = eroded[..., 0]
    eroded = eroded.astype(bool)
    interior = eroded if bool(eroded.any()) else mask
    edge_ring = mask & ~interior
    if not bool(edge_ring.any()):
        edge_ring = mask

    rgb_flat = image_bgr[:, :, ::-1].reshape(-1, 3)
    interior_flat = interior.reshape(-1)
    edge_flat = edge_ring.reshape(-1)
    if interior_flat.shape[0] != rgb_flat.shape[0]:
        interior_flat = interior_flat[: rgb_flat.shape[0]]
    if edge_flat.shape[0] != rgb_flat.shape[0]:
        edge_flat = edge_flat[: rgb_flat.shape[0]]
    interior_rgb = rgb_flat[interior_flat].astype(np.float32)
    edge_rgb = rgb_flat[edge_flat].astype(np.float32)
    if interior_rgb.shape[0] < 16 or edge_rgb.shape[0] < 16:
        return {"interior_var": 0.0, "edge_var": 0.0, "edge_delta_e": 0.0, "leakage_score": 0.0}

    interior_lab = _rgb_array_to_lab(interior_rgb)
    edge_lab = _rgb_array_to_lab(edge_rgb)
    interior_var = float(np.var(interior_lab, axis=0).mean())
    edge_var = float(np.var(edge_lab, axis=0).mean())
    edge_delta = float(np.linalg.norm(edge_lab.mean(axis=0) - interior_lab.mean(axis=0)))
    leakage = min(1.0, (edge_var / max(interior_var, 1.0)) * 0.25 + edge_delta / 40.0)
    return {
        "interior_var": round(interior_var, 4),
        "edge_var": round(edge_var, 4),
        "edge_delta_e": round(edge_delta, 4),
        "leakage_score": round(leakage, 4),
    }


def _background_fill_color(image_path: str) -> tuple[int, int, int]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        arr = np.array(im, copy=False)
    h, w = arr.shape[:2]
    patch = max(8, min(24, min(h, w) // 10))
    corners = np.concatenate(
        [
            arr[:patch, :patch, :].reshape(-1, 3),
            arr[:patch, -patch:, :].reshape(-1, 3),
            arr[-patch:, :patch, :].reshape(-1, 3),
            arr[-patch:, -patch:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(int(x) for x in np.median(corners, axis=0))


def _write_palette_image(path: Path, title: str, labels: list[str]) -> None:
    swatches = labels[:5] if labels else ["unknown"]
    width = 110 * len(swatches)
    img = Image.new("RGB", (max(220, width), 64), color=(14, 18, 28))
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), title[:40], fill=(235, 240, 250))
    for idx, label in enumerate(swatches):
        x = 10 + idx * 105
        color = PALETTE_SWATCHES.get(label.lower(), "#7f8aa3")
        draw.rounded_rectangle((x, 28, x + 84, 54), radius=8, fill=color, outline=(255, 255, 255))
        draw.text((x + 4, 56 - 12), label[:12], fill=(235, 240, 250))
    img.save(path)


def _augment_and_save(image_path: Path, mask_path: Path, out_dir: Path) -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    bg = _background_fill_color(str(image_path))
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        image = image.convert("RGB")
        mask = mask.convert("L")

        variants = [
            ("brightness_up", ImageEnhance.Brightness(image).enhance(1.15), mask),
            ("brightness_down", ImageEnhance.Brightness(image).enhance(0.88), mask),
            ("contrast_up", ImageEnhance.Contrast(image).enhance(1.12), mask),
            ("contrast_down", ImageEnhance.Contrast(image).enhance(0.90), mask),
            ("rotate_pos", image.rotate(5.0, resample=Image.Resampling.BILINEAR, fillcolor=bg), mask.rotate(5.0, resample=Image.Resampling.NEAREST, fillcolor=0)),
            ("rotate_neg", image.rotate(-5.0, resample=Image.Resampling.BILINEAR, fillcolor=bg), mask.rotate(-5.0, resample=Image.Resampling.NEAREST, fillcolor=0)),
        ]

        for name, aug_img, aug_mask in variants:
            img_path = out_dir / f"{name}.png"
            mask_out = out_dir / f"{name}_mask.png"
            aug_img.save(img_path)
            aug_mask.save(mask_out)
            out.append((name, img_path, mask_out))
    return out


def _evaluate_color(image_path: Path, mask_path: Path, palette_dir: Path) -> dict[str, Any]:
    lab_result = extract_dominant_colors(str(image_path), mask_path=str(mask_path))
    lab_profile = _lab_profile(str(image_path), str(mask_path), seed=42)
    hsv_profile = _hsv_profile(str(image_path), str(mask_path), seed=42)

    centroid_runs = [_lab_profile(str(image_path), str(mask_path), seed=seed) for seed in (7, 17, 29, 41, 53)]
    centroids = np.array([run.centroid_lab for run in centroid_runs], dtype=np.float32)
    mean_centroid = centroids.mean(axis=0)
    centroid_var = float(np.mean(np.linalg.norm(centroids - mean_centroid[None, :], axis=1)))

    with tempfile.TemporaryDirectory() as tmpdir:
        aug_dir = Path(tmpdir)
        variants = _augment_and_save(image_path, mask_path, aug_dir)
        lab_drifts: list[float] = []
        hsv_drifts: list[float] = []
        lab_labels = {lab_profile.label}
        hsv_labels = {hsv_profile.label}
        for _name, aug_image, aug_mask in variants:
            aug_lab = _lab_profile(str(aug_image), str(aug_mask), seed=42)
            aug_hsv = _hsv_profile(str(aug_image), str(aug_mask), seed=42)
            lab_drifts.append(_delta_e(lab_profile.centroid_lab, aug_lab.centroid_lab))
            hsv_drifts.append(_delta_e(hsv_profile.centroid_lab, aug_hsv.centroid_lab))
            lab_labels.add(aug_lab.label)
            hsv_labels.add(aug_hsv.label)

    leakage = _edge_leakage(str(image_path), str(mask_path))
    mean_lab_drift = float(np.mean(lab_drifts)) if lab_drifts else 0.0
    mean_hsv_drift = float(np.mean(hsv_drifts)) if hsv_drifts else 0.0
    improvement = ((mean_hsv_drift - mean_lab_drift) / max(mean_hsv_drift, 1e-6)) * 100.0 if mean_hsv_drift > 0 else 0.0
    label_instability = max(0, len(lab_labels) - 1) * 8.0
    disagreement_penalty = 10.0 if lab_profile.label != hsv_profile.label else 0.0
    stability_score = max(
        0.0,
        100.0
        - mean_lab_drift * 2.0
        - centroid_var * 4.0
        - leakage["leakage_score"] * 22.0
        - label_instability
        - disagreement_penalty,
    )
    color_ok = stability_score >= 62.0 and mean_lab_drift <= 18.0

    _write_palette_image(palette_dir / f"{_safe_name(image_path)}_{mask_path.stem}.png", image_path.name, lab_result.palette)

    return {
        "lab_primary": lab_result.primary_color,
        "lab_palette": lab_result.palette,
        "lab_confidence": lab_result.primary_confidence,
        "lab_centroid": [round(float(x), 4) for x in lab_profile.centroid_lab],
        "hsv_primary": hsv_profile.label,
        "hsv_centroid": [round(float(x), 4) for x in hsv_profile.centroid_lab],
        "mean_lab_drift": round(mean_lab_drift, 4),
        "mean_hsv_drift": round(mean_hsv_drift, 4),
        "cluster_centroid_variance": round(centroid_var, 4),
        "color_stability_score": round(stability_score, 2),
        "hsv_vs_lab_improvement_pct": round(improvement, 2),
        "lab_label_variants": sorted(lab_labels),
        "hsv_label_variants": sorted(hsv_labels),
        "background_leakage": leakage,
        "color_ok": color_ok,
        "suspected_color_failure": (not color_ok) or (lab_profile.label != hsv_profile.label and mean_lab_drift > 12.0),
    }


def _copy_failure(case: ImageCase, category: str, target_dir: Path, payload: dict[str, Any]) -> None:
    _ensure_dir(target_dir / category)
    safe = _safe_name(case.path)
    dest = target_dir / category / f"{safe}{case.path.suffix.lower()}"
    shutil.copy2(case.path, dest)
    (target_dir / category / f"{safe}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_vision_evaluation(cases: list[ImageCase], out_dir: Path) -> dict[str, Any]:
    vision_dir = _ensure_dir(out_dir / "vision")
    masks_dir = _ensure_dir(vision_dir / "masks")
    palettes_dir = _ensure_dir(vision_dir / "palettes")
    failures_dir = _ensure_dir(vision_dir / "failures")
    worst_dir = _ensure_dir(vision_dir / "top_20_worst")

    records: list[dict[str, Any]] = []
    detection_failures = 0
    segmentation_failures = 0
    color_failures = 0
    pipeline_success = 0
    detection_evaluable = 0
    segmentation_items = 0
    color_items = 0
    mask_scores: list[float] = []
    color_scores: list[float] = []
    lab_drifts: list[float] = []
    hsv_improvements: list[float] = []
    weak_label_counter = Counter(case.weak_label for case in cases)
    failure_counter = Counter()

    for case in cases:
        image_rgb = load_image_rgb(str(case.path))
        detections = detect_upper_lower(str(case.path), conf=0.20, max_det=20)
        best = _best_detections(detections)
        detection_evaluable += 1
        detection_ok, detection_status = _detection_status(case, detections)
        if not detection_ok:
            detection_failures += 1

        labels_to_check: list[str]
        if case.weak_label == "upper_only":
            labels_to_check = ["upper"]
        elif case.weak_label == "lower_only":
            labels_to_check = ["lower"]
        elif case.weak_label == "full_outfit":
            labels_to_check = ["upper", "lower"]
        else:
            labels_to_check = sorted(best.keys())

        segment_records: list[dict[str, Any]] = []
        image_failures: set[str] = set()

        if not detection_ok:
            image_failures.add(detection_status)

        for label in labels_to_check:
            det = best.get(label)
            if det is None:
                if case.weak_label in {"upper_only", "lower_only", "full_outfit"}:
                    image_failures.add("missed_detection")
                continue

            segmentation_items += 1
            bbox = (int(det.x1), int(det.y1), int(det.x2), int(det.y2))
            mask_path = masks_dir / f"{_safe_name(case.path)}_{label}_sam2.png"

            sam1_mask: np.ndarray | None = None
            try:
                mask_bool = segment_bbox_sam2(image_rgb, bbox)
                save_mask_png(mask_bool, str(mask_path))
            except Exception as exc:
                segmentation_failures += 1
                image_failures.add("poor_segmentation")
                segment_records.append(
                    {
                        "label": label,
                        "bbox_xyxy": list(bbox),
                        "yolo_confidence": round(float(det.confidence), 4),
                        "sam2_error": str(exc),
                    }
                )
                continue

            try:
                sam1_mask = segment_bbox_sam1(image_rgb, bbox)
            except Exception:
                sam1_mask = None

            quality = _mask_quality(mask_bool, bbox, sam1_mask)
            mask_scores.append(quality["quality_score"])
            if quality["quality_score"] < 0.55:
                segmentation_failures += 1
                image_failures.add("poor_segmentation")

            color_items += 1
            color_eval = _evaluate_color(case.path, mask_path, palettes_dir)
            color_scores.append(float(color_eval["color_stability_score"]))
            lab_drifts.append(float(color_eval["mean_lab_drift"]))
            hsv_improvements.append(float(color_eval["hsv_vs_lab_improvement_pct"]))
            if not bool(color_eval["color_ok"]):
                color_failures += 1
                image_failures.add("color_misclassification")

            segment_records.append(
                {
                    "label": label,
                    "bbox_xyxy": list(bbox),
                    "yolo_confidence": round(float(det.confidence), 4),
                    "mask_path": str(mask_path.relative_to(out_dir)),
                    "mask_quality": quality,
                    "color": color_eval,
                }
            )

        if not image_failures and (segment_records or detection_ok):
            pipeline_success += 1

        overall_score = (
            (1.0 if detection_ok else 0.0) * 0.40
            + (float(np.mean([r.get("mask_quality", {}).get("quality_score", 0.0) for r in segment_records])) if segment_records else 0.0) * 0.30
            + (float(np.mean([r.get("color", {}).get("color_stability_score", 0.0) / 100.0 for r in segment_records])) if segment_records else 0.0) * 0.30
        )

        record = {
            "image_path": str(case.path),
            "weak_label": case.weak_label,
            "weak_label_source": case.weak_label_source,
            "silhouette": case.silhouette,
            "detections": [
                {
                    "label": det.label,
                    "confidence": round(float(det.confidence), 4),
                    "bbox_xyxy": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
                }
                for det in detections
            ],
            "detection_ok": detection_ok,
            "detection_status": detection_status,
            "segments": segment_records,
            "failures": sorted(image_failures),
            "overall_score": round(overall_score, 4),
        }
        records.append(record)

        for failure in image_failures:
            failure_counter[failure] += 1
            category = "detection" if failure in {"missed_detection", "wrong_class"} else ("segmentation" if failure == "poor_segmentation" else "color")
            _copy_failure(case, category, failures_dir, record)

    records.sort(key=lambda item: float(item["overall_score"]))
    for record in records[:20]:
        src = Path(record["image_path"])
        dest = worst_dir / f"{_safe_name(src)}{src.suffix.lower()}"
        shutil.copy2(src, dest)

    color_pass_rate = round((color_items - color_failures) / max(color_items, 1), 4)
    summary = {
        "dataset_size": len(cases),
        "weak_label_distribution": dict(weak_label_counter),
        "detection_accuracy_proxy": round((detection_evaluable - detection_failures) / max(detection_evaluable, 1), 4),
        "segmentation_success_rate": round((segmentation_items - segmentation_failures) / max(segmentation_items, 1), 4),
        # Backward-compatible name kept for existing consumers.
        "color_success_rate": color_pass_rate,
        # Preferred explicit name (this is a stability/heuristic pass metric, not GT accuracy).
        "color_stability_pass_rate": color_pass_rate,
        "pipeline_success_rate": round(pipeline_success / max(len(cases), 1), 4),
        "failure_counts": dict(failure_counter),
        "mean_mask_quality_score": round(float(np.mean(mask_scores)) if mask_scores else 0.0, 4),
        "mean_color_stability_score": round(float(np.mean(color_scores)) if color_scores else 0.0, 2),
        "mean_lab_drift": round(float(np.mean(lab_drifts)) if lab_drifts else 0.0, 4),
        "mean_hsv_improvement_pct": round(float(np.mean(hsv_improvements)) if hsv_improvements else 0.0, 2),
        "top_20_worst_images": [record["image_path"] for record in records[:20]],
    }

    (vision_dir / "cases.json").write_text(json.dumps([case.to_dict() for case in cases], indent=2), encoding="utf-8")
    (vision_dir / "vision_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (vision_dir / "vision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table_rows = [
        "image_path,weak_label,detection_status,failures,overall_score",
    ]
    for record in records:
        table_rows.append(
            f"\"{record['image_path']}\",{record['weak_label']},{record['detection_status']},\"{'|'.join(record['failures'])}\",{record['overall_score']}"
        )
    (vision_dir / "vision_records.csv").write_text("\n".join(table_rows) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "records": records,
    }
