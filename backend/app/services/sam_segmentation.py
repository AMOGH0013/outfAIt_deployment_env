# app/services/sam_segmentation.py

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

_IMPORT_ERROR: Exception | None = None
_LOAD_ERROR: Exception | None = None

_PREDICTOR = None
_DEVICE = "cpu"

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e
else:
    try:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _REPO_ROOT = Path(__file__).resolve().parents[2]
        _CHECKPOINT = _REPO_ROOT / "sam_models" / "sam_vit_b.pth"
        if not _CHECKPOINT.exists():
            raise FileNotFoundError(f"SAM checkpoint not found at: {_CHECKPOINT}")

        sam = sam_model_registry["vit_b"](checkpoint=str(_CHECKPOINT))
        sam.to(device=_DEVICE)
        _PREDICTOR = SamPredictor(sam)
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = e


def _largest_component(mask_bool: np.ndarray) -> np.ndarray:
    """Keep largest connected component if OpenCV is available."""
    if cv2 is None:
        return mask_bool
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_bool
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, [largest], contourIdx=-1, color=255, thickness=cv2.FILLED)
    return filled > 0


def _postprocess_mask(mask_bool: np.ndarray) -> np.ndarray:
    """
    Clean up jagged/noisy masks and fill small holes.
    """
    if cv2 is None:
        return mask_bool
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    close_kernel = np.ones((5, 5), np.uint8)
    open_kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Fill holes via flood fill on inverted mask.
    inv = cv2.bitwise_not(opened)
    h, w = inv.shape[:2]
    flood = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 0)
    # After flood fill on the inverted mask, enclosed holes remain 255 while
    # border-connected background becomes 0. OR-ing fills holes without
    # turning the entire background into foreground.
    holes = flood
    cleaned = cv2.bitwise_or(opened, holes)
    return cleaned > 0


def _area_ratio(mask_bool: np.ndarray) -> float:
    return float(mask_bool.sum()) / float(mask_bool.size + 1e-6)


def _touches_all_borders(mask_bool: np.ndarray) -> bool:
    if mask_bool.size == 0:
        return False
    top = mask_bool[0, :].any()
    bottom = mask_bool[-1, :].any()
    left = mask_bool[:, 0].any()
    right = mask_bool[:, -1].any()
    return bool(top and bottom and left and right)


def _fallback_threshold_mask(image: np.ndarray) -> np.ndarray:
    """
    Background removal for bright or uniform backgrounds.

    - Prefer GrabCut when OpenCV is available (handles gray backgrounds better)
    - Fall back to mean-channel thresholding
    """
    if cv2 is not None:
        h, w = image.shape[:2]
        rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            mask_bool = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
            if not mask_bool[h // 2, w // 2]:
                mask_bool = ~mask_bool
            mask_bool = _largest_component(mask_bool)
            mask_bool = _postprocess_mask(mask_bool)
            area_ratio = _area_ratio(mask_bool)
            if 0.03 <= area_ratio <= 0.95:
                return (mask_bool.astype(np.uint8) * 255)
        except Exception:
            pass

    grayish = image.mean(axis=2)
    mask_bool = grayish < 235  # tighter threshold for gray backgrounds

    # If most vertical samples are excluded, the garment is likely darker than background; invert.
    h, w = mask_bool.shape
    center_ok = bool(mask_bool[h // 2, w // 2])
    top_ok = bool(mask_bool[int(h * 0.30), w // 2])
    bottom_ok = bool(mask_bool[int(h * 0.70), w // 2])
    if sum([center_ok, top_ok, bottom_ok]) < 2:
        mask_bool = ~mask_bool

    mask_bool = _largest_component(mask_bool)
    mask_bool = _postprocess_mask(mask_bool)
    return (mask_bool.astype(np.uint8) * 255)


def segment_clothing(image_path: str, output_mask_path: str) -> str:
    """
    Create a binary clothing mask using SAM.

    - Foreground prompt: tight 3x3 grid in the center (label=1)
      plus background corner/edge points (label=0)
    - Output: 8-bit grayscale PNG where 255 = clothing, 0 = background
    - Returns: output_mask_path
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "segment_anything/torch not available; install dependencies to enable SAM segmentation."
        ) from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("SAM model failed to load.") from _LOAD_ERROR
    if _PREDICTOR is None:
        raise RuntimeError("SAM predictor is not initialized.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            image = np.array(rgb, copy=True)
    except Exception as e:
        raise ValueError(f"Unable to read image: {image_path}") from e

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions")

    try:
        _PREDICTOR.set_image(image)
        fg_x = [0.35, 0.5, 0.65]
        fg_y = [0.40, 0.5, 0.60]
        fg_points = [[width * x, height * y] for y in fg_y for x in fg_x]
        bg_points = [
            [width * 0.03, height * 0.03],
            [width * 0.50, height * 0.03],
            [width * 0.97, height * 0.03],
            [width * 0.03, height * 0.50],
            [width * 0.97, height * 0.50],
            [width * 0.03, height * 0.97],
            [width * 0.50, height * 0.97],
            [width * 0.97, height * 0.97],
        ]
        point_coords = np.array(fg_points + bg_points, dtype=np.float32)
        point_labels = np.array([1] * len(fg_points) + [0] * len(bg_points), dtype=np.int32)
        masks, scores, _ = _PREDICTOR.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    except Exception as e:
        raise RuntimeError("SAM inference failed") from e

    if masks is None or len(masks) < 1:
        raise RuntimeError("SAM returned no masks")

    candidates = []
    for i, m in enumerate(masks):
        area = float(_area_ratio(m))
        score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        candidates.append((area, score, m))

    # Prefer masks that include the center and have reasonable area.
    valid = [
        c for c in candidates
        if c[2][height // 2, width // 2]
        and 0.03 <= c[0] <= 0.90
        and not _touches_all_borders(c[2])
    ]
    if valid:
        # Highest score among valid masks
        target = max(valid, key=lambda x: x[1])
    else:
        # Fall back to highest score within area bounds, else highest score overall
        area_ok = [c for c in candidates if 0.05 <= c[0] <= 0.85]
        target = max(area_ok, key=lambda x: x[1]) if area_ok else max(candidates, key=lambda x: x[1])

    area_ratio, _score, mask_bool = target

    center_ok = bool(mask_bool[height // 2, width // 2])
    top_ok = bool(mask_bool[int(height * 0.30), width // 2])
    bottom_ok = bool(mask_bool[int(height * 0.70), width // 2])
    if sum([center_ok, top_ok, bottom_ok]) < 2:
        mask_bool = ~mask_bool
        area_ratio = 1.0 - area_ratio

    # Remove skin pixels if they dominate the mask (helps when SAM grabs the person).
    if cv2 is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_ch = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)
        skin_hsv = (
            (h_ch <= 25)
            & (
                ((s >= 12) & (s <= 200) & (v >= 40) & (v <= 255))
                | ((s < 12) & (v >= 170))
                | ((s >= 20) & (v >= 20) & (v < 90))
            )
        )
        skin_ycrcb = (cr >= 135) & (cr <= 185) & (cb >= 85) & (cb <= 135)
        skin_mask = skin_hsv | skin_ycrcb
        if skin_mask.any():
            skin_ratio = float((mask_bool & skin_mask).sum()) / float(mask_bool.sum() + 1e-6)
            if skin_ratio > 0.15:
                mask_bool = mask_bool & (~skin_mask)
                area_ratio = float(_area_ratio(mask_bool))

    mask_bool = _largest_component(mask_bool)
    mask_bool = _postprocess_mask(mask_bool)

    # Recompute area after cleanup and guard against full-image masks.
    area_ratio = _area_ratio(mask_bool)
    if area_ratio < 0.05 or area_ratio > 0.90 or _touches_all_borders(mask_bool):
        mask = _fallback_threshold_mask(image)
    else:
        mask = (mask_bool.astype(np.uint8) * 255)

    out_dir = os.path.dirname(output_mask_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        Image.fromarray(mask, mode="L").save(output_mask_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write mask to: {output_mask_path}") from e

    return output_mask_path


def _sam_device() -> str:
    """For diagnostics/logging."""
    return _DEVICE


def segment_bbox_sam1(
    image_rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    pad_ratio: float = 0.20,
    erode_pixels: int = 3,
) -> np.ndarray:
    """Segment a garment inside a YOLO bbox using SAM v1 (segment-anything).

    This is used as a **fallback** when SAM2 is unavailable or fails.

    Returns a full-image boolean mask (H, W) where True = garment.

    Notes:
    - Uses a **box prompt** (not point prompt).
    - Expands bbox by `pad_ratio` to include some background for separation.
    - Rejects degenerate masks that cover the entire ROI.
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "segment_anything/torch not available; install dependencies to enable SAM segmentation."
        ) from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("SAM model failed to load.") from _LOAD_ERROR
    if _PREDICTOR is None:
        raise RuntimeError("SAM predictor is not initialized.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    height, width = image_rgb.shape[:2]

    def clip_xyxy(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(1, min(int(x2), width))
        y2 = max(1, min(int(y2), height))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return x1, y1, x2, y2

    def expand_xyxy(box: tuple[int, int, int, int], ratio: float) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad_x = int(round(w * ratio))
        pad_y = int(round(h * ratio))
        return clip_xyxy((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y))

    x1, y1, x2, y2 = clip_xyxy(bbox_xyxy)
    roi_x1, roi_y1, roi_x2, roi_y2 = expand_xyxy((x1, y1, x2, y2), pad_ratio)

    roi = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        raise ValueError("Empty ROI for SAM1 bbox segmentation")

    # Box prompt in ROI coordinates
    box = np.array([x1 - roi_x1, y1 - roi_y1, x2 - roi_x1, y2 - roi_y1], dtype=np.float32)

    _PREDICTOR.set_image(roi)

    try:
        masks, scores, _ = _PREDICTOR.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True,
        )
    except Exception as e:
        raise RuntimeError("SAM1 bbox inference failed") from e

    if masks is None or len(masks) < 1:
        raise RuntimeError("SAM1 returned no masks")

    cx = int(round((x1 + x2) / 2.0)) - roi_x1
    cy = int(round((y1 + y2) / 2.0)) - roi_y1
    cx = max(0, min(cx, roi.shape[1] - 1))
    cy = max(0, min(cy, roi.shape[0] - 1))

    candidates: list[tuple[float, float, np.ndarray]] = []
    for i, m in enumerate(masks):
        m_bool = m.astype(bool)
        score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        area = _area_ratio(m_bool)
        candidates.append((score, area, m_bool))

    # Prefer masks that include the center of the bbox and have reasonable area.
    valid = [
        c for c in candidates
        if c[2][cy, cx]
        and 0.02 <= c[1] <= 0.95
        and not _touches_all_borders(c[2])
    ]
    if valid:
        _score, _area, mask_roi = max(valid, key=lambda x: x[0])
    else:
        area_ok = [c for c in candidates if 0.05 <= c[1] <= 0.90]
        _score, _area, mask_roi = max(area_ok, key=lambda x: x[0]) if area_ok else max(candidates, key=lambda x: x[0])

    mask_roi = _largest_component(mask_roi)
    mask_roi = _postprocess_mask(mask_roi)

    if cv2 is not None and erode_pixels > 0:
        k = max(1, int(erode_pixels))
        kernel = np.ones((k, k), np.uint8)
        eroded = cv2.erode(mask_roi.astype(np.uint8) * 255, kernel, iterations=1)
        mask_roi = eroded > 0

    area = _area_ratio(mask_roi)
    if area < 0.02:
        raise RuntimeError("SAM1 produced tiny/empty ROI mask")
    if _touches_all_borders(mask_roi) and area > 0.85:
        raise RuntimeError("SAM1 produced degenerate full-ROI mask")

    full = np.zeros((height, width), dtype=bool)
    full[roi_y1:roi_y2, roi_x1:roi_x2] = mask_roi
    return full
