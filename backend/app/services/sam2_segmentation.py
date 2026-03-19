# app/services/sam2_segmentation.py

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Sam2MaskResult:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask_bool: np.ndarray  # HxW bool in full-image coordinates


def _clip_xyxy(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
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


def _expand_xyxy(box: tuple[int, int, int, int], width: int, height: int, pad_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(round(w * pad_ratio))
    pad_y = int(round(h * pad_ratio))
    return _clip_xyxy((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _area_ratio(mask_bool: np.ndarray) -> float:
    return float(mask_bool.sum()) / float(mask_bool.size + 1e-6)


def _touches_all_borders(mask_bool: np.ndarray) -> bool:
    if mask_bool.size == 0:
        return False
    top = bool(mask_bool[0, :].any())
    bottom = bool(mask_bool[-1, :].any())
    left = bool(mask_bool[:, 0].any())
    right = bool(mask_bool[:, -1].any())
    return bool(top and bottom and left and right)


def _mask_bbox_xyxy(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return tight bbox of True pixels in ROI coordinates, or None if empty."""
    if mask_bool.size == 0 or not bool(mask_bool.any()):
        return None
    ys, xs = np.where(mask_bool)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def _bbox_fill_ratio(mask_bool: np.ndarray) -> float:
    """Area(mask) / Area(tight bbox).  Rectangle-like masks approach 1.0."""
    bbox = _mask_bbox_xyxy(mask_bool)
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    bbox_area = float(max(1, x2 - x1) * max(1, y2 - y1))
    return float(mask_bool.sum()) / bbox_area


def _is_boxy_mask(mask_bool: np.ndarray, *, min_area: float = 0.20, fill_thresh: float = 0.985) -> bool:
    """Heuristic: reject masks that are essentially a filled rectangle."""
    area = _area_ratio(mask_bool)
    if area < min_area:
        return False
    fill = _bbox_fill_ratio(mask_bool)
    if fill < fill_thresh:
        return False

    # If it also hugs ROI borders, it's almost certainly a prompt-box echo.
    top = bool(mask_bool[0, :].any())
    bottom = bool(mask_bool[-1, :].any())
    left = bool(mask_bool[:, 0].any())
    right = bool(mask_bool[:, -1].any())
    border_hits = int(top) + int(bottom) + int(left) + int(right)
    return border_hits >= 2


def _largest_component(mask_bool: np.ndarray) -> np.ndarray:
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
    """Close small holes and remove speckle."""
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
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 0)
    # After flood fill on the inverted mask:
    # - border-connected background becomes 0
    # - enclosed holes remain 255
    # OR-ing the holes back into the original mask fills them.
    holes = flood
    cleaned = cv2.bitwise_or(opened, holes)
    return cleaned > 0


def _erode_mask(mask_bool: np.ndarray, pixels: int = 3) -> np.ndarray:
    if cv2 is None or pixels <= 0:
        return mask_bool
    k = max(1, int(pixels))
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(mask_bool.astype(np.uint8) * 255, kernel, iterations=1)
    return eroded > 0


def _find_sam2_cfg() -> Path:
    env = (os.getenv("SAM2_MODEL_CFG") or "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"SAM2_MODEL_CFG points to missing file: {p}")
        return p

    # Try to locate configs shipped inside the installed sam2 package.
    try:
        import sam2  # type: ignore

        pkg_root = Path(sam2.__file__).resolve().parent
        candidates = [
            pkg_root / "configs" / "sam2" / "sam2_hiera_l.yaml",
            pkg_root / "configs" / "sam2" / "sam2_hiera_large.yaml",
            pkg_root / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
        ]
        for c in candidates:
            if c.exists():
                return c

        hits = list(pkg_root.rglob("*hiera_l*.yaml"))
        if hits:
            return hits[0]
    except Exception:
        pass

    raise FileNotFoundError(
        "SAM2 config YAML not found. Set SAM2_MODEL_CFG to a sam2_hiera_*.yaml config path."
    )


try:
    import torch
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e
else:
    def _remap_sam2_state_dict(sd: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        """
        Some SAM2 checkpoints (including older/local exports) use legacy key prefixes:
          - transformer.encoder.*      -> memory_attention.*
          - maskmem_backbone.*         -> memory_encoder.*

        The current `sam2` package expects the newer names from the YAML configs
        shipped with the library. We remap here so the existing checkpoint can
        load cleanly without forcing users to re-download weights.
        """
        out: dict[str, "torch.Tensor"] = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("transformer.encoder."):
                nk = "memory_attention." + nk[len("transformer.encoder.") :]
            if nk.startswith("maskmem_backbone."):
                nk = "memory_encoder." + nk[len("maskmem_backbone.") :]
            out[nk] = v
        return out

    try:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _REPO_ROOT = Path(__file__).resolve().parents[2]
        ckpt_env = (os.getenv("SAM2_CHECKPOINT_PATH") or "").strip()
        checkpoint = Path(ckpt_env) if ckpt_env else (_REPO_ROOT / "sam2_hiera_large.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at: {checkpoint}. Set SAM2_CHECKPOINT_PATH or place sam2_hiera_large.pt in repo root."
            )

        model_cfg = _find_sam2_cfg()

        # Build the architecture first; then load weights with a key-remap pass.
        # (The stock `build_sam2(..., ckpt_path=...)` loads strictly and will
        # fail if the checkpoint uses legacy prefixes.)
        sam2_model = build_sam2(str(model_cfg), ckpt_path=None, device=_DEVICE)
        ckpt_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("model"), dict):
            sd = ckpt_obj["model"]
        elif isinstance(ckpt_obj, dict):
            sd = ckpt_obj
        else:
            raise RuntimeError("Unexpected SAM2 checkpoint format (expected a dict).")

        sd = _remap_sam2_state_dict(sd)
        missing, unexpected = sam2_model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"SAM2 checkpoint mismatch after remap: missing={len(missing)} unexpected={len(unexpected)}"
            )

        _PREDICTOR = SAM2ImagePredictor(sam2_model)
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = e


def segment_bbox_sam2(
    image_rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    pad_ratio: float = 0.20,
    erode_pixels: int = 3,
) -> np.ndarray:
    """Segment a garment inside a YOLO bbox using SAM2.

    Returns a full-image boolean mask (H, W) where True = garment.

    Strategy:
      - Expand bbox by `pad_ratio` to include a little background (critical)
      - Use SAM2 **box prompt** + a few background points (corners) to avoid
        the common "full ROI rectangle" failure mode on product photos
      - Reject degenerate masks that cover the entire ROI (common failure mode)
      - Postprocess (largest component + morphology + hole fill) + erode edges
    """
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "sam2/torch not available; install Segment Anything 2 to enable SAM2 segmentation."
        ) from _IMPORT_ERROR
    if _LOAD_ERROR is not None:
        raise RuntimeError("SAM2 model failed to load.") from _LOAD_ERROR
    if _PREDICTOR is None:
        raise RuntimeError("SAM2 predictor is not initialized.")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(bbox_xyxy, width=width, height=height)
    roi_x1, roi_y1, roi_x2, roi_y2 = _expand_xyxy((x1, y1, x2, y2), width=width, height=height, pad_ratio=pad_ratio)

    roi = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        raise ValueError("Empty ROI for segmentation")

    # Box prompt in ROI coordinates
    box = np.array([x1 - roi_x1, y1 - roi_y1, x2 - roi_x1, y2 - roi_y1], dtype=np.float32)

    # Add a robust point prompt set:
    # - foreground points sampled *inside the box* that are far from the ROI background color
    #   (prevents pants failing when the bbox-center lands between the legs / on background)
    # - background points at ROI corners (helps separate garment from background)
    h, w = roi.shape[:2]
    margin = max(2, int(round(min(h, w) * 0.02)))

    def clamp_point(x: int, y: int) -> tuple[int, int]:
        return (max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1)))

    # Background color estimate from padded ROI corners.
    corner_pts = [
        clamp_point(margin, margin),
        clamp_point(w - 1 - margin, margin),
        clamp_point(margin, h - 1 - margin),
        clamp_point(w - 1 - margin, h - 1 - margin),
    ]
    bg = np.median(np.array([roi[y, x] for (x, y) in corner_pts], dtype=np.float32), axis=0)

    # Candidate FG points on a small grid inside the prompt box (ROI coords).
    bx1, by1, bx2, by2 = [float(v) for v in box.tolist()]
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    # Avoid extreme edges of the box where background bleed is common.
    inset_x = max(2.0, bw * 0.08)
    inset_y = max(2.0, bh * 0.08)
    xs = [bx1 + inset_x + bw * f for f in (0.18, 0.50, 0.82)]
    ys = [by1 + inset_y + bh * f for f in (0.30, 0.60, 0.85)]

    candidates: list[tuple[float, tuple[int, int]]] = []
    for xf in xs:
        for yf in ys:
            x, y = clamp_point(int(round(xf)), int(round(yf)))
            rgb = roi[y, x].astype(np.float32)
            dist = float(np.linalg.norm(rgb - bg))
            candidates.append((dist, (x, y)))

    # Select top-N unique FG points.
    candidates.sort(key=lambda t: t[0], reverse=True)
    fg_points: list[tuple[int, int]] = []
    for dist, pt in candidates:
        if pt in fg_points:
            continue
        fg_points.append(pt)
        if len(fg_points) >= 3:
            break

    # Center-of-box: only add as foreground if it doesn't look like background.
    # (For pants, the center often lands between the legs; forcing it as FG makes SAM2
    # return a big background-filled mask.)
    cx = int(round((bx1 + bx2) / 2.0))
    cy = int(round((by1 + by2) / 2.0))
    center_pt = clamp_point(cx, cy)
    center_rgb = roi[center_pt[1], center_pt[0]].astype(np.float32)
    center_dist = float(np.linalg.norm(center_rgb - bg))

    bg_points: list[tuple[int, int]] = list(corner_pts)
    if center_dist >= 18.0:
        if center_pt not in fg_points:
            fg_points.append(center_pt)
    else:
        # If center looks like background, add it as an explicit BG point to prevent
        # the model from filling the leg-gap / background regions.
        if center_pt not in bg_points:
            bg_points.append(center_pt)

    # Build prompt arrays: FG points + BG corner points.
    point_coords = np.array([*fg_points, *bg_points], dtype=np.float32)
    point_labels = np.array([*([1] * len(fg_points)), *([0] * len(bg_points))], dtype=np.int32)

    fg_count = len(fg_points)

    _PREDICTOR.set_image(roi)

    def run_predict(use_points: bool):
        try:
            if use_points:
                return _PREDICTOR.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )
            return _PREDICTOR.predict(
                box=box,
                multimask_output=True,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("SAM2 inference failed") from e

    masks = scores = None
    last_err: Exception | None = None
    for use_points in (True, False):
        try:
            masks, scores, _ = run_predict(use_points=use_points)
        except Exception as e:
            last_err = e
            continue
        if masks is None or len(masks) < 1:
            last_err = RuntimeError("SAM2 returned no masks")
            continue

        candidates_masks: list[tuple[int, float, float, float, np.ndarray]] = []
        for i, m in enumerate(masks):
            m_bool = m.astype(bool) if m.dtype == bool else (m > 0)
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            area = _area_ratio(m_bool)
            fill = _bbox_fill_ratio(m_bool)
            # Count how many FG points land inside the mask.
            hits = 0
            for (px, py) in fg_points:
                if m_bool[py, px]:
                    hits += 1
            candidates_masks.append((hits, score, area, fill, m_bool))

        # Prefer masks that cover multiple FG points, are not gigantic, and are not "boxy".
        valid = [
            c for c in candidates_masks
            if c[2] >= 0.02 and c[2] <= 0.95
            and c[0] >= 1
            and not _is_boxy_mask(c[4])
        ]
        if not valid:
            # Relax FG hit requirement if everything is messy; still avoid boxy rectangles.
            valid = [c for c in candidates_masks if c[2] >= 0.02 and c[2] <= 0.95 and not _is_boxy_mask(c[4])]

        if valid:
            hits, _score, _area, _fill, mask_roi = max(valid, key=lambda x: (x[0], x[1], -x[2]))
        else:
            # As a last resort, pick the highest-score mask and let postprocessing try.
            hits, _score, _area, _fill, mask_roi = max(candidates_masks, key=lambda x: x[1])

        mask_roi = _largest_component(mask_roi)
        mask_roi = _postprocess_mask(mask_roi)
        mask_roi = _erode_mask(mask_roi, pixels=erode_pixels)

        # Reject degenerate masks (too small OR essentially a rectangle).
        area = _area_ratio(mask_roi)
        if area < 0.02:
            last_err = RuntimeError("SAM2 produced tiny/empty ROI mask")
            continue
        if _touches_all_borders(mask_roi) and area > 0.85:
            last_err = RuntimeError("SAM2 produced degenerate full-ROI mask")
            continue
        if _is_boxy_mask(mask_roi, min_area=0.25, fill_thresh=0.99):
            last_err = RuntimeError("SAM2 produced degenerate box-like mask")
            continue

        full = np.zeros((height, width), dtype=bool)
        full[roi_y1:roi_y2, roi_x1:roi_x2] = mask_roi
        return full

    if last_err is not None:
        raise last_err
    raise RuntimeError("SAM2 segmentation failed")


def save_mask_png(mask_bool: np.ndarray, output_mask_path: str) -> str:
    out_dir = os.path.dirname(output_mask_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8, mode="L").save(output_mask_path)
    return output_mask_path


def load_image_rgb(image_path: str) -> np.ndarray:
    try:
        with Image.open(image_path) as im:
            return np.array(im.convert("RGB"), copy=True)
    except Exception as e:
        raise ValueError(f"Unable to read image: {image_path}") from e


def _sam2_device() -> str:
    return _DEVICE
