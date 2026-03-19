from __future__ import annotations

import colorsys
from typing import Sequence

import numpy as np

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

_ANCHOR_CORRECTIONS = {
    "pink": [
        (lambda r, g, b: r > g + 40 and r > 200, "salmon"),
        (lambda r, g, b: r > 180 and g < 120, "hot pink"),
    ],
    "baby pink": [
        (lambda r, g, b: r > g + 35 and r > 190, "salmon"),
    ],
    "hot pink": [
        (lambda r, g, b: r > g + 40 and g > 120, "salmon"),
    ],
    "blue": [
        (lambda r, g, b: b < 120 and r < 80, "denim blue"),
    ],
    "gray": [
        (lambda r, g, b: r > 100 and (r - b) > 20, "beige"),
        (lambda r, g, b: b > r + 10, "slate"),
    ],
    "light gray": [
        (lambda r, g, b: r > 105 and (r - b) > 18, "beige"),
        (lambda r, g, b: b > r + 12, "slate"),
    ],
    "brown": [
        (lambda r, g, b: g > 140 and b < 100, "khaki"),
    ],
}


def remove_neutral_gray_background_pixels(
    rgb_pixels: np.ndarray,
    *,
    greyness_threshold: int = 20,
    brightness_threshold: int = 150,
    min_keep_ratio: float = 0.30,
) -> np.ndarray:
    """
    Remove likely neutral/gray bright background pixels from product-shot style images.

    We use channel-difference greyness + brightness threshold:
      greyness = |r-g| + |g-b| + |r-b|
      drop if greyness < threshold and r is bright.

    The fallback guard keeps original pixels if filtering is too aggressive.
    """
    if rgb_pixels.size == 0:
        return rgb_pixels

    rgb = np.asarray(rgb_pixels)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return rgb_pixels

    r = rgb[:, 0].astype(np.int32)
    g = rgb[:, 1].astype(np.int32)
    b = rgb[:, 2].astype(np.int32)
    greyness = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    keep = ~((greyness < int(greyness_threshold)) & (r > int(brightness_threshold)))
    filtered = rgb[keep]
    if filtered.shape[0] < max(120, int(rgb.shape[0] * float(min_keep_ratio))):
        return rgb_pixels
    return filtered


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


def dominant_hsv_label(rgb_pixels: np.ndarray) -> str:
    if rgb_pixels.size == 0:
        return "unknown"

    rgb = rgb_pixels.astype(np.float32) / 255.0
    if cv2 is not None:
        hsv = cv2.cvtColor((rgb.reshape(-1, 1, 3) * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        hue = hsv[:, 0] / 179.0
        sat = hsv[:, 1] / 255.0
        val = hsv[:, 2] / 255.0
    else:
        hue = np.zeros((rgb.shape[0],), dtype=np.float32)
        sat = np.zeros((rgb.shape[0],), dtype=np.float32)
        val = np.zeros((rgb.shape[0],), dtype=np.float32)
        for idx, (r, g, b) in enumerate(rgb):
            h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            hue[idx], sat[idx], val[idx] = h, s, v

    weights = np.clip(sat * 0.8 + val * 0.2, 0.05, 1.0)
    bins = 24
    hist = np.zeros((bins,), dtype=np.float32)
    for h, w in zip(hue, weights):
        bi = int(np.floor((h % 1.0) * bins)) % bins
        hist[bi] += float(w)
    peak = int(np.argmax(hist))
    mask = np.zeros_like(hue, dtype=bool)
    for offset in (-1, 0, 1):
        mask |= (((np.floor((hue % 1.0) * bins).astype(int) + bins) % bins) == ((peak + offset + bins) % bins))

    if not bool(mask.any()):
        mask = np.ones_like(hue, dtype=bool)
    h_mean = float(np.mod(np.angle(np.mean(np.exp(1j * 2.0 * np.pi * hue[mask]))) / (2.0 * np.pi), 1.0))
    s_mean = float(np.mean(sat[mask]))
    v_mean = float(np.mean(val[mask]))
    return _hsv_name(h_mean, s_mean, v_mean)


def pick_stable_color_label(
    *,
    lab_label: str,
    lab_confidence: float,
    palette: Sequence[str],
    hsv_label: str,
) -> str:
    """
    Deprecated compatibility shim.

    We now run LAB-first naming and explicitly avoid HSV-based override
    in the final label decision path.
    """
    _ = (lab_confidence, palette, hsv_label)
    if lab_label:
        return lab_label
    return "unknown"


def apply_anchor_corrections(label: str, rgb_triplet: tuple[int, int, int] | None) -> str:
    """
    Lightweight shade corrections for known unstable boundaries.
    Keeps outputs in our existing palette labels.
    """
    if not label or rgb_triplet is None:
        return label
    base = (label or "").strip().lower()
    checks = _ANCHOR_CORRECTIONS.get(base)
    if not checks:
        return label
    r, g, b = (int(rgb_triplet[0]), int(rgb_triplet[1]), int(rgb_triplet[2]))
    for condition, corrected in checks:
        try:
            if condition(r, g, b):
                return corrected
        except Exception:
            continue
    return label
