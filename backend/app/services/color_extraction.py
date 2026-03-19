# app/services/color_extraction.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import colorsys
import math
import numpy as np
from PIL import Image
import warnings
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from app.services.color_detector import apply_anchor_corrections, remove_neutral_gray_background_pixels
from app.services.color_labels import normalize_color_name

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ColorExtractionResult:
    primary_color: str
    palette: list[str]
    primary_confidence: float
    palette_confidences: list[float]


# ---------------------------------------------------------------------------
# LAB color anchors and display mapping
# ---------------------------------------------------------------------------

_COLOR_ANCHORS_RGB: dict[str, tuple[int, int, int]] = {
    # Reds / pinks
    "red": (200, 30, 30),
    "dark_red": (120, 10, 10),
    "maroon": (110, 25, 35),
    "pink": (230, 130, 150),
    "light_pink": (255, 200, 210),
    "hot_pink": (255, 105, 180),
    "dusty_rose": (196, 124, 145),
    "salmon": (250, 160, 140),
    # Oranges / yellows
    "orange": (220, 110, 30),
    "yellow": (220, 200, 40),
    # Greens
    "green": (40, 140, 60),
    "forest_green": (20, 80, 45),
    "dark_green": (10, 70, 25),
    "olive": (100, 110, 40),
    "teal_green": (25, 100, 70),
    # Blues
    "light_blue": (100, 160, 220),
    "denim_blue": (180, 205, 225),   # light-wash denim
    "powder_blue": (190, 215, 235),  # pale blue
    "blue": (30, 80, 190),
    "dark_denim": (30, 50, 90),
    "navy": (5, 10, 70),
    "dark_navy": (8, 12, 50),
    "indigo": (45, 55, 120),
    # Purples
    "purple": (128, 0, 128),
    "royal_purple": (90, 60, 150),
    "dark_purple": (70, 0, 90),
    "violet": (148, 0, 211),
    "lavender": (180, 150, 210),
    # Neutrals
    "brown": (120, 70, 40),
    "beige": (210, 190, 150),
    "tan": (210, 180, 140),
    "khaki": (195, 176, 145),
    "white": (240, 240, 240),
    "off_white": (232, 228, 218),
    "light_gray": (190, 190, 190),
    "gray": (128, 128, 128),
    "dark_gray": (70, 70, 70),
    "charcoal": (50, 55, 60),
    "slate": (70, 80, 90),
    "black": (20, 20, 20),
    # Teal / cyan
    "teal": (20, 140, 130),
    "dark_teal": (15, 80, 75),
}

_DISPLAY_NAME: dict[str, str] = {
    "dark_red": "burgundy",
    "dark_green": "dark green",
    "dark_gray": "dark gray",
    "dark_denim": "dark denim",
    "dark_navy": "dark navy",
    "dark_purple": "dark purple",
    "dark_teal": "dark teal",
    "denim_blue": "denim blue",
    "dusty_rose": "dusty rose",
    "forest_green": "forest green",
    "hot_pink": "hot pink",
    "indigo": "indigo",
    "khaki": "khaki",
    "lavender": "lavender",
    "light_blue": "sky blue",
    "light_gray": "light gray",
    "light_pink": "baby pink",
    "maroon": "maroon",
    "off_white": "off white",
    "powder_blue": "powder blue",
    "royal_purple": "royal purple",
    "salmon": "salmon",
    "slate": "slate",
    "tan": "tan",
    "teal_green": "sea green",
    "violet": "violet",
}


# ---------------------------------------------------------------------------
# RGB -> LAB helpers (pure numpy, D65)
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    def linearise(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin, g_lin, b_lin = linearise(r), linearise(g), linearise(b)

    # sRGB -> XYZ (D65)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Normalise by white point
    x /= 0.95047
    z /= 1.08883

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    return L, a, b_val


def _rgb_array_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    arr = rgb_array.astype(np.float32) / 255.0

    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ M.T
    xyz[:, 0] /= 0.95047
    xyz[:, 2] /= 1.08883

    def f_vec(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, t ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx = f_vec(xyz[:, 0])
    fy = f_vec(xyz[:, 1])
    fz = f_vec(xyz[:, 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=1).astype(np.float32)


_ANCHOR_NAMES: list[str] = list(_COLOR_ANCHORS_RGB.keys())
_ANCHOR_LAB: np.ndarray = np.array([_rgb_to_lab(v) for v in _COLOR_ANCHORS_RGB.values()], dtype=np.float32)


def _lab_to_color_name(
    lab: tuple[float, float, float],
    confidence_gap: float = 12.0,
) -> tuple[str, str, str, float, bool]:
    # Special-case near-neutral colors so we don't confuse dark grays/blacks with navy,
    # and keep whites/grays stable under different lighting / shadows.
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    chroma = math.sqrt(a * a + b * b)
    if chroma < 10.0:
        if L >= 92.0:
            return "white", "white", "white", 999.0, True
        if L >= 82.0:
            return "off white", "off_white", "white", 999.0, True
        if L <= 18.0:
            return "black", "black", "black", 999.0, True
        if L <= 32.0:
            return "charcoal", "charcoal", "dark_gray", 999.0, True
        if L <= 55.0:
            return "dark gray", "dark_gray", "gray", 999.0, True
        if L >= 72.0:
            return "light gray", "light_gray", "gray", 999.0, True
        return "gray", "gray", "gray", 999.0, True

    diffs = _ANCHOR_LAB - np.array(lab, dtype=np.float32)
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    order = np.argsort(dists)
    best_idx, second_idx = order[0], order[1]
    best_internal = _ANCHOR_NAMES[best_idx]
    second_internal = _ANCHOR_NAMES[second_idx]
    gap = float(dists[second_idx] - dists[best_idx])
    confident = gap >= confidence_gap

    best_display = _DISPLAY_NAME.get(best_internal, best_internal.replace("_", " "))
    second_display = _DISPLAY_NAME.get(second_internal, second_internal.replace("_", " "))

    # Expose a single label (compound labels were confusing in the UI).
    # Runner-up and gap are still returned for debugging.
    label = best_display

    return label, best_internal, second_internal, gap, confident


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _resize_max_side_bgr(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image_bgr
    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if cv2 is not None:
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = image_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb, mode="RGB")
    pil = pil.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    resized_rgb = np.array(pil, copy=True)
    return resized_rgb[:, :, ::-1]


def _load_image_bgr(path: str, max_side: int) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return _resize_max_side_bgr(img, max_side=max_side)
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            im = im.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        rgb = np.array(im, copy=True)
        return rgb[:, :, ::-1]


def _load_mask_bool(path: str, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    if cv2 is not None:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask at path: {path}")
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask > 0
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != (width, height):
            im = im.resize((width, height), resample=Image.Resampling.NEAREST)
        mask = np.array(im, copy=True)
        return mask > 0


# ---------------------------------------------------------------------------
# Pixel filtering
# ---------------------------------------------------------------------------

def _filter_garment_pixels_bgr(bgr_pixels: np.ndarray) -> np.ndarray:
    """
    Drop near-white, near-gray, and deep shadow pixels to reduce background bleed.
    """
    if bgr_pixels.size == 0:
        return bgr_pixels

    if cv2 is not None:
        reshaped = bgr_pixels.reshape(-1, 1, 3)
        hsv = cv2.cvtColor(reshaped, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        ycrcb = cv2.cvtColor(reshaped, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    else:
        # Fallback: convert each pixel to HSV via colorsys
        rgb = bgr_pixels[:, ::-1].astype(np.float32) / 255.0
        hsv_list = []
        for r, g, b in rgb:
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_list.append([h * 179.0, s * 255.0, v * 255.0])
        hsv = np.array(hsv_list, dtype=np.float32)
        ycrcb = None

    s = hsv[:, 1]
    v = hsv[:, 2]
    h_ch = hsv[:, 0].astype(np.float32)
    rgb = bgr_pixels[:, ::-1].astype(np.float32)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    not_white = ~((s < 30) & (v > 210))
    not_gray = ~((s < 22) & (v > 45) & (v < 215))
    not_deep_black = v >= 25

    skin_hsv = (
        (h_ch <= 30) &
        (
            ((s >= 10) & (s <= 200) & (v >= 35) & (v <= 255)) |  # typical skin
            ((s < 30) & (v >= 160)) |  # very pale skin
            ((s >= 10) & (v >= 20) & (v < 90))  # darker skin tones
        )
    )
    if ycrcb is not None:
        cr = ycrcb[:, 1].astype(np.float32)
        cb = ycrcb[:, 2].astype(np.float32)
        skin_ycrcb = (cr >= 130) & (cr <= 190) & (cb >= 80) & (cb <= 150)
    else:
        skin_ycrcb = np.zeros_like(skin_hsv, dtype=bool)

    skin_rgb = (
        (r > 45) & (g > 25) & (b > 15) &
        (r > g) & (g > b) &
        ((r - g) > 5) & ((r - b) > 10)
    )
    skin_mask = skin_hsv | skin_ycrcb | skin_rgb
    if float(skin_mask.mean()) < 0.05:
        skin_mask = np.zeros_like(skin_mask, dtype=bool)

    not_skin = ~skin_mask

    keep = not_white & not_gray & not_deep_black & not_skin
    filtered = bgr_pixels[keep]

    # Remove bright neutral gray backdrop pixels that can still survive
    # thresholding on product-shot style images.
    if filtered.size:
        rgb_filtered = filtered[:, ::-1]
        rgb_filtered = remove_neutral_gray_background_pixels(
            rgb_filtered,
            greyness_threshold=20,
            brightness_threshold=150,
            min_keep_ratio=0.35,
        )
        filtered = rgb_filtered[:, ::-1]

    # If we filtered too much (<10%), fall back to original
    if filtered.shape[0] < max(100, int(0.1 * bgr_pixels.shape[0])):
        return bgr_pixels
    return filtered


def _shrink_mask_for_color(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.size == 0 or not bool(mask_bool.any()) or cv2 is None:
        return mask_bool

    area = int(mask_bool.sum())
    erode_px = 1
    if area >= 2500:
        erode_px = 2
    if area >= 12000:
        erode_px = 3
    if area >= 40000:
        erode_px = 4

    kernel = np.ones((erode_px, erode_px), np.uint8)
    eroded = cv2.erode(mask_bool.astype(np.uint8) * 255, kernel, iterations=1) > 0
    if int(eroded.sum()) >= max(64, int(area * 0.35)):
        return eroded
    return mask_bool


def _erode_mask_interior(mask_bool: np.ndarray, erosion_px: int = 10) -> np.ndarray:
    if mask_bool.size == 0 or not bool(mask_bool.any()) or cv2 is None:
        return mask_bool
    px = max(1, int(erosion_px))
    kernel = np.ones((px, px), np.uint8)
    eroded = cv2.erode(mask_bool.astype(np.uint8) * 255, kernel, iterations=1) > 0
    if int(eroded.sum()) >= max(48, int(mask_bool.sum() * 0.18)):
        return eroded
    return mask_bool


def _center_weighted_sample_bgr(
    image_bgr: np.ndarray,
    selection: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    ys, xs = np.where(selection)
    if ys.size == 0:
        return image_bgr.reshape(-1, 3)

    n = int(ys.size)
    if n <= sample_size:
        return image_bgr[ys, xs]

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    distances = np.sqrt((ys.astype(np.float32) - cy) ** 2 + (xs.astype(np.float32) - cx) ** 2)
    max_dist = float(np.max(distances)) if distances.size else 1.0
    if max_dist <= 1e-6:
        probs = None
    else:
        weights = 1.0 - (distances / max_dist) * 0.7
        weights = np.clip(weights, 0.05, None)
        probs = weights / float(np.sum(weights))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False, p=probs)
    return image_bgr[ys[idx], xs[idx]]


def _mode_label_from_lab_histogram(sample_lab: np.ndarray, *, confidence_gap: float) -> tuple[str, tuple[int, int, int] | None]:
    if sample_lab.size == 0:
        return "unknown", None

    lab_u8 = np.clip(np.round(sample_lab), 0, 255).astype(np.uint8)
    l_bins = lab_u8[:, 0] // 8
    a_bins = lab_u8[:, 1] // 8
    b_bins = lab_u8[:, 2] // 8
    codes = (l_bins.astype(np.int32) * 1024) + (a_bins.astype(np.int32) * 32) + b_bins.astype(np.int32)
    counts = np.bincount(codes, minlength=32 * 32 * 32)
    mode_code = int(np.argmax(counts))

    l_mode = (mode_code // 1024) * 8 + 4
    a_mode = ((mode_code % 1024) // 32) * 8 + 4
    b_mode = (mode_code % 32) * 8 + 4
    mode_lab = (float(l_mode), float(a_mode), float(b_mode))
    mode_label, _i1, _i2, _gap, _conf = _lab_to_color_name(mode_lab, confidence_gap=confidence_gap)

    rgb_triplet: tuple[int, int, int] | None = None
    if cv2 is not None:
        lab_pixel = np.array([[[int(l_mode), int(a_mode), int(b_mode)]]], dtype=np.uint8)
        rgb = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
        rgb_triplet = tuple(int(x) for x in rgb[0, 0].tolist())
    return mode_label, rgb_triplet


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_dominant_colors(
    image_path: str,
    mask_path: Optional[str] = None,
    *,
    k: int = 5,
    sample_size: int = 5000,
    max_side: int = 256,
    seed: int = 42,
    min_cluster_weight: float = 0.08,
    confidence_gap: float = 12.0,
) -> ColorExtractionResult:
    """
    Dominant color extraction with LAB nearest-neighbor naming.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if sample_size < 50:
        raise ValueError("sample_size must be >= 50")
    if max_side < 32:
        raise ValueError("max_side must be >= 32")

    image_bgr = _load_image_bgr(image_path, max_side=max_side)
    height, width = image_bgr.shape[:2]

    mask_bool: Optional[np.ndarray] = None
    if mask_path:
        mask_bool = _load_mask_bool(mask_path, size_hw=(height, width))

    if mask_bool is not None:
        selection = _shrink_mask_for_color(mask_bool)
        selection = _erode_mask_interior(selection, erosion_px=10)
    else:
        selection = np.ones((height, width), dtype=bool)

    sample = _center_weighted_sample_bgr(
        image_bgr,
        selection,
        sample_size=max(sample_size * 2, 6000),
        seed=seed,
    )
    if sample.shape[0] < max(220, k):
        sample = image_bgr.reshape(-1, 3)

    # Filter background bleed before clustering.
    sample = _filter_garment_pixels_bgr(sample)
    if sample.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(sample.shape[0], size=sample_size, replace=False)
        sample = sample[idx]

    n_samples = int(sample.shape[0])
    k_eff = min(int(k), n_samples)
    if k_eff <= 0:
        return ColorExtractionResult(
            primary_color="unknown",
            palette=[],
            primary_confidence=0.0,
            palette_confidences=[],
        )

    sample_rgb = sample[:, ::-1].astype(np.float32)  # BGR -> RGB
    sample_lab = _rgb_array_to_lab(sample_rgb)
    mode_label, mode_rgb = _mode_label_from_lab_histogram(sample_lab, confidence_gap=confidence_gap)

    model = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = model.fit_predict(sample_lab)
    centers_lab = model.cluster_centers_

    counts = np.bincount(labels, minlength=k_eff).astype(np.float32)
    total = float(counts.sum()) if float(counts.sum()) > 0 else 1.0
    order = np.argsort(-counts)

    aggregated: dict[str, float] = {}

    for cluster_idx in order.tolist():
        frac = float(counts[cluster_idx] / total)
        if frac < min_cluster_weight:
            continue
        lab = tuple(float(x) for x in centers_lab[cluster_idx])
        label, _, _, _, _ = _lab_to_color_name(lab, confidence_gap=confidence_gap)
        aggregated[label] = aggregated.get(label, 0.0) + frac

    if not aggregated:
        lab = tuple(float(x) for x in centers_lab[order[0]])
        label, _, _, _, _ = _lab_to_color_name(lab, confidence_gap=confidence_gap)
        aggregated[label] = 1.0

    palette_sorted = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
    palette = [name for name, _ in palette_sorted]
    confidences = [float(frac) for _, frac in palette_sorted]

    primary_color = palette[0] if palette else "unknown"
    primary_confidence = confidences[0] if confidences else 0.0

    # If the most-common cluster is neutral, but there's a substantial non-neutral
    # cluster (common with beige/tan items under harsh lighting), prefer the
    # non-neutral label for UX.
    neutral_override_bases = {"white", "gray", "black", "beige"}
    if normalize_color_name(primary_color) in neutral_override_bases and primary_confidence < 0.80:
        for alt_name, alt_frac in palette_sorted[1:]:
            if normalize_color_name(alt_name) not in neutral_override_bases and float(alt_frac) >= 0.18:
                primary_color = alt_name
                primary_confidence = float(alt_frac)
                break

    # Multicolor print handling:
    # Only mark "multicolor" when we have multiple meaningful (non-neutral) colors.
    # Ignore white/gray/black which often appear due to highlights/shadows or minor mask bleed.
    if primary_confidence < 0.45 and len(palette_sorted) >= 3:
        non_neutral = [
            name
            for name, frac in palette_sorted
            if normalize_color_name(name) not in ("white", "gray", "black", "beige") and float(frac) >= 0.12
        ]
        if len(set(non_neutral)) >= 2:
            primary_color = "multicolor"
            primary_confidence = 1.0

    # Histogram LAB mode is often more robust than KMeans centroid averaging on
    # textured garments (e.g., navy with light stitching).
    mode_base = normalize_color_name(mode_label)
    primary_base = normalize_color_name(primary_color)
    if mode_label != "unknown":
        if primary_color == "multicolor":
            pass
        elif primary_confidence < 0.58:
            primary_color = mode_label
            primary_base = mode_base
        elif (
            mode_base != primary_base
            and mode_base not in {"white", "gray", "black"}
            and primary_base in {"white", "gray", "black", "beige"}
        ):
            primary_color = mode_label
            primary_base = mode_base

    primary_color = apply_anchor_corrections(primary_color, mode_rgb)

    return ColorExtractionResult(
        primary_color=primary_color,
        palette=palette,
        primary_confidence=round(primary_confidence, 4),
        palette_confidences=[round(c, 4) for c in confidences],
    )
