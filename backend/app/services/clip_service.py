from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from app.services.embedding_service import compute_image_embedding, get_clip_components

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]

_HF_MODEL = None
_HF_PROCESSOR = None
_HF_DEVICE = "cpu"


def _apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    mask_l = mask.convert("L")
    if mask_l.size != image.size:
        mask_l = mask_l.resize(image.size, resample=Image.Resampling.NEAREST)
    black = Image.new("RGB", image.size, (0, 0, 0))
    return Image.composite(image, black, mask_l)


def _ensure_hf_loaded() -> bool:
    global _HF_MODEL, _HF_PROCESSOR, _HF_DEVICE
    if _HF_MODEL is not None and _HF_PROCESSOR is not None:
        return True
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        return False
    try:
        _HF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _HF_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _HF_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _HF_MODEL.to(_HF_DEVICE)
        _HF_MODEL.eval()
        return True
    except Exception:
        _HF_MODEL = None
        _HF_PROCESSOR = None
        _HF_DEVICE = "cpu"
        return False


def _compute_hf_embedding(image_path: str, mask_path: Optional[str] = None) -> list[float]:
    if not _ensure_hf_loaded():
        raise RuntimeError("transformers CLIP backend unavailable")

    with Image.open(image_path) as im:
        image = im.convert("RGB")
    if mask_path:
        with Image.open(mask_path) as m:
            image = _apply_mask(image, m)

    inputs = _HF_PROCESSOR(images=image, return_tensors="pt")  # type: ignore[operator]
    inputs = {k: v.to(_HF_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():  # type: ignore[union-attr]
        emb = _HF_MODEL.get_image_features(**inputs)  # type: ignore[operator]
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    vec = emb.squeeze(0).detach().float().cpu().tolist()
    return [float(x) for x in vec]


def compute_clip_embedding(image_path: str, mask_path: Optional[str] = None) -> list[float]:
    """
    Compute a CLIP embedding for a garment image crop/mask.

    Backend selection:
    - CLIP_BACKEND=openai_clip (default): uses existing `embedding_service`
    - CLIP_BACKEND=transformers: uses HuggingFace CLIP
    - CLIP_BACKEND=auto: tries transformers then falls back to openai_clip
    """
    backend = (os.getenv("CLIP_BACKEND") or "openai_clip").strip().lower()
    if backend == "transformers":
        return _compute_hf_embedding(image_path, mask_path=mask_path)
    if backend == "auto":
        try:
            return _compute_hf_embedding(image_path, mask_path=mask_path)
        except Exception:
            pass
    return compute_image_embedding(image_path, mask_path=mask_path)


def compatibility_score(emb_a: list[float], emb_b: list[float]) -> float:
    if not emb_a or not emb_b:
        return 0.0
    a = np.asarray(emb_a, dtype=np.float32)
    b = np.asarray(emb_b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def preload_clip_models() -> tuple[bool, str]:
    """
    Preload CLIP backend at app startup to avoid first-request latency spikes.
    """
    backend = (os.getenv("CLIP_BACKEND") or "openai_clip").strip().lower()
    try:
        if backend == "transformers":
            ok = _ensure_hf_loaded()
            if not ok:
                return False, "transformers backend unavailable"
            return True, "transformers backend loaded"
        if backend == "auto":
            if _ensure_hf_loaded():
                return True, "transformers backend loaded (auto)"
            get_clip_components()
            return True, "openai_clip backend loaded (auto fallback)"

        # openai_clip default
        get_clip_components()
        return True, "openai_clip backend loaded"
    except Exception as e:
        return False, str(e)
