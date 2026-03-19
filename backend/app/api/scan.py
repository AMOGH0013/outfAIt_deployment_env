# app/api/scan.py

from __future__ import annotations

import os
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.dependencies import get_current_user, get_db
from app.models.scan import ScanSession
from app.models.user import User
from app.models.wardrobe import WardrobeItem
from app.services.clip_service import compute_clip_embedding
from app.services.color_extraction import extract_dominant_colors
from app.services.dev_event_log import log_user_event
from app.services.item_type_suggester import suggest_item_types
from app.services.sam2_segmentation import load_image_rgb, save_mask_png, segment_bbox_sam2
from app.services.sam_segmentation import segment_bbox_sam1, segment_clothing
from app.services.yolo_detection import YoloDetection, detect_upper_lower

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads") or "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
TOP_ITEM_TYPES = {"shirt", "tshirt", "kurta", "hoodie"}
BOTTOM_ITEM_TYPES = {"jeans", "trousers", "chinos", "shorts"}

router = APIRouter(prefix="/scan", tags=["Scan"])


def _to_public_upload_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("/uploads/") or value.startswith("http://") or value.startswith("https://"):
        return value
    filename = os.path.basename(str(value).replace("\\", "/"))
    if not filename:
        return value
    return f"/uploads/{filename}"


def _auto_item_type_from_suggestions(
    suggestions: list[dict] | None,
    category: str,
    *,
    min_confidence: float = 0.35,
) -> tuple[str, float | None]:
    if not suggestions:
        return "unknown", None
    allowed = TOP_ITEM_TYPES if category == "top" else BOTTOM_ITEM_TYPES
    for s in suggestions:
        item_type = str((s or {}).get("item_type") or "").strip().lower()
        try:
            conf = float((s or {}).get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        if item_type in allowed and conf >= min_confidence:
            return item_type, conf
    return "unknown", None


@router.post("/start")
def start_scan(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = ScanSession(user_id=current_user.id, status="pending")
    db.add(scan)
    db.commit()
    db.refresh(scan)

    log_user_event(user_id=current_user.id, event="scan_start", meta={"scan_id": str(scan.id)})
    return {"scan_id": scan.id, "status": scan.status}


@router.post("/upload/{scan_id}")
async def upload_scan_image(
    scan_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = (
        db.query(ScanSession)
        .filter(
            ScanSession.id == scan_id,
            ScanSession.user_id == current_user.id,
        )
        .first()
    )

    if not scan:
        raise HTTPException(status_code=404, detail="Scan session not found")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large",
        )

    original_name = os.path.basename(file.filename or "upload")
    filename = f"{uuid4()}_{original_name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    public_url = f"/uploads/{filename}"

    base_name, _ext = os.path.splitext(filename)

    warnings: list[str] = []
    created_ids: list[UUID] = []
    items_out: list[dict] = []

    def _safe_extract_colors(mask_path: str | None, label: str):
        try:
            return extract_dominant_colors(filepath, mask_path=mask_path)
        except Exception as e:
            warnings.append(f"color_failed_{label}_fallback_full")
            log_user_event(
                user_id=current_user.id,
                event="scan_color_fallback",
                meta={"scan_id": str(scan.id), "label": label, "error": str(e)},
            )
            return extract_dominant_colors(filepath, mask_path=None)

    def _safe_embedding_and_suggestions(mask_path: str | None, label: str):
        embedding = None
        suggestions: list[dict] = []
        try:
            embedding = compute_clip_embedding(filepath, mask_path=mask_path)
        except Exception as e:
            warnings.append(f"embedding_unavailable_{label}")
            log_user_event(
                user_id=current_user.id,
                event="scan_embedding_unavailable",
                meta={"scan_id": str(scan.id), "label": label, "error": str(e)},
            )
            return None, []

        try:
            suggestions = suggest_item_types(embedding, top_k=3)
        except Exception as e:
            warnings.append(f"item_type_suggestion_unavailable_{label}")
            log_user_event(
                user_id=current_user.id,
                event="scan_item_type_suggester_unavailable",
                meta={"scan_id": str(scan.id), "label": label, "error": str(e)},
            )
            suggestions = []

        return embedding, suggestions

    try:
        with open(filepath, "wb") as buffer:
            buffer.write(content)

        scan.image_url = public_url
        scan.status = "processing"
        db.add(scan)
        db.flush()

        # --- New pipeline ---
        # Upload image -> YOLO detector (best_stage2_FINAL.pt fallback-safe) upper/lower -> Crop ROI(s) -> SAM2 -> postprocess -> LAB+KMeans
        image_rgb = load_image_rgb(filepath)
        img_h, img_w = image_rgb.shape[:2]
        global_fg_mask: np.ndarray | None = None

        try:
            patch = max(12, min(28, min(img_h, img_w) // 10))
            tl = image_rgb[:patch, :patch, :].reshape(-1, 3)
            tr = image_rgb[:patch, -patch:, :].reshape(-1, 3)
            bl = image_rgb[-patch:, :patch, :].reshape(-1, 3)
            br = image_rgb[-patch:, -patch:, :].reshape(-1, 3)
            corners = np.concatenate([tl, tr, bl, br], axis=0).astype(np.float32)
            corners_std = float(corners.std(axis=0).mean())
            if corners_std < 32.0:
                bg = np.median(corners, axis=0)
                dist = np.linalg.norm(image_rgb.astype(np.float32) - bg[None, None, :], axis=2)
                fg_mask = dist > 24.0
                if float(fg_mask.mean()) >= 0.01:
                    global_fg_mask = fg_mask
        except Exception:
            global_fg_mask = None

        detections: list[YoloDetection] = []
        try:
            # Lower the confidence a bit so "lower" garments (pants/skirts) are not missed.
            detections = detect_upper_lower(filepath, conf=0.20, max_det=20)
        except Exception as e:
            warnings.append("yolo_unavailable")
            log_user_event(
                user_id=current_user.id,
                event="scan_yolo_unavailable",
                meta={"scan_id": str(scan.id), "error": str(e)},
            )

        # Single-garment mode:
        # user uploads one clothing item at a time, so we must return only one
        # detection (either upper or lower), never synthesize a second class.
        best: dict[str, YoloDetection] = {}

        img_area = float(img_w * img_h + 1e-6)

        def area_frac(d: YoloDetection) -> float:
            w = max(1, int(d.x2) - int(d.x1))
            h = max(1, int(d.y2) - int(d.y1))
            return float(w * h) / img_area

        def height_frac(d: YoloDetection) -> float:
            return float(max(1, int(d.y2) - int(d.y1))) / float(img_h + 1e-6)

        def center_y_frac(d: YoloDetection) -> float:
            return (float(d.y1) + float(d.y2)) * 0.5 / float(img_h + 1e-6)

        def bbox_iou(a: YoloDetection, b: YoloDetection) -> float:
            ix1 = max(int(a.x1), int(b.x1))
            iy1 = max(int(a.y1), int(b.y1))
            ix2 = min(int(a.x2), int(b.x2))
            iy2 = min(int(a.y2), int(b.y2))
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = float((ix2 - ix1) * (iy2 - iy1))
            area_a = float(max(1, int(a.x2) - int(a.x1)) * max(1, int(a.y2) - int(a.y1)))
            area_b = float(max(1, int(b.x2) - int(b.x1)) * max(1, int(b.y2) - int(b.y1)))
            return inter / max(1e-6, area_a + area_b - inter)

        def single_item_rank(d: YoloDetection) -> float:
            af = area_frac(d)
            hf = height_frac(d)
            if af < 0.015 or hf < 0.10:
                return -1.0
            cy = center_y_frac(d)
            # Slight center preference for flat-lay single-item uploads.
            center_bonus = max(0.0, 0.22 - abs(cy - 0.50))
            return float(d.confidence) * (1.0 + af) + center_bonus

        filtered = [d for d in detections if single_item_rank(d) > 0.0]
        if filtered:
            filtered.sort(key=single_item_rank, reverse=True)
            winner = filtered[0]
            if len(filtered) > 1:
                second = filtered[1]
                if (
                    winner.label != second.label
                    and bbox_iou(winner, second) >= 0.55
                    and abs(float(winner.confidence) - float(second.confidence)) <= 0.12
                ):
                    # Overlapping conflicting labels with similar confidence:
                    # prefer upper on ties for shirt-like ambiguity, else higher confidence.
                    if float(second.confidence) > float(winner.confidence):
                        winner = second
                    elif abs(float(second.confidence) - float(winner.confidence)) <= 1e-6 and second.label == "upper":
                        winner = second
                    warnings.append("single_item_overlap_conflict_resolved")
            best[winner.label] = winner

        if not best:
            # Fallback: SAM v1 full-image segmentation (keeps app usable)
            warnings.append("yolo_no_detections_using_sam1_full")

            mask_filename = f"{base_name}_mask.png"
            mask_filepath = os.path.join(UPLOAD_DIR, mask_filename)
            mask_public_url = f"/uploads/{mask_filename}"

            segment_clothing(filepath, mask_filepath)
            colors = _safe_extract_colors(mask_filepath, label="full")
            embedding, suggestions = _safe_embedding_and_suggestions(mask_filepath, label="full")

            top_suggestion = suggestions[0] if suggestions else None
            auto_item_type, auto_conf = _auto_item_type_from_suggestions(suggestions, category="top")
            item = WardrobeItem(
                user_id=current_user.id,
                image_url=public_url,
                mask_url=mask_public_url,
                item_type=auto_item_type,
                category="top",
                color=colors.primary_color,
                color_palette=colors.palette,
                embedding=embedding,
                suggested_item_type=(top_suggestion["item_type"] if top_suggestion else None),
                suggested_item_type_confidence=(top_suggestion["confidence"] if top_suggestion else None),
                confidence_scores={
                    "segmentation": "sam1_full",
                    "color": colors.primary_confidence,
                    "color_palette": dict(zip(colors.palette, colors.palette_confidences)),
                    "auto_item_type_confidence": auto_conf,
                },
            )

            db.add(item)
            db.flush()
            created_ids.append(item.id)

            items_out.append(
                {
                    "wardrobe_item_id": str(item.id),
                    "label": "full",
                    "category": item.category,
                    "segmentation": "sam1_full",
                    "mask_url": mask_public_url,
                    "extracted_color": colors.primary_color,
                    "color_palette": colors.palette,
                    "color_confidence": colors.primary_confidence,
                    "item_type": item.item_type,
                    "item_type_suggestions": suggestions,
                    "yolo_confidence": None,
                    "bbox_xyxy": None,
                }
            )
        else:
            # Build per-detection masks and items
            for label in ("upper", "lower"):
                det = best.get(label)
                if det is None:
                    continue

                # det is YoloDetection
                x1 = int(det.x1)  # type: ignore[attr-defined]
                y1 = int(det.y1)  # type: ignore[attr-defined]
                x2 = int(det.x2)  # type: ignore[attr-defined]
                y2 = int(det.y2)  # type: ignore[attr-defined]
                yolo_conf = float(det.confidence)  # type: ignore[attr-defined]

                # Clip bbox
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(1, min(x2, img_w))
                y2 = max(1, min(y2, img_h))

                mask_filename = f"{base_name}_{label}_mask.png"
                mask_filepath = os.path.join(UPLOAD_DIR, mask_filename)
                mask_public_url = f"/uploads/{mask_filename}"

                seg_backend = "sam2"
                try:
                    mask_bool = segment_bbox_sam2(image_rgb, (x1, y1, x2, y2))
                except Exception as e:
                    warnings.append(f"sam2_failed_{label}")
                    log_user_event(
                        user_id=current_user.id,
                        event="scan_sam2_failed",
                        meta={
                            "scan_id": str(scan.id),
                            "label": label,
                            "error": str(e),
                        },
                    )

                    # Fallback 1: SAM v1 box-prompt inside bbox
                    try:
                        mask_bool = segment_bbox_sam1(image_rgb, (x1, y1, x2, y2))
                        seg_backend = "sam1"
                        warnings.append(f"sam1_used_{label}")
                    except Exception as e2:
                        warnings.append(f"sam1_failed_{label}")
                        log_user_event(
                            user_id=current_user.id,
                            event="scan_sam1_bbox_failed",
                            meta={
                                "scan_id": str(scan.id),
                                "label": label,
                                "error": str(e2),
                            },
                        )
                        # Last resort: bbox rectangle mask (keeps pipeline alive)
                        mask_bool = np.zeros((img_h, img_w), dtype=bool)
                        mask_bool[y1:y2, x1:x2] = True
                        seg_backend = "rect"

                save_mask_png(mask_bool, mask_filepath)

                colors = _safe_extract_colors(mask_filepath, label=label)
                embedding, suggestions = _safe_embedding_and_suggestions(mask_filepath, label=label)

                top_suggestion = suggestions[0] if suggestions else None
                category = "top" if label == "upper" else "bottom"
                auto_item_type, auto_conf = _auto_item_type_from_suggestions(suggestions, category=category)

                item = WardrobeItem(
                    user_id=current_user.id,
                    image_url=public_url,
                    mask_url=mask_public_url,
                    item_type=auto_item_type,
                    category=category,
                    color=colors.primary_color,
                    color_palette=colors.palette,
                    embedding=embedding,
                    suggested_item_type=(top_suggestion["item_type"] if top_suggestion else None),
                    suggested_item_type_confidence=(top_suggestion["confidence"] if top_suggestion else None),
                    confidence_scores={
                        "segmentation": seg_backend,
                        "yolo": yolo_conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "color": colors.primary_confidence,
                        "color_palette": dict(zip(colors.palette, colors.palette_confidences)),
                        "auto_item_type_confidence": auto_conf,
                    },
                )

                db.add(item)
                db.flush()
                created_ids.append(item.id)

                items_out.append(
                    {
                        "wardrobe_item_id": str(item.id),
                        "label": label,
                        "category": category,
                        "segmentation": seg_backend,
                        "mask_url": mask_public_url,
                        "extracted_color": colors.primary_color,
                        "color_palette": colors.palette,
                        "color_confidence": colors.primary_confidence,
                        "item_type": item.item_type,
                        "item_type_suggestions": suggestions,
                        "yolo_confidence": yolo_conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )

        scan.status = "completed"
        db.add(scan)
        db.commit()

    except Exception as e:
        db.rollback()
        scan.status = "failed"
        scan.error_message = str(e)[:250]
        db.add(scan)
        db.commit()

        # Best-effort cleanup
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

        log_user_event(
            user_id=current_user.id,
            event="scan_upload_failed",
            meta={"scan_id": str(scan.id), "error": str(e)},
        )

        raise HTTPException(
            status_code=400,
            detail=f"Scan processing failed: {e}",
        )

    # Backwards-compatible top-level fields (scan.html expects single mask/color).
    primary = items_out[0] if items_out else {}

    log_user_event(
        user_id=current_user.id,
        event="scan_upload_completed",
        meta={
            "scan_id": str(scan.id),
            "wardrobe_item_ids": [str(x) for x in created_ids],
            "warnings": warnings,
        },
    )

    return {
        "scan_id": scan.id,
        "status": scan.status,
        "wardrobe_item_created": bool(created_ids),
        "wardrobe_item_ids": [str(x) for x in created_ids],
        "mask_url": primary.get("mask_url"),
        "extracted_color": primary.get("extracted_color"),
        "color_palette": primary.get("color_palette"),
        "color_confidence": primary.get("color_confidence"),
        "item_type_suggestions": primary.get("item_type_suggestions", []),
        "warnings": warnings,
        "items": items_out,
    }


@router.get("/history")
def get_scan_history(
    limit: int = 200,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Timeline proxy for scans based on created wardrobe items.
    safe_limit = max(1, min(int(limit), 1000))
    items = (
        db.query(WardrobeItem)
        .filter(WardrobeItem.user_id == current_user.id)
        .order_by(WardrobeItem.created_at.desc())
        .limit(safe_limit)
        .all()
    )

    payload = []
    for item in items:
        label = "upper" if item.category == "top" else "lower"
        payload.append(
            {
                "wardrobe_item_id": item.id,
                "scan_id": None,
                "uploaded_image_url": item.image_url,
                "uploaded_image_preview_url": _to_public_upload_url(item.image_url),
                "mask_url": item.mask_url,
                "mask_preview_url": _to_public_upload_url(item.mask_url),
                "detected_label": label,
                "category": item.category,
                "item_type": item.item_type,
                "color": item.color,
                "wear_count": item.wear_count,
                "created_at": item.created_at,
                "is_active": item.is_active,
            }
        )

    return {"count": len(payload), "items": payload}
