# app/api/outfits.py

import json
import math
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.body_profile import BodyProfile
from app.models.feedback import Feedback
from app.models.wardrobe import WardrobeItem
from app.models.outfit import Outfit, OutfitItem
from app.services.outfit_engine import generate_weekly_plan
from app.services.outfit_engine import cosine_similarity
from app.dependencies import get_db, get_current_user
from app.services.color_labels import normalize_color_name
from app.services.mf_recommender import MFRecommender
from app.services.style_profile import compute_user_style_profile
from app.services.dev_event_log import log_user_event

router = APIRouter(prefix="/outfits", tags=["Outfits"])

REPEAT_BLOCK_DAYS = 7
ITEM_COOLDOWN_DAYS = 2
EXPLORATION_RATE = 6
CHALLENGER_CONFIDENCE_CUTOFF = 0.55

_FEEDBACK_WEIGHTS: dict[str, float] = {
    "liked": 0.08,
    "worn": 0.06,
    "skipped": -0.03,
    "disliked": -0.10,
}
_FEEDBACK_HALF_LIFE_DAYS = 30.0
_MAX_PAIR_BIAS = 0.15
_RECENT_EMBEDDING_DAYS = 7
_RECENT_EMBEDDING_LIMIT = 50
_ROTATION_HISTORY_DAYS = 30
_ROTATION_MAX_FREQUENCY = 0.30
_MF_FACTORS = 20


class MarkWornPayload(BaseModel):
    outfit_id: UUID | None = None
    top_item_id: UUID | None = None
    bottom_item_id: UUID | None = None
    worn_date: date | None = None


def _feedback_decay(created_at: datetime, now: datetime) -> float:
    ts = created_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days_since = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (days_since / _FEEDBACK_HALF_LIFE_DAYS)


def _build_pair_biases(db: Session, user_id: UUID) -> dict[tuple[UUID, UUID], float]:
    allowed = tuple(_FEEDBACK_WEIGHTS.keys())
    feedback_subq = (
        db.query(
            Feedback.id.label("feedback_id"),
            Feedback.outfit_id.label("outfit_id"),
            Feedback.feedback_type.label("feedback_type"),
            Feedback.created_at.label("created_at"),
        )
        .filter(
            Feedback.user_id == user_id,
            Feedback.feedback_type.in_(allowed),
        )
        .order_by(Feedback.created_at.desc())
        .limit(300)
        .subquery()
    )

    rows = (
        db.query(
            feedback_subq.c.feedback_id,
            feedback_subq.c.feedback_type,
            feedback_subq.c.created_at,
            OutfitItem.wardrobe_item_id,
            WardrobeItem.category,
        )
        .join(OutfitItem, OutfitItem.outfit_id == feedback_subq.c.outfit_id)
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .all()
    )

    feedback_items: dict[UUID, dict] = {}
    for feedback_id, feedback_type, created_at, wardrobe_item_id, category in rows:
        entry = feedback_items.setdefault(
            feedback_id,
            {
                "feedback_type": feedback_type,
                "created_at": created_at,
                "top_id": None,
                "bottom_id": None,
            },
        )
        if category == "top":
            entry["top_id"] = wardrobe_item_id
        elif category == "bottom":
            entry["bottom_id"] = wardrobe_item_id

    now = datetime.now(timezone.utc)
    pair_biases: dict[tuple[UUID, UUID], float] = {}
    for entry in feedback_items.values():
        top_id = entry["top_id"]
        bottom_id = entry["bottom_id"]
        if not top_id or not bottom_id:
            continue

        weight = _FEEDBACK_WEIGHTS.get(entry["feedback_type"])
        if not weight:
            continue

        bias = weight * _feedback_decay(entry["created_at"], now)
        key = (top_id, bottom_id)
        pair_biases[key] = pair_biases.get(key, 0.0) + bias

    for key, value in list(pair_biases.items()):
        if value > _MAX_PAIR_BIAS:
            pair_biases[key] = _MAX_PAIR_BIAS
        elif value < -_MAX_PAIR_BIAS:
            pair_biases[key] = -_MAX_PAIR_BIAS

    return pair_biases


def _get_recent_pairs(db: Session, user_id: UUID, days: int) -> set[tuple[UUID, UUID]]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    rows = (
        db.query(
            OutfitItem.outfit_id,
            OutfitItem.wardrobe_item_id,
            WardrobeItem.category,
        )
        .join(Outfit, Outfit.id == OutfitItem.outfit_id)
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .filter(
            Outfit.user_id == user_id,
            Outfit.outfit_date >= cutoff,
        )
        .all()
    )
    grouped: dict[UUID, dict[str, UUID]] = {}
    for outfit_id, wid, cat in rows:
        entry = grouped.setdefault(outfit_id, {"top": None, "bottom": None})
        if cat == "top" and entry["top"] is None:
            entry["top"] = wid
        elif cat == "bottom" and entry["bottom"] is None:
            entry["bottom"] = wid
    result = set()
    for entry in grouped.values():
        if entry["top"] and entry["bottom"]:
            result.add((entry["top"], entry["bottom"]))
    return result


def _similarity_penalty(sim: float) -> float:
    if sim >= 0.90:
        return -0.25
    if sim >= 0.80:
        return -0.15
    if sim >= 0.70:
        return -0.05
    return 0.0


def _get_negative_signals(
    db: Session,
    user_id: UUID,
    wardrobe_items: list[WardrobeItem],
) -> tuple[set[UUID], dict[UUID, float]]:
    """Return (banned_ids, similarity_penalties per item_id)."""
    disliked_rows = (
        db.query(
            OutfitItem.wardrobe_item_id,
            WardrobeItem.embedding,
        )
        .join(Outfit, Outfit.id == OutfitItem.outfit_id)
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .join(Feedback, Feedback.outfit_id == Outfit.id)
        .filter(
            Feedback.user_id == user_id,
            Feedback.feedback_type == "disliked",
        )
        .all()
    )

    banned_ids: set[UUID] = set()
    disliked_embeddings: list[list[float]] = []
    for wid, emb in disliked_rows:
        banned_ids.add(wid)
        if isinstance(emb, list) and emb:
            disliked_embeddings.append(emb)

    similarity_penalties: dict[UUID, float] = {}
    if disliked_embeddings:
        for item in wardrobe_items:
            emb = getattr(item, "embedding", None)
            if not (isinstance(emb, list) and emb):
                continue
            max_sim = 0.0
            for d_emb in disliked_embeddings:
                max_sim = max(max_sim, cosine_similarity(emb, d_emb))
            penalty = _similarity_penalty(max_sim)
            if penalty:
                similarity_penalties[item.id] = penalty

    return banned_ids, similarity_penalties


def _parse_explanation(explanation: str | None):
    if not explanation:
        return None
    try:
        return json.loads(explanation)
    except json.JSONDecodeError:
        return explanation


def _prefetch_outfit_items(db: Session, outfit_ids: List[UUID]) -> Dict[UUID, List[UUID]]:
    if not outfit_ids:
        return {}
    rows = (
        db.query(OutfitItem.outfit_id, OutfitItem.wardrobe_item_id)
        .filter(OutfitItem.outfit_id.in_(outfit_ids))
        .all()
    )
    mapping: Dict[UUID, List[UUID]] = {}
    for outfit_id, wardrobe_item_id in rows:
        mapping.setdefault(outfit_id, []).append(wardrobe_item_id)
    return mapping


def _serialize_outfit(db: Session, outfit: Outfit, item_ids_map: Dict[UUID, List[UUID]] | None = None) -> dict:
    if item_ids_map is not None and outfit.id in item_ids_map:
        item_ids = item_ids_map[outfit.id]
    else:
        item_ids_rows = (
            db.query(OutfitItem.wardrobe_item_id)
            .filter(OutfitItem.outfit_id == outfit.id)
            .all()
        )
        item_ids = [row[0] for row in item_ids_rows]

    score_val = outfit.score
    try:
        score_val = float(score_val) if score_val is not None else None
    except Exception:
        pass

    return {
        "outfit_id": outfit.id,
        "date": outfit.outfit_date,
        "score": score_val,
        "final_score": score_val,
        "item_ids": item_ids,
        "explanation_data": _parse_explanation(outfit.explanation),
    }


def _get_recent_item_embeddings(db: Session, user_id: UUID) -> list[list[float]]:
    """
    Collect embeddings for recently worn items (by WardrobeItem.last_worn_at).
    This is cheap and avoids depending on outfit-history completeness.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_RECENT_EMBEDDING_DAYS)

    items = (
        db.query(WardrobeItem.embedding)
        .filter(
            WardrobeItem.user_id == user_id,
            WardrobeItem.is_active.is_(True),
            WardrobeItem.last_worn_at.is_not(None),
            WardrobeItem.last_worn_at >= cutoff,
            WardrobeItem.embedding.is_not(None),
        )
        .order_by(WardrobeItem.last_worn_at.desc())
        .limit(_RECENT_EMBEDDING_LIMIT)
        .all()
    )

    result: list[list[float]] = []
    for (emb,) in items:
        if isinstance(emb, list) and emb:
            result.append(emb)
    return result


def _mf_item_key(category: str | None, item_type: str | None, color: str | None) -> str:
    cat = (category or "unknown").strip().lower()
    itype = (item_type or "unknown").strip().lower()
    col = normalize_color_name(color)
    return f"{cat}|{itype}|{col}"


def _build_mf_item_scores(
    db: Session,
    user_id: UUID,
    wardrobe_items: list[WardrobeItem],
) -> dict[UUID, float]:
    """
    Build collaborative scores from cross-user wear patterns.
    We factorize a user x item-archetype matrix (category|item_type|base_color).
    """
    rows = (
        db.query(
            WardrobeItem.user_id,
            WardrobeItem.category,
            WardrobeItem.item_type,
            WardrobeItem.color,
            WardrobeItem.wear_count,
        )
        .filter(
            WardrobeItem.is_active.is_(True),
            WardrobeItem.item_type != "unknown",
        )
        .all()
    )

    if not rows:
        return {}

    user_keys = sorted({str(r[0]) for r in rows})
    if str(user_id) not in user_keys:
        user_keys.append(str(user_id))
    item_keys = sorted({_mf_item_key(r[1], r[2], r[3]) for r in rows})
    if len(user_keys) < 2 or len(item_keys) < 2:
        return {}

    user_idx = {u: i for i, u in enumerate(user_keys)}
    item_idx = {k: i for i, k in enumerate(item_keys)}
    matrix = [[0.0 for _ in item_keys] for _ in user_keys]

    for row_user_id, row_category, row_type, row_color, row_wear_count in rows:
        u = user_idx[str(row_user_id)]
        k = item_idx[_mf_item_key(row_category, row_type, row_color)]
        wear = float(max(0, int(row_wear_count or 0)))
        matrix[u][k] += math.log1p(wear)

    model = MFRecommender(n_factors=_MF_FACTORS).fit(
        wear_matrix=matrix,
        item_ids=item_keys,
        user_ids=user_keys,
    )
    if not model.fitted:
        return {}

    key_scores = model.normalized_scores_for_user(str(user_id))
    out: dict[UUID, float] = {}
    for item in wardrobe_items:
        key = _mf_item_key(item.category, item.item_type, item.color)
        score = key_scores.get(key)
        if score is None:
            score = 0.5
        out[item.id] = float(score)
    return out


def _get_recent_recommendation_top_history(
    db: Session,
    user_id: UUID,
    days: int = _ROTATION_HISTORY_DAYS,
) -> tuple[dict[UUID, int], int]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    rows = (
        db.query(Outfit.id, OutfitItem.wardrobe_item_id)
        .join(OutfitItem, OutfitItem.outfit_id == Outfit.id)
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .filter(
            Outfit.user_id == user_id,
            Outfit.outfit_date >= cutoff,
            WardrobeItem.category == "top",
        )
        .all()
    )
    counts: dict[UUID, int] = {}
    for _outfit_id, top_id in rows:
        counts[top_id] = int(counts.get(top_id, 0)) + 1
    return counts, int(sum(counts.values()))


def _to_public_upload_url(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("/uploads/") or value.startswith("http://") or value.startswith("https://"):
        return value
    filename = str(value).replace("\\", "/").split("/")[-1]
    if not filename:
        return value
    return f"/uploads/{filename}"


def _parse_month_start(month: str) -> date:
    try:
        dt = datetime.strptime(f"{month}-01", "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=422, detail='month must be in "YYYY-MM" format')
    return dt.date()


@router.post("/generate")
def generate_outfits(
    start_date: date,
    days: int = 7,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if days < 1 or days > 30:
        raise HTTPException(status_code=422, detail="days must be between 1 and 30")

    # Load active wardrobe items. Do not hard-block on unknown item_type:
    # scanning should remain usable before manual item_type curation.
    wardrobe_items = (
        db.query(WardrobeItem)
        .filter(
            WardrobeItem.user_id == current_user.id,
            WardrobeItem.is_active.is_(True),
        )
        .all()
    )

    if not wardrobe_items:
        raise HTTPException(
            status_code=400,
            detail="No wardrobe items found. Scan at least one top and one bottom first.",
        )

    has_top = any(i.category == "top" for i in wardrobe_items)
    has_bottom = any(i.category == "bottom" for i in wardrobe_items)
    if not (has_top and has_bottom):
        raise HTTPException(
            status_code=400,
            detail="Need at least one active top and one active bottom.",
        )

    results = []
    created_count = 0

    pair_biases = _build_pair_biases(db, current_user.id)
    recent_embeddings = _get_recent_item_embeddings(db, current_user.id)
    recent_pairs = _get_recent_pairs(db, current_user.id, REPEAT_BLOCK_DAYS)
    rotation_top_history, rotation_total = _get_recent_recommendation_top_history(
        db,
        current_user.id,
        days=_ROTATION_HISTORY_DAYS,
    )
    banned_ids, similarity_penalties = _get_negative_signals(db, current_user.id, wardrobe_items)

    # Remove banned items up front (never-show should be a hard ban)
    wardrobe_items = [w for w in wardrobe_items if w.id not in banned_ids]
    mf_item_scores = _build_mf_item_scores(db, current_user.id, wardrobe_items)

    profile = (
        db.query(BodyProfile)
        .filter(BodyProfile.user_id == current_user.id)
        .first()
    )
    style_profile = compute_user_style_profile(db, current_user.id)
    user_ctx = SimpleNamespace(
        id=current_user.id,
        body_shape=(profile.body_shape if profile else None),
        forbidden_items=current_user.forbidden_items,
        fit_preference=(profile.fit_preference if profile else None),
        style_profile={
            **style_profile.to_public(),
            "liked_centroid": style_profile.liked_centroid,
            "disliked_centroid": style_profile.disliked_centroid,
        },
    )

    # Generate weekly plan
    weekly_plan = generate_weekly_plan(
        user=user_ctx,
        wardrobe_items=wardrobe_items,
        days=days,
        pair_biases=pair_biases,
        recent_item_embeddings=recent_embeddings,
        recent_pairs=recent_pairs,
        similarity_penalties=similarity_penalties,
        mf_item_scores=mf_item_scores,
        banned_item_ids=banned_ids,
        rotation_top_history=rotation_top_history,
        rotation_total_count=rotation_total,
        rotation_max_frequency=_ROTATION_MAX_FREQUENCY,
    )

    end_date = start_date + timedelta(days=days - 1)
    existing_outfits = (
        db.query(Outfit)
        .filter(
            Outfit.user_id == current_user.id,
            Outfit.outfit_date >= start_date,
            Outfit.outfit_date <= end_date,
        )
        .all()
    )
    existing_by_date = {o.outfit_date: o for o in existing_outfits}
    outfit_item_map = _prefetch_outfit_items(db, [o.id for o in existing_outfits])

    for i in range(days):
        outfit_date = start_date + timedelta(days=i)

        # Ensure idempotency: one outfit per day
        existing = existing_by_date.get(outfit_date)

        if existing:
            existing_item_ids = set(outfit_item_map.get(existing.id, []))
            if banned_ids and existing_item_ids.intersection(banned_ids):
                # Remove outdated outfit that contains banned items, then regenerate.
                db.query(OutfitItem).filter(OutfitItem.outfit_id == existing.id).delete()
                db.delete(existing)
                db.flush()
            else:
                results.append({**_serialize_outfit(db, existing, outfit_item_map), "created": False})
                continue

        if i >= len(weekly_plan):
            # Not enough candidate outfits to fill requested range.
            continue

        plan = weekly_plan[i]

        outfit = Outfit(
            user_id=current_user.id,
            outfit_date=outfit_date,
            score=plan["score"],
            explanation=json.dumps(plan["explanation_data"])
        )

        db.add(outfit)
        db.flush()  # get outfit.id

        for item in plan["items"]:
            db.add(
                OutfitItem(
                    outfit_id=outfit.id,
                    wardrobe_item_id=item.id
                )
            )
        outfit_item_map[outfit.id] = [itm.id for itm in plan["items"]]

        created_count += 1
        results.append(
            {
                "outfit_id": outfit.id,
                "date": outfit_date,
                "score": plan["score"],
                "final_score": plan["score"],
                "item_ids": [i.id for i in plan["items"]],
                "explanation_data": plan["explanation_data"],
                "created": True,
            }
        )

    db.commit()

    log_user_event(
        user_id=current_user.id,
        event="outfits_generate",
        meta={
            "start_date": str(start_date),
            "days": days,
            "eligible_items": len(wardrobe_items),
            "generated": created_count,
            "returned": len(results),
        },
    )
    return {
        "generated": created_count,
        "returned": len(results),
        "outfits": results,
    }


@router.get("/history")
def get_outfit_history(
    month: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_date = _parse_month_start(month)
    if start_date.month == 12:
        end_date_exclusive = date(start_date.year + 1, 1, 1)
    else:
        end_date_exclusive = date(start_date.year, start_date.month + 1, 1)

    outfits = (
        db.query(Outfit)
        .filter(
            Outfit.user_id == current_user.id,
            Outfit.outfit_date >= start_date,
            Outfit.outfit_date < end_date_exclusive,
        )
        .order_by(Outfit.outfit_date.asc(), Outfit.created_at.asc())
        .all()
    )
    if not outfits:
        return {"month": month, "count": 0, "items": []}

    outfit_ids = [o.id for o in outfits]
    item_rows = (
        db.query(
            OutfitItem.outfit_id,
            WardrobeItem.id,
            WardrobeItem.category,
            WardrobeItem.item_type,
            WardrobeItem.color,
            WardrobeItem.image_url,
            WardrobeItem.mask_url,
        )
        .join(WardrobeItem, WardrobeItem.id == OutfitItem.wardrobe_item_id)
        .filter(OutfitItem.outfit_id.in_(outfit_ids))
        .all()
    )
    by_outfit: dict[UUID, list[dict]] = {}
    for outfit_id, item_id, category, item_type, color, image_url, mask_url in item_rows:
        by_outfit.setdefault(outfit_id, []).append(
            {
                "id": item_id,
                "category": category,
                "item_type": item_type,
                "color": color,
                "image_url": image_url,
                "image_preview_url": _to_public_upload_url(image_url),
                "mask_url": mask_url,
                "mask_preview_url": _to_public_upload_url(mask_url),
            }
        )

    worn_outfit_ids = {
        outfit_id
        for (outfit_id,) in db.query(Feedback.outfit_id)
        .filter(
            Feedback.user_id == current_user.id,
            Feedback.feedback_type == "worn",
            Feedback.outfit_id.in_(outfit_ids),
        )
        .all()
    }

    payload = []
    for outfit in outfits:
        items = by_outfit.get(outfit.id, [])
        top_item = next((it for it in items if it["category"] == "top"), None)
        bottom_item = next((it for it in items if it["category"] == "bottom"), None)

        score_val = outfit.score
        try:
            score_val = float(score_val) if score_val is not None else None
        except Exception:
            score_val = None

        payload.append(
            {
                "outfit_id": outfit.id,
                "date": outfit.outfit_date,
                "score": score_val,
                "explanation_data": _parse_explanation(outfit.explanation),
                "top_item": top_item,
                "bottom_item": bottom_item,
                "items": items,
                "is_worn": outfit.id in worn_outfit_ids,
            }
        )

    return {"month": month, "count": len(payload), "items": payload}


@router.post("/mark-worn")
def mark_outfit_worn(
    payload: MarkWornPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    target_date = payload.worn_date or datetime.now(timezone.utc).date()
    outfit: Outfit | None = None
    item_ids: list[UUID] = []

    if payload.outfit_id:
        outfit = (
            db.query(Outfit)
            .filter(Outfit.id == payload.outfit_id, Outfit.user_id == current_user.id)
            .first()
        )
        if not outfit:
            raise HTTPException(status_code=404, detail="Outfit not found")

        item_ids = [
            wid
            for (wid,) in db.query(OutfitItem.wardrobe_item_id)
            .filter(OutfitItem.outfit_id == outfit.id)
            .all()
        ]
    else:
        if not payload.top_item_id or not payload.bottom_item_id:
            raise HTTPException(
                status_code=422,
                detail='Provide either "outfit_id" or both "top_item_id" and "bottom_item_id".',
            )

        top_item = (
            db.query(WardrobeItem)
            .filter(
                WardrobeItem.id == payload.top_item_id,
                WardrobeItem.user_id == current_user.id,
                WardrobeItem.is_active.is_(True),
            )
            .first()
        )
        bottom_item = (
            db.query(WardrobeItem)
            .filter(
                WardrobeItem.id == payload.bottom_item_id,
                WardrobeItem.user_id == current_user.id,
                WardrobeItem.is_active.is_(True),
            )
            .first()
        )
        if not top_item or not bottom_item:
            raise HTTPException(status_code=404, detail="Top or bottom wardrobe item not found")
        if top_item.category != "top" or bottom_item.category != "bottom":
            raise HTTPException(status_code=422, detail="Item categories must be top and bottom")

        item_ids = [top_item.id, bottom_item.id]

        existing_same_day = (
            db.query(Outfit)
            .filter(
                Outfit.user_id == current_user.id,
                Outfit.outfit_date == target_date,
            )
            .all()
        )
        for cand in existing_same_day:
            cand_item_ids = {
                wid
                for (wid,) in db.query(OutfitItem.wardrobe_item_id)
                .filter(OutfitItem.outfit_id == cand.id)
                .all()
            }
            if set(item_ids).issubset(cand_item_ids):
                outfit = cand
                break

        if outfit is None:
            outfit = Outfit(
                user_id=current_user.id,
                outfit_date=target_date,
                score=None,
                explanation=json.dumps(
                    {
                        "source": "manual_mark_worn",
                        "reasoning": {
                            "color": "Marked as worn by user from dashboard widget.",
                            "style": "Recorded from explicit user action.",
                        },
                    }
                ),
            )
            db.add(outfit)
            db.flush()
            for wid in item_ids:
                db.add(OutfitItem(outfit_id=outfit.id, wardrobe_item_id=wid))

    if not outfit:
        raise HTTPException(status_code=400, detail="Unable to resolve outfit")

    existing_worn = (
        db.query(Feedback)
        .filter(
            Feedback.user_id == current_user.id,
            Feedback.outfit_id == outfit.id,
            Feedback.feedback_type == "worn",
        )
        .first()
    )
    updated_items = 0
    if existing_worn is None and item_ids:
        now = datetime.now(timezone.utc)
        feedback = Feedback(
            user_id=current_user.id,
            outfit_id=outfit.id,
            feedback_type="worn",
        )
        db.add(feedback)
        updated_items = (
            db.query(WardrobeItem)
            .filter(
                WardrobeItem.user_id == current_user.id,
                WardrobeItem.id.in_(item_ids),
            )
            .update(
                {
                    WardrobeItem.wear_count: WardrobeItem.wear_count + 1,
                    WardrobeItem.last_worn_at: now,
                    WardrobeItem.updated_at: now,
                },
                synchronize_session=False,
            )
        )

    db.commit()
    log_user_event(
        user_id=current_user.id,
        event="outfit_mark_worn",
        meta={
            "outfit_id": str(outfit.id),
            "created_feedback": existing_worn is None,
            "updated_items": updated_items,
            "worn_date": str(target_date),
        },
    )
    return {
        "ok": True,
        "outfit_id": outfit.id,
        "date": target_date,
        "created_feedback": existing_worn is None,
        "updated_items": int(updated_items),
    }


@router.get("/{outfit_date}")
def get_outfit_by_date(
    outfit_date: date,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    outfit = (
        db.query(Outfit)
        .filter(
            Outfit.user_id == current_user.id,
            Outfit.outfit_date == outfit_date
        )
        .first()
    )

    if not outfit:
        raise HTTPException(
            status_code=404,
            detail="Outfit not found"
        )

    items = (
        db.query(WardrobeItem)
        .join(
            OutfitItem,
            OutfitItem.wardrobe_item_id == WardrobeItem.id
        )
        .filter(OutfitItem.outfit_id == outfit.id)
        .all()
    )

    explanation = None
    if outfit.explanation:
        try:
            explanation = json.loads(outfit.explanation)
        except json.JSONDecodeError:
            explanation = outfit.explanation

    return {
        "date": outfit.outfit_date,
        "outfit_id": outfit.id,
        "score": float(outfit.score) if outfit.score is not None else None,
        "final_score": float(outfit.score) if outfit.score is not None else None,
        "items": [
            {
                "id": item.id,
                "type": item.item_type,
                "color": item.color
            }
            for item in items
        ],
        "explanation": explanation
    }
