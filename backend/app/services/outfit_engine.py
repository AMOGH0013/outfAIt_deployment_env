# app/services/outfit_engine.py

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional, Mapping, Sequence
from uuid import UUID
import random
import math

from app.services.color_labels import normalize_color_name

# -------------------------
# CONSTANTS
# -------------------------

NEUTRAL_COLORS = {"black", "white", "gray", "navy", "beige", "brown", "teal", "olive"}

NOVELTY_COOLDOWN_DAYS = 7
REPEAT_BLOCK_DAYS = 7
ITEM_COOLDOWN_DAYS = 2
ITEM_COOLDOWN_PENALTY = -0.20
REPEAT_PAIR_PENALTY = -0.45
SESSION_REPEAT_TOP_PENALTY = -0.45
SESSION_RECENT_TOP_PENALTY = -0.55
EXPLORATION_RATE = 6  # every N outfits, allow a challenger
CONFIDENCE_CUTOFF = 0.55
CONFIDENCE_WEIGHTS = (1.0, 0.8, 1.2, 0.8)  # rule, fit, diversity_penalty, repetition_risk
FIT_BONUS_MAX = 0.05
STYLE_AFFINITY_MAX = 0.05
MMR_LAMBDA = 0.70
MMR_TOP_N = 18
SCORE_NORMALIZATION_MAX = 1.35
SCORE_FLOOR = 0.0
SCORE_CEILING = 1.0
SCORE_ROUND_DIGITS = 3


def _primary_color(color_str: Optional[str]) -> str:
    return normalize_color_name(color_str)


# -------------------------
# DATA INTERFACES (Duck Typing)
# -------------------------
# These are NOT ORM models.
# They describe what attributes are expected.

class UserLike:
    id: UUID
    body_shape: Optional[str]
    forbidden_items: Optional[List[str]]
    fit_preference: Optional[str]
    # Optional derived, per-user style fingerprint (computed upstream).
    # Expected keys (if present): preferred_* and centroid vectors.
    style_profile: Optional[dict]


class WardrobeItemLike:
    id: UUID
    category: str            # "top" or "bottom"
    item_type: str           # "shirt", "trouser"
    color: Optional[str]
    fit: Optional[str]
    last_worn_at: Optional[datetime]
    wear_count: int
    is_active: bool
    embedding: Optional[List[float]]


# -------------------------
# FILTERING
# -------------------------

def filter_wardrobe_items(
    items: List[WardrobeItemLike],
    forbidden_items: Optional[List[str]],
    banned_item_ids: Optional[set] = None,
) -> List[WardrobeItemLike]:
    """Remove inactive and forbidden items."""
    forbidden = set(forbidden_items or [])
    banned = banned_item_ids or set()
    return [
        item for item in items
        if item.is_active and item.item_type not in forbidden and item.id not in banned
    ]


# -------------------------
# CANDIDATE GENERATION
# -------------------------

def split_by_category(
    items: List[WardrobeItemLike]
) -> Tuple[List[WardrobeItemLike], List[WardrobeItemLike]]:
    tops = [i for i in items if i.category == "top"]
    bottoms = [i for i in items if i.category == "bottom"]
    return tops, bottoms


def generate_outfit_candidates(
    tops: List[WardrobeItemLike],
    bottoms: List[WardrobeItemLike]
) -> List[Tuple[WardrobeItemLike, WardrobeItemLike]]:
    return [(top, bottom) for top in tops for bottom in bottoms]


# -------------------------
# SCORING FUNCTIONS
# -------------------------

def body_shape_score(user: UserLike, top: WardrobeItemLike) -> float:
    if not user.body_shape:
        return 1.0

    shape = user.body_shape.lower()
    fit = (top.fit or "").lower()

    if shape == "triangle":
        return 1.0 if fit in ("structured", "regular") else 0.6
    if shape == "apple":
        return 0.6 if fit == "tight" else 1.0

    return 1.0


def color_harmony_score(
    top: WardrobeItemLike,
    bottom: WardrobeItemLike
) -> float:
    top_color = _primary_color(getattr(top, "color", None))
    bottom_color = _primary_color(getattr(bottom, "color", None))

    if top_color in NEUTRAL_COLORS and bottom_color in NEUTRAL_COLORS:
        return 0.8
    if top_color in NEUTRAL_COLORS or bottom_color in NEUTRAL_COLORS:
        return 1.0
    return 0.7


def novelty_score(item: WardrobeItemLike) -> float:
    if not item.last_worn_at:
        return 1.0

    now_utc = datetime.now(timezone.utc)
    last_worn = item.last_worn_at
    if last_worn.tzinfo is None:
        last_worn = last_worn.replace(tzinfo=timezone.utc)

    days_since = (now_utc - last_worn).days
    return min(days_since / NOVELTY_COOLDOWN_DAYS, 1.0)


def _normalize_fit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    aliases = {
        "tight": "slim",
        "skinny": "slim",
        "slim": "slim",
        "regular": "regular",
        "relaxed": "loose",
        "loose": "loose",
        "oversized": "loose",
    }
    return aliases.get(v, v)


def _default_fit_for_item_type(item_type: str) -> Optional[str]:
    t = (item_type or "").strip().lower()
    defaults = {
        "hoodie": "loose",
        "kurta": "regular",
        "shirt": "regular",
        "tshirt": "regular",
        "jeans": "regular",
        "trousers": "regular",
        "chinos": "regular",
        "shorts": "regular",
    }
    return defaults.get(t)


def fit_preference_score(user: UserLike, items: List[WardrobeItemLike]) -> float:
    """
    Small bounded modifier based on user's fit_preference vs item fit (or a safe default by item_type).
    Returns value in [-0.05, +0.05].
    """
    pref = _normalize_fit(getattr(user, "fit_preference", None))
    if not pref:
        return 0.0
    if pref not in {"slim", "regular", "loose"}:
        return 0.0

    def fit_to_level(f: str) -> int:
        return {"slim": -1, "regular": 0, "loose": 1}.get(f, 0)

    pref_level = fit_to_level(pref)
    per_item: List[float] = []
    for item in items:
        item_fit = _normalize_fit(getattr(item, "fit", None)) or _default_fit_for_item_type(item.item_type)
        if not item_fit or item_fit not in {"slim", "regular", "loose"}:
            continue
        diff = abs(pref_level - fit_to_level(item_fit))
        if diff == 0:
            per_item.append(FIT_BONUS_MAX)
        elif diff == 1:
            per_item.append(0.0)
        else:
            per_item.append(-FIT_BONUS_MAX)

    if not per_item:
        return 0.0
    score = sum(per_item) / float(len(per_item))
    return round(max(-FIT_BONUS_MAX, min(score, FIT_BONUS_MAX)), 3)


# -------------------------
# EMBEDDING DIVERSITY
# -------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity in [0, 1] for non-negative CLIP similarities (embeddings are normalized).
    Robust to non-normalized inputs (will normalize via norms).
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        norm_a += float(x) * float(x)
        norm_b += float(y) * float(y)

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 1e-12:
        return 0.0
    sim = dot / denom
    # Numerical guard
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def _normalized_vec(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    norm = math.sqrt(sum(float(x) * float(x) for x in values))
    if norm <= 1e-12:
        return None
    return [float(x) / norm for x in values]


def _pair_embedding(top: WardrobeItemLike, bottom: WardrobeItemLike) -> list[float] | None:
    top_emb = getattr(top, "embedding", None)
    bottom_emb = getattr(bottom, "embedding", None)
    top_vec = _normalized_vec(top_emb) if isinstance(top_emb, list) else None
    bottom_vec = _normalized_vec(bottom_emb) if isinstance(bottom_emb, list) else None

    if top_vec and bottom_vec and len(top_vec) == len(bottom_vec):
        avg = [(a + b) * 0.5 for a, b in zip(top_vec, bottom_vec)]
        return _normalized_vec(avg)
    return top_vec or bottom_vec


def _max_similarity_to_pool(vec: list[float] | None, pool: Sequence[Sequence[float]]) -> float:
    if vec is None or not pool:
        return 0.0
    best = 0.0
    for prev in pool:
        sim = cosine_similarity(vec, prev)
        if sim > best:
            best = sim
    return best


def _similarity_to_effect(max_sim: float) -> float:
    """
    Map max similarity to bounded penalty/bonus (deterministic, rule-based).

    - >= 0.90: -0.30
    - 0.80-0.90: -0.15
    - 0.70-0.80: -0.05
    - < 0.70: +0.05
    """
    if max_sim >= 0.90:
        return -0.30
    if max_sim >= 0.80:
        return -0.15
    if max_sim >= 0.70:
        return -0.05
    return 0.05


def embedding_diversity_score(
    items: List[WardrobeItemLike],
    recent_item_embeddings: Optional[Sequence[Sequence[float]]],
) -> Tuple[float, Dict[str, float]]:
    """
    Compare each candidate item to recent worn-item embeddings and produce a bounded additive score.

    Returns:
      (score, diagnostics) where diagnostics includes per-item max similarity.

    Cold start: if no recent embeddings, returns (0.0, {}).
    Missing embeddings: that item contributes 0.0 and is omitted from diagnostics.
    """
    if not recent_item_embeddings:
        return 0.0, {}

    effects: List[float] = []
    sims_by_item: Dict[str, float] = {}

    for item in items:
        if not item.embedding:
            continue

        max_sim = 0.0
        for prev in recent_item_embeddings:
            sim = cosine_similarity(item.embedding, prev)
            if sim > max_sim:
                max_sim = sim

        sims_by_item[str(item.id)] = round(max_sim, 4)
        effects.append(_similarity_to_effect(max_sim))

    if not effects:
        return 0.0, {}

    # Average keeps total contribution bounded regardless of number of items in outfit.
    score = sum(effects) / float(len(effects))
    score = max(-0.30, min(score, 0.05))
    return round(score, 3), sims_by_item


# -------------------------
# STYLE AFFINITY (Derived)
# -------------------------

def style_affinity_score(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
) -> tuple[float, str | None]:
    """
    Small bounded modifier based on a derived style fingerprint.

    Returns (score in [-0.05, +0.05], reason_or_none).
    No ML training; fully deterministic.
    """
    sp = getattr(user, "style_profile", None)
    if not sp or not isinstance(sp, dict):
        return 0.0, None

    score = 0.0
    reasons: list[str] = []

    def has_any(x) -> bool:
        return bool(x) and isinstance(x, (list, tuple, set))

    top_colors = set(c for c in (sp.get("preferred_top_colors") or []) if isinstance(c, str))
    bottom_colors = set(c for c in (sp.get("preferred_bottom_colors") or []) if isinstance(c, str))
    top_types = set(t for t in (sp.get("preferred_top_types") or []) if isinstance(t, str))
    bottom_types = set(t for t in (sp.get("preferred_bottom_types") or []) if isinstance(t, str))

    tc = (top.color or "").strip().lower()
    bc = (bottom.color or "").strip().lower()
    tc_primary = _primary_color(top.color)
    bc_primary = _primary_color(bottom.color)

    if tc_primary and tc_primary in top_colors:
        score += 0.02
    if bc_primary and bc_primary in bottom_colors:
        score += 0.02

    if has_any(top_types) and (top.item_type or "").strip().lower() in top_types:
        score += 0.01
    if has_any(bottom_types) and (bottom.item_type or "").strip().lower() in bottom_types:
        score += 0.01

    liked_centroid = sp.get("liked_centroid")
    disliked_centroid = sp.get("disliked_centroid")
    if liked_centroid and top.embedding:
        sim = cosine_similarity(top.embedding, liked_centroid)
        if sim >= 0.85:
            score += 0.01
    if liked_centroid and bottom.embedding:
        sim = cosine_similarity(bottom.embedding, liked_centroid)
        if sim >= 0.85:
            score += 0.01

    if disliked_centroid and top.embedding:
        sim = cosine_similarity(top.embedding, disliked_centroid)
        if sim >= 0.85:
            score -= 0.02
    if disliked_centroid and bottom.embedding:
        sim = cosine_similarity(bottom.embedding, disliked_centroid)
        if sim >= 0.85:
            score -= 0.02

    score = max(-STYLE_AFFINITY_MAX, min(score, STYLE_AFFINITY_MAX))
    score = round(score, 3)

    if abs(score) < 0.01:
        return 0.0, None

    if score > 0:
        reasons.append("matches your learned style preferences")
    else:
        reasons.append("may not match your learned style preferences")

    return score, "; ".join(reasons)


# -------------------------
# FINAL SCORING
# -------------------------

def rule_score(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
) -> float:
    score = 0.0
    score += 0.34 * body_shape_score(user, top)
    score += 0.24 * color_harmony_score(top, bottom)
    score += 0.14 * novelty_score(top)
    score += 0.14 * novelty_score(bottom)
    return score


def score_outfit(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
    mf_pair_score: Optional[float] = None,
    pair_bias: float = 0.0,
    embedding_score: float = 0.0,
    fit_score: float = 0.0,
    style_score: float = 0.0,
    cooldown_penalty: float = 0.0,
    dislike_penalty: float = 0.0,
    repeat_penalty: float = 0.0,
) -> float:
    base_rule_score = rule_score(user, top, bottom)
    if mf_pair_score is None:
        base_score = base_rule_score
    else:
        mf = max(0.0, min(float(mf_pair_score), 1.0))
        base_score = 0.35 * base_rule_score + 0.65 * mf

    raw_score = (
        base_score
        + pair_bias
        + embedding_score
        + fit_score
        + style_score
        + cooldown_penalty
        + dislike_penalty
        + repeat_penalty
    )
    normalized = raw_score / SCORE_NORMALIZATION_MAX
    score = max(SCORE_FLOOR, min(normalized, SCORE_CEILING))
    return round(score, SCORE_ROUND_DIGITS)


# -------------------------
# EXPLANATION DATA
# -------------------------

def build_explanation_data(
    user: UserLike,
    top: WardrobeItemLike,
    bottom: WardrobeItemLike,
    final_score: float,
    rule_score_value: float,
    pair_bias: float = 0.0,
    mf_pair_score: float | None = None,
    embedding_score: float = 0.0,
    fit_score: float = 0.0,
    style_score: float = 0.0,
    style_reason: str | None = None,
    similarity_diagnostics: Optional[Dict[str, float]] = None,
    confidence_score: float | None = None,
    challenger: bool = False,
    repetition_blocked_reason: str | None = None,
) -> Dict:
    data: Dict = {
        "top": top.item_type,
        "bottom": bottom.item_type,
        "top_color": top.color,
        "bottom_color": bottom.color,
        "body_shape": user.body_shape,
        "final_score": round(final_score, 2),
        "rule_score": round(rule_score_value, 3),
        "feedback_bias": round(pair_bias, 3),
        "mf_pair_score": (round(mf_pair_score, 3) if mf_pair_score is not None else None),
        "embedding_diversity_score": round(embedding_score, 3),
        "fit_score": round(fit_score, 3),
        "style_affinity_score": round(style_score, 3),
        "confidence_score": round(confidence_score, 3) if confidence_score is not None else None,
        "challenger": challenger,
        "reasoning": {
            "body_shape": "balanced fit for body shape",
            "color": "good color harmony",
            "novelty": "not worn recently"
        }
    }

    if abs(pair_bias) >= 0.01:
        if pair_bias > 0:
            data["reasoning"]["feedback"] = "boosted from your past feedback"
        else:
            data["reasoning"]["feedback"] = "penalized from your past feedback"

    if mf_pair_score is not None:
        if mf_pair_score >= 0.65:
            data["reasoning"]["collaborative"] = "ranked higher by collaborative wear-pattern signal"
        elif mf_pair_score <= 0.35:
            data["reasoning"]["collaborative"] = "ranked lower by collaborative wear-pattern signal"

    if abs(embedding_score) >= 0.01:
        if embedding_score > 0:
            data["reasoning"]["embedding_diversity"] = "boosted for visual diversity vs recently worn outfits"
        else:
            data["reasoning"]["embedding_diversity"] = "penalized for visual similarity to recently worn outfits"
        if similarity_diagnostics:
            data["max_similarities"] = similarity_diagnostics

    if abs(fit_score) >= 0.01:
        if fit_score > 0:
            data["reasoning"]["fit"] = "matches your fit preference"
        else:
            data["reasoning"]["fit"] = "mismatches your fit preference"

    if abs(style_score) >= 0.01 and style_reason:
        data["reasoning"]["style"] = style_reason

    if repetition_blocked_reason:
        data["repetition_blocked_reason"] = repetition_blocked_reason

    return data


def _days_since(dt: Optional[datetime]) -> Optional[int]:
    if not dt:
        return None
    now_utc = datetime.now(timezone.utc)
    val = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return (now_utc - val).days


def _cooldown_penalty(top: WardrobeItemLike, bottom: WardrobeItemLike) -> float:
    penalties = []
    for itm in (top, bottom):
        days = _days_since(itm.last_worn_at)
        if days is not None and days < ITEM_COOLDOWN_DAYS:
            penalties.append(ITEM_COOLDOWN_PENALTY)
    if not penalties:
        return 0.0
    return min(penalties)  # negative values


def _compute_confidence(rule_score_value: float, fit_score: float, embedding_score: float, repetition_risk: float) -> float:
    # embedding_score is negative when similar; treat penalty as positive risk
    diversity_penalty = max(0.0, -embedding_score)
    w_rule, w_fit, w_div, w_rep = CONFIDENCE_WEIGHTS
    raw = w_rule * rule_score_value + w_fit * fit_score - w_div * diversity_penalty - w_rep * repetition_risk
    conf = 1 / (1 + math.exp(-raw))
    return max(0.0, min(conf, 1.0))


# -------------------------
# PICK BEST OUTFIT
# -------------------------

def pick_best_outfit(
    user: UserLike,
    wardrobe_items: List[WardrobeItemLike],
    pair_biases: Optional[Mapping[Tuple[UUID, UUID], float]] = None,
    recent_item_embeddings: Optional[Sequence[Sequence[float]]] = None,
    recent_pairs: Optional[set] = None,
    similarity_penalties: Optional[Mapping[UUID, float]] = None,
    mf_item_scores: Optional[Mapping[UUID, float]] = None,
    exploration_index: int = 0,
    banned_item_ids: Optional[set] = None,
    session_used_top_ids: Optional[set] = None,
    session_recent_top_ids: Optional[set] = None,
    session_pair_embeddings: Optional[Sequence[Sequence[float]]] = None,
    rotation_top_history: Optional[Mapping[UUID, int]] = None,
    rotation_total_count: int = 0,
    rotation_max_frequency: float = 0.30,
) -> Optional[Dict]:
    filtered = filter_wardrobe_items(
        wardrobe_items,
        user.forbidden_items,
        banned_item_ids=banned_item_ids,
    )

    tops, bottoms = split_by_category(filtered)
    candidates = generate_outfit_candidates(tops, bottoms)

    if not candidates:
        return None

    def _score_candidates(allow_repeats: bool, override_reason: str | None):
        scored_local = []
        for top, bottom in candidates:
            repeat_pair = recent_pairs and (top.id, bottom.id) in recent_pairs
            if repeat_pair and not allow_repeats:
                continue

            pair_bias = 0.0
            if pair_biases:
                pair_bias = pair_biases.get((top.id, bottom.id), 0.0)

            mf_pair_score = None
            if mf_item_scores:
                mf_vals: list[float] = []
                for iid in (top.id, bottom.id):
                    val = mf_item_scores.get(iid)
                    if isinstance(val, (int, float)):
                        mf_vals.append(float(val))
                if mf_vals:
                    mf_pair_score = float(sum(mf_vals) / len(mf_vals))

            embedding_score, sim_diag = embedding_diversity_score(
                items=[top, bottom],
                recent_item_embeddings=recent_item_embeddings,
            )

            dislike_penalty = 0.0
            if similarity_penalties:
                dislike_penalty += similarity_penalties.get(top.id, 0.0)
                dislike_penalty += similarity_penalties.get(bottom.id, 0.0)

            fit_score = fit_preference_score(user, [top, bottom])
            style_score, style_reason = style_affinity_score(user, top, bottom)
            cooldown_pen = _cooldown_penalty(top, bottom)
            session_repeat_pen = (
                SESSION_REPEAT_TOP_PENALTY
                if (session_used_top_ids and top.id in session_used_top_ids)
                else 0.0
            )
            session_recent_pen = (
                SESSION_RECENT_TOP_PENALTY
                if (session_recent_top_ids and top.id in session_recent_top_ids)
                else 0.0
            )
            repeat_pen = (
                (REPEAT_PAIR_PENALTY if (repeat_pair and allow_repeats) else 0.0)
                + session_repeat_pen
                + session_recent_pen
            )
            rule_score_value = rule_score(user, top, bottom)
            score = score_outfit(
                user,
                top,
                bottom,
                mf_pair_score=mf_pair_score,
                pair_bias=pair_bias,
                embedding_score=embedding_score,
                fit_score=fit_score,
                style_score=style_score,
                cooldown_penalty=cooldown_pen,
                dislike_penalty=dislike_penalty,
                repeat_penalty=repeat_pen,
            )
            repetition_risk = abs(min(0.0, cooldown_pen)) + abs(min(0.0, repeat_pen))
            conf = _compute_confidence(rule_score_value, fit_score, embedding_score, repetition_risk)
            is_exploration_slot = (exploration_index % EXPLORATION_RATE == 0)
            challenger = is_exploration_slot and conf < CONFIDENCE_CUTOFF
            pair_emb = _pair_embedding(top, bottom)

            scored_local.append(
                (
                    score,
                    rule_score_value,
                    top,
                    bottom,
                    pair_bias,
                    embedding_score,
                    fit_score,
                    style_score,
                    style_reason,
                    sim_diag,
                    conf,
                    cooldown_pen,
                    dislike_penalty,
                    challenger,
                    override_reason if (repeat_pair and allow_repeats) else None,
                    mf_pair_score,
                    pair_emb,
                )
            )
        return scored_local

    scored = _score_candidates(False, None)
    if not scored:
        # Safety override: allow repeats if no alternatives, but penalize them
        scored = _score_candidates(True, "override_no_alternatives")
    if not scored:
        return None

    def _apply_rotation_quota(scored_rows):
        if not rotation_top_history:
            return scored_rows
        denom = max(1, int(rotation_total_count))
        filtered_rows = []
        for row in scored_rows:
            top_item = row[2]
            ratio = float(rotation_top_history.get(top_item.id, 0)) / float(denom)
            if ratio < float(rotation_max_frequency):
                filtered_rows.append(row)
        return filtered_rows or scored_rows

    mmr_history: list[list[float]] = []
    for emb in list(recent_item_embeddings or []):
        vec = _normalized_vec(emb)
        if vec:
            mmr_history.append(vec)
    for emb in list(session_pair_embeddings or []):
        vec = _normalized_vec(emb)
        if vec:
            mmr_history.append(vec)

    def _apply_mmr(scored_rows):
        if not mmr_history or not scored_rows:
            return scored_rows
        head = list(scored_rows[:MMR_TOP_N])
        tail = list(scored_rows[MMR_TOP_N:])
        if not any(row[16] is not None for row in head):
            return scored_rows

        def _mmr_value(row) -> float:
            relevance = float(row[0])
            pair_vec = row[16]
            diversity = 1.0 - _max_similarity_to_pool(pair_vec, mmr_history)
            return MMR_LAMBDA * relevance + (1.0 - MMR_LAMBDA) * diversity

        head.sort(key=_mmr_value, reverse=True)
        return head + tail

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = _apply_rotation_quota(scored)
    scored = _apply_mmr(scored)

    (
        best_score,
        best_rule_score,
        best_top,
        best_bottom,
        best_pair_bias,
        best_embedding_score,
        best_fit_score,
        best_style_score,
        best_style_reason,
        best_sim_diag,
        best_confidence,
        best_cooldown_pen,
        best_dislike_pen,
        best_challenger,
        best_repeat_override,
        best_mf_pair_score,
        best_pair_embedding,
    ) = scored[0]

    return {
        "score": best_score,
        "items": [best_top, best_bottom],
        "pair_embedding": best_pair_embedding,
        "explanation_data": build_explanation_data(
            user,
            best_top,
            best_bottom,
            final_score=best_score,
            rule_score_value=best_rule_score,
            pair_bias=best_pair_bias,
            mf_pair_score=best_mf_pair_score,
            embedding_score=best_embedding_score,
            fit_score=best_fit_score,
            style_score=best_style_score,
            style_reason=best_style_reason,
            similarity_diagnostics=best_sim_diag,
            confidence_score=best_confidence,
            challenger=best_challenger,
            repetition_blocked_reason=best_repeat_override,
        )
    }


# -------------------------
# WEEKLY PLAN
# -------------------------

def generate_weekly_plan(
    user: UserLike,
    wardrobe_items: List[WardrobeItemLike],
    days: int = 7,
    pair_biases: Optional[Mapping[Tuple[UUID, UUID], float]] = None,
    recent_item_embeddings: Optional[Sequence[Sequence[float]]] = None,
    recent_pairs: Optional[set] = None,
    similarity_penalties: Optional[Mapping[UUID, float]] = None,
    mf_item_scores: Optional[Mapping[UUID, float]] = None,
    banned_item_ids: Optional[set] = None,
    rotation_top_history: Optional[Mapping[UUID, int]] = None,
    rotation_total_count: int = 0,
    rotation_max_frequency: float = 0.30,
) -> List[Dict]:
    plan = []
    used_pairs: set[Tuple[UUID, UUID]] = set()
    used_top_ids: set[UUID] = set()
    recent_top_ids: List[UUID] = []
    chosen_pair_embeddings: List[list[float]] = []
    rotation_counts: dict[UUID, int] = dict(rotation_top_history or {})
    rotation_total = int(max(0, rotation_total_count))

    for day_index in range(days):
        # Avoid repeating the exact same top+bottom pair within the generated plan
        # while still allowing item reuse when alternatives exist.
        combined_recent_pairs = set(recent_pairs or set())
        combined_recent_pairs.update(used_pairs)

        def _pick(pool, force_unused_tops: bool):
            extra_banned = set(banned_item_ids or set())
            if force_unused_tops:
                extra_banned.update(used_top_ids)
            return pick_best_outfit(
                user,
                pool,
                pair_biases=pair_biases,
                recent_item_embeddings=recent_item_embeddings,
                recent_pairs=combined_recent_pairs,
                similarity_penalties=similarity_penalties,
                mf_item_scores=mf_item_scores,
                exploration_index=day_index,
                banned_item_ids=extra_banned,
                session_used_top_ids=used_top_ids,
                session_recent_top_ids=set(recent_top_ids[-2:]),
                session_pair_embeddings=chosen_pair_embeddings,
                rotation_top_history=rotation_counts,
                rotation_total_count=rotation_total,
                rotation_max_frequency=rotation_max_frequency,
            )

        unused_top_ids = {
            item.id for item in wardrobe_items
            if item.category == "top" and item.id not in used_top_ids
        }
        force_unused = bool(unused_top_ids)

        result = _pick(wardrobe_items, force_unused)
        if not result and force_unused:
            result = _pick(wardrobe_items, False)
        if not result:
            break

        top, bottom = result["items"]
        used_pairs.add((top.id, bottom.id))
        used_top_ids.add(top.id)
        recent_top_ids.append(top.id)
        pair_embedding = result.get("pair_embedding")
        if isinstance(pair_embedding, list) and pair_embedding:
            chosen_pair_embeddings.append(pair_embedding)
        rotation_counts[top.id] = int(rotation_counts.get(top.id, 0)) + 1
        rotation_total += 1

        plan.append(result)

    return plan
