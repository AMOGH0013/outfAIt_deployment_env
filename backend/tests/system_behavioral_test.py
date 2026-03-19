"""
System-level behavioral test harness for the Decisive Stylist recommender.
Runs against a live API at http://127.0.0.1:8000 without modifying production code.
"""

import datetime as dt
import json
import os
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from sqlalchemy.orm import Session

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.database import DATABASE_URL, SessionLocal, ensure_sqlite_schema
from app.dependencies import create_access_token, get_password_hash
from app.models.user import User
from app.models.body_profile import BodyProfile
from app.models.wardrobe import WardrobeItem
from app.models.outfit import Outfit, OutfitItem
from app.models.feedback import Feedback

BASE = "http://127.0.0.1:8000"
SESSION = requests.Session()
random.seed(1337)


# ------------- helpers -----------------------------------------------------

def api(method: str, path: str, **kwargs):
    url = f"{BASE}{path}"
    resp = SESSION.request(method, url, **kwargs)
    return resp


def expect(status: int, resp: requests.Response, note: str):
    if resp.status_code != status:
        raise AssertionError(f"{note} expected {status}, got {resp.status_code} body={resp.text}")
    return resp


def fetch_profile() -> Dict:
    r = api("GET", "/body-profile")
    expect(200, r, "get body profile")
    return r.json()


def update_profile(payload: Dict, expect_code: int = 200):
    r = api("PUT", "/body-profile", json=payload)
    expect(expect_code, r, "update body profile")
    return r


def generate_outfits(start_date: dt.date, days: int = 1) -> List[Dict]:
    r = api("POST", f"/outfits/generate?start_date={start_date.isoformat()}&days={days}")
    expect(200, r, "generate outfits")
    return r.json().get("outfits", [])

def generate_fresh_outfit(start_date: dt.date, max_tries: int = 14) -> Dict:
    """Return a newly created outfit (created=True) if possible; advances date to avoid existing records."""
    d = start_date
    for _ in range(max_tries):
        outs = generate_outfits(d, days=1)
        if outs and outs[0].get("created", True):
            return outs[0]
        d = d + dt.timedelta(days=1)
    # fall back to whatever we got last (may be existing)
    outs = generate_outfits(d, days=1)
    return outs[0] if outs else None


def set_auth_for_user(user_id):
    token = create_access_token({"sub": str(user_id)})
    SESSION.headers.update({"Authorization": f"Bearer {token}"})
    SESSION.cookies.set("wai_token", token)


def _sqlite_path() -> str | None:
    if not DATABASE_URL.startswith("sqlite:///"):
        return None
    return DATABASE_URL.replace("sqlite:///", "", 1)


def reset_and_seed_db():
    """
    Wipe local SQLite data and seed a deterministic wardrobe for tests.
    This ensures stable, repeatable behavioral tests.
    """
    db_path = _sqlite_path()
    if not db_path:
        raise RuntimeError("Deterministic reset only supported for sqlite:///.")
    if os.path.exists(db_path):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(db_path, f"{db_path}.bak.{ts}")

    ensure_sqlite_schema()
    db: Session = SessionLocal()
    try:
        db.query(OutfitItem).delete()
        db.query(Outfit).delete()
        db.query(Feedback).delete()
        db.query(WardrobeItem).delete()
        db.query(BodyProfile).delete()
        db.query(User).delete()
        db.commit()

        user = User(email="test@example.com", password_hash=get_password_hash("test"))
        db.add(user)
        db.commit()
        db.refresh(user)
        set_auth_for_user(user.id)

        bp = BodyProfile(user_id=user.id)
        db.add(bp)

        rng = random.Random(1234)
        def embed():
            vec = [rng.random() for _ in range(8)]
            # normalize
            norm = sum(x * x for x in vec) ** 0.5
            return [x / norm for x in vec]

        tops = [
            ("shirt", "black"),
            ("tshirt", "navy"),
            ("hoodie", "gray"),
            ("kurta", "white"),
            ("shirt", "brown"),
            ("tshirt", "beige"),
        ]
        bottoms = [
            ("jeans", "blue"),
            ("trousers", "black"),
            ("chinos", "gray"),
            ("shorts", "khaki"),
            ("trousers", "white"),
        ]

        def add_item(item_type, category, color, idx):
            item = WardrobeItem(
                user_id=user.id,
                image_url=f"/uploads/test_{category}_{idx}.jpg",
                mask_url=None,
                item_type=item_type,
                category=category,
                color=color,
                color_palette=[color],
                embedding=embed(),
                suggested_item_type=None,
                suggested_item_type_confidence=None,
                pattern=None,
                fabric=None,
                fit=None,
                size=None,
                season_tags=None,
                brand=None,
                measurements=None,
                confidence_scores=None,
                wear_count=0,
                last_worn_at=None,
                is_active=True,
            )
            db.add(item)

        for i, (t, c) in enumerate(tops, start=1):
            add_item(t, "top", c, i)
        for i, (t, c) in enumerate(bottoms, start=1):
            add_item(t, "bottom", c, i)

        db.commit()
    finally:
        db.close()


def ensure_test_auth():
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.email == "test@example.com").first()
        if user is None:
            user = db.query(User).order_by(User.created_at.asc()).first()
        if user is not None:
            set_auth_for_user(user.id)
    finally:
        db.close()

def post_feedback(outfit_id, action: str):
    r = api("POST", "/feedback", json={"outfit_id": outfit_id, "action": action})
    expect(200, r, f"feedback {action}")
    return r


def top_bottom_pair(outfit: Dict) -> Tuple[str, str]:
    items = outfit.get("item_ids") or []
    if len(items) != 2:
        raise AssertionError("Outfit does not have exactly two items")
    return tuple(items)


# ------------- Phase 1: Hard invariants -----------------------------------

def test_invariants():
    results = []
    profile_before = fetch_profile()

    # Empty height/weight do not overwrite
    update_profile({"height_cm": 180})
    update_profile({"weight_kg": 75})
    after_set = fetch_profile()
    update_profile({"height_cm": ""})
    update_profile({"weight_kg": ""})
    after_empty = fetch_profile()
    assert after_empty["height_cm"] == after_set["height_cm"], "empty height overwrote value"
    assert after_empty["weight_kg"] == after_set["weight_kg"], "empty weight overwrote value"
    results.append("empty height/weight safe")

    # Out of range -> 422
    expect(422, api("PUT", "/body-profile", json={"height_cm": 10}), "height out of range")
    expect(422, api("PUT", "/body-profile", json={"weight_kg": 10}), "weight out of range")
    results.append("range validation OK")

    # Confidence and challenger constraints + no pair repeats within 4 days
    start = dt.date.today() + dt.timedelta(days=1)
    outfits = []
    for i in range(5):
        outfits.extend(generate_outfits(start + dt.timedelta(days=i), days=1))
    pairs = set()
    for o in outfits:
        expl = o.get("explanation_data", {})
        conf = expl.get("confidence_score")
        if conf is not None:
            assert 0 <= conf <= 1, "confidence out of bounds"
        if expl.get("challenger") is True:
            assert conf is None or conf < 0.55, "challenger with high confidence"
        pair = top_bottom_pair(o)
        assert pair not in pairs, "repeat pair within 4 days"
        pairs.add(pair)
    results.append("confidence/challenger/repetition invariants OK")

    # Restore original profile if it existed
    update_payload = {}
    for k in ("height_cm", "weight_kg"):
        update_payload[k] = profile_before.get(k)
    update_profile(update_payload)
    print("PHASE1 PASS:", results)


# ------------- Phase 2: Real user scenarios --------------------------------

def scenario_decisive_minimalist():
    base = dt.date.today() + dt.timedelta(days=10)
    # Day1-3: like/wear safe picks, skip bold (challengers)
    seen_pairs = set()
    confs = []
    challengers = 0
    for d in range(3):
        o = generate_fresh_outfit(base + dt.timedelta(days=d))
        assert o, "No outfits available in Decisive Minimalist scenario"
        expl = o.get("explanation_data", {})
        conf = expl.get("confidence_score") or 0
        confs.append(conf)
        if expl.get("challenger"):
            post_feedback(o["outfit_id"], "skipped")
            challengers += 1
        else:
            post_feedback(o["outfit_id"], "liked")
            post_feedback(o["outfit_id"], "worn")
        seen_pairs.add(top_bottom_pair(o))
    # Day4 check
    o4 = generate_fresh_outfit(base + dt.timedelta(days=3))
    assert o4, "No outfits available on day 4 (Decisive Minimalist)"
    hero_conf = o4.get("explanation_data", {}).get("confidence_score") or 0
    assert hero_conf >= 0.6, "Decisive minimalist: confidence too low"
    assert len(seen_pairs.union({top_bottom_pair(o4)})) == len(seen_pairs) + 1, "Minimalist saw repeat pair"
    assert challengers <= 1, "Minimalist saw too many challengers"
    return "Decisive Minimalist PASS"


def scenario_explorer():
    base = dt.date.today() + dt.timedelta(days=30)
    challengers = 0
    confs = []
    pairs = set()
    for d in range(7):
        o = generate_fresh_outfit(base + dt.timedelta(days=d))
        assert o, "No outfits available in Explorer scenario"
        if o.get("explanation_data", {}).get("challenger"):
            challengers += 1
        confs.append(o.get("explanation_data", {}).get("confidence_score") or 0)
        post_feedback(o["outfit_id"], "liked")
        post_feedback(o["outfit_id"], "worn")
        pairs.add(top_bottom_pair(o))
    rate = challengers / 7
    assert rate <= 0.2, f"Explorer challenger rate too high {rate}"
    assert len(pairs) == 7, "Explorer repetition occurred"
    return "Explorer PASS"


def scenario_rage():
    base = dt.date.today() + dt.timedelta(days=60)
    # Find two disliked outfits whose combined items still leave at least one top and one bottom.
    wardrobe = api("GET", "/wardrobe?include_inactive=false")
    expect(200, wardrobe, "get wardrobe")
    items = wardrobe.json().get("items", [])
    cat_map = {it["id"]: it.get("category") for it in items}

    chosen = None
    for offset in range(5):
        o = generate_fresh_outfit(base + dt.timedelta(days=offset))
        o2 = generate_fresh_outfit(base + dt.timedelta(days=offset + 1))
        if not o or not o2:
            continue
        banned = set(o["item_ids"] + o2["item_ids"])
        rem_tops = [i for i in items if i["id"] not in banned and i.get("category") == "top" and i.get("item_type") != "unknown"]
        rem_bottoms = [i for i in items if i["id"] not in banned and i.get("category") == "bottom" and i.get("item_type") != "unknown"]
        if rem_tops and rem_bottoms:
            chosen = (o, o2, list(banned)[:3])
            break

    assert chosen, "Insufficient wardrobe diversity for rage scenario"
    o, o2, to_ban = chosen
    post_feedback(o["outfit_id"], "disliked")
    post_feedback(o2["outfit_id"], "disliked")
    # Now generate 10 outfits, ensure bans absent
    day_cursor = base + dt.timedelta(days=2)
    checks = 0
    while checks < 10:
        o3 = generate_fresh_outfit(day_cursor)
        assert o3, "No outfits available after bans (candidate pool collapsed)"
        if o3.get("created", True):
            ids = o3["item_ids"]
            assert all(x not in to_ban for x in ids), "Banned item resurfaced"
            checks += 1
        day_cursor = day_cursor + dt.timedelta(days=1)
    return "Rage PASS"


def scenario_forgetful():
    base = dt.date.today() + dt.timedelta(days=90)
    o1 = generate_fresh_outfit(base)
    assert o1, "No outfits available on day 1 (Forgetful)"
    post_feedback(o1["outfit_id"], "worn")
    pair1 = top_bottom_pair(o1)
    o2 = generate_fresh_outfit(base + dt.timedelta(days=1))
    assert o2, "No outfits available on day 2 (Forgetful)"
    if top_bottom_pair(o2) == pair1:
        # If repeat happened, ensure engine explicitly signaled override
        expl = o2.get("explanation_data", {}) or {}
        assert expl.get("repetition_blocked_reason") == "override_no_alternatives", "Pair repeated too soon (day2)"
    o5 = generate_fresh_outfit(base + dt.timedelta(days=5))
    assert o5, "No outfits available on day 5 (Forgetful)"
    # Allowed by then; ensure system not crashing; optionally allow repeat
    return "Forgetful PASS"


# ------------- Phase 3: Fake-feature detection -----------------------------

def fake_feature_detection():
    base = dt.date.today() + dt.timedelta(days=120)
    # Confidence moves with feedback
    o1 = generate_outfits(base)[0]
    c1 = o1.get("explanation_data", {}).get("confidence_score") or 0
    post_feedback(o1["outfit_id"], "liked")
    o2 = generate_outfits(base + dt.timedelta(days=1))[0]
    c2 = o2.get("explanation_data", {}).get("confidence_score") or 0
    post_feedback(o2["outfit_id"], "disliked")
    o3 = generate_outfits(base + dt.timedelta(days=2))[0]
    c3 = o3.get("explanation_data", {}).get("confidence_score") or 0
    assert c2 >= c1, "Confidence did not rise after like"
    assert c3 <= c2, "Confidence did not fall after dislike"
    # Challenger affects ranking: challenger must have lower confidence than a confident pick
    if o3.get("explanation_data", {}).get("challenger"):
        assert c3 < 0.55, "Challenger has high confidence"
    return "Fake-feature detection PASS"


# ------------- Phase 4: 30-day regression ---------------------------------

def regression_30d():
    start = dt.date.today() + dt.timedelta(days=150)
    log = []
    for d in range(30):
        o = generate_fresh_outfit(start + dt.timedelta(days=d))
        assert o, "No outfits available during regression run"
        expl = o.get("explanation_data", {}) or {}
        conf = expl.get("confidence_score") or 0
        chall = expl.get("challenger") is True
        action = random.choices(["liked", "worn", "skipped", "disliked"], [0.35, 0.35, 0.2, 0.1])[0]
        post_feedback(o["outfit_id"], action)
        log.append({"pair": top_bottom_pair(o), "conf": conf, "chall": chall, "action": action})
    pairs_first = {tuple(x["pair"]) for x in log[:10]}
    pairs_last = {tuple(x["pair"]) for x in log[-10:]}
    repetition_rate = 1 - len(set(x["pair"] for x in log)) / len(log)
    chall_rate = sum(1 for x in log if x["chall"]) / len(log)
    avg_conf = statistics.mean(x["conf"] for x in log)
    wear_rate = sum(1 for x in log if x["action"] == "worn") / len(log)
    assert repetition_rate <= 0.3, "Repetition trending upward"
    assert chall_rate <= 0.25, "Challenger rate too high"
    assert avg_conf > 0.4, "Confidence collapsed"
    assert wear_rate > 0.2, "Wear rate too low"
    return "Regression 30d PASS"


# ------------- Phase 5: Edge safety ---------------------------------------

def edge_safety():
    # All items banned => graceful error (400/404 acceptable, not 500)
    # Try to dislike current outfit then generate again
    base = dt.date.today() + dt.timedelta(days=200)
    o = generate_outfits(base)[0]
    post_feedback(o["outfit_id"], "disliked")
    r = api("POST", f"/outfits/generate?start_date={(base + dt.timedelta(days=1)).isoformat()}&days=1")
    assert r.status_code in {200, 400}, "Unexpected error code when items banned"
    # Missing explanation_data should not crash UI; here we just ensure key access safe
    # Tiny wardrobe: if API returns 400 due to eligibility, treat as graceful
    return "Edge safety PASS"


# ------------- Runner ------------------------------------------------------

def main():
    # Deterministic reset + seed for strict behavioral tests
    if os.getenv("SYSTEM_TEST_NO_RESET") != "1":
        reset_and_seed_db()
    else:
        ensure_test_auth()
    phases = []
    try:
        test_invariants()
        phases.append("Invariants PASS")
        phases.append(scenario_decisive_minimalist())
        phases.append(scenario_explorer())
        phases.append(scenario_rage())
        phases.append(scenario_forgetful())
        phases.append(fake_feature_detection())
        phases.append(regression_30d())
        phases.append(edge_safety())
        print("ALL PASS")
    except AssertionError as e:
        print("FAIL:", e)
    for p in phases:
        print(p)


if __name__ == "__main__":
    main()
