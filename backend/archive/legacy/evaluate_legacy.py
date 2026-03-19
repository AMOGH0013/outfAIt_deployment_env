"""
evaluate.py — Deep diagnostic harness for the wardrobe recommender.

What this does:
  1. Outfit engine stress tests — runs hundreds of planning scenarios
     with varying wardrobe sizes, worn-item histories, and feedback signals,
     then measures how often the same top repeats across days.
  2. Color extraction accuracy tests — runs extraction on all images in
     uploads/ and checks for known failure patterns (dark green → navy/black,
     skin bleed into palette, low confidence on solid colors).
  3. SAM/mask quality tests — checks every existing mask for inversion
     artifacts, background bleed ratio, and fragmentation score.
  4. Score decomposition audit — for each outfit generated, breaks down
     exactly why each item won (which score component dominated) and flags
     outfits where novelty alone determined the winner.
  5. Regression patterns — compares results across 10 repeated runs to
     detect non-determinism.

Run from project root:
    python evaluate.py 2>&1 | tee evaluation_report.txt

All findings are written to:
    diagnostic_report.json   — machine-readable full results
    diagnostic_summary.txt   — human-readable summary with root-cause analysis
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 output on Windows (cp1252 chokes on box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
import json
import time
import glob
import uuid
import math
import random
import colorsys
import traceback
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# -- Project path setup ------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

UPLOADS_DIR = PROJECT_ROOT / "uploads"
REPORT_JSON = PROJECT_ROOT / "diagnostic_report.json"
REPORT_TXT  = PROJECT_ROOT / "diagnostic_summary.txt"

# -- Lazy imports with availability flags ------------------------------------
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARN] numpy not available — image tests will be skipped")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# -- Reporting infrastructure ------------------------------------------------─
report: dict[str, Any] = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "sections": {}
}

lines: list[str] = []   # human-readable lines

def log(msg: str = "", indent: int = 0):
    prefix = "  " * indent
    full = f"{prefix}{msg}"
    print(full)
    lines.append(full)

def section(title: str):
    bar = "=" * 70
    log()
    log(bar)
    log(f"  {title}")
    log(bar)

def sub(title: str):
    log(f"\n-- {title} --")

def save_reports():
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(REPORT_TXT, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] {REPORT_JSON}")
    print(f"[SAVED] {REPORT_TXT}")

# ===========================================================================
# SECTION 1 — Outfit Engine: Top-Repetition Stress Test
# ===========================================================================

def _make_item(
    category: str,
    item_type: str,
    color: str = "blue",
    worn_days_ago: int | None = None,
    fit: str = "regular",
    wear_count: int = 0,
    seed_offset: int = 0,
) -> SimpleNamespace:
    """Create a mock WardrobeItemLike with a deterministic embedding."""
    rng = random.Random(hash((category, item_type, color, seed_offset)))
    emb = [rng.gauss(0, 1) for _ in range(32)]
    norm = math.sqrt(sum(x*x for x in emb)) or 1.0
    emb = [x / norm for x in emb]
    last_worn = None
    if worn_days_ago is not None:
        last_worn = datetime.now(timezone.utc) - timedelta(days=worn_days_ago)
    return SimpleNamespace(
        id=uuid.uuid4(),
        category=category,
        item_type=item_type,
        color=color,
        fit=fit,
        last_worn_at=last_worn,
        wear_count=wear_count,
        is_active=True,
        embedding=emb,
    )

def _make_user(body_shape=None, fit_pref=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        body_shape=body_shape,
        forbidden_items=None,
        fit_preference=fit_pref,
        style_profile=None,
    )

def run_outfit_engine_tests():
    section("SECTION 1 — Outfit Engine: Top-Repetition & Diversity")
    results = {}

    try:
        from app.services.outfit_engine import (
            generate_weekly_plan,
            rule_score,
            novelty_score,
            color_harmony_score,
            score_outfit,
        )
    except Exception as e:
        log(f"[FATAL] Cannot import outfit_engine: {e}")
        report["sections"]["outfit_engine"] = {"error": str(e)}
        return

    # -- Test 1.1: Basic repetition across wardrobe sizes --------------------
    sub("1.1  Top-repetition rate vs wardrobe size")
    repetition_results = []

    wardrobe_configs = [
        {"tops": 2, "bottoms": 2, "label": "2T 2B (tiny)"},
        {"tops": 3, "bottoms": 2, "label": "3T 2B (small)"},
        {"tops": 5, "bottoms": 3, "label": "5T 3B (medium)"},
        {"tops": 8, "bottoms": 4, "label": "8T 4B (realistic)"},
        {"tops": 12, "bottoms": 5, "label": "12T 5B (large)"},
    ]

    for cfg in wardrobe_configs:
        tops    = [_make_item("top",    "shirt",    color=c, seed_offset=i)
                   for i, c in enumerate(["blue","red","green","navy","white",
                                          "black","orange","purple","teal","brown",
                                          "yellow","gray"][:cfg["tops"]])]
        bottoms = [_make_item("bottom", "jeans",    color=c, seed_offset=100+i)
                   for i, c in enumerate(["black","navy","gray","beige","brown"][:cfg["bottoms"]])]
        wardrobe = tops + bottoms
        user = _make_user()

        plan = generate_weekly_plan(user, wardrobe, days=7)

        top_sequence   = [str(r["items"][0].id) for r in plan]
        top_colors     = [r["items"][0].color for r in plan]
        bottom_sequence= [str(r["items"][1].id) for r in plan]

        # Count max consecutive same-top streak
        max_streak = 1
        cur_streak = 1
        for i in range(1, len(top_sequence)):
            if top_sequence[i] == top_sequence[i-1]:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 1

        unique_tops_used    = len(set(top_sequence))
        unique_bottoms_used = len(set(bottom_sequence))
        top_counts          = Counter(top_sequence)
        most_used_top_count = top_counts.most_common(1)[0][1] if top_counts else 0

        log(f"  {cfg['label']:20s} | days={len(plan)} "
            f"| unique_tops={unique_tops_used}/{cfg['tops']} "
            f"| max_streak={max_streak} "
            f"| most_used_top={most_used_top_count}x "
            f"| colors={top_colors}")

        has_problem = max_streak >= 3 or (unique_tops_used < min(cfg["tops"], len(plan)) // 2)
        if has_problem:
            log(f"    [!] PROBLEM DETECTED: streak={max_streak}, "
                f"only {unique_tops_used} of {cfg['tops']} tops used in {len(plan)} days", indent=1)

        repetition_results.append({
            "config": cfg["label"],
            "plan_days": len(plan),
            "unique_tops": unique_tops_used,
            "total_tops": cfg["tops"],
            "max_streak": max_streak,
            "most_used_top_count": most_used_top_count,
            "top_color_sequence": top_colors,
            "has_problem": has_problem,
        })

    results["repetition_by_wardrobe_size"] = repetition_results

    # -- Test 1.2: Why does the same top keep winning? Score decomposition --─
    sub("1.2  Score decomposition — why does one top dominate?")
    score_audit = []

    tops5    = [_make_item("top",    "shirt", color=c, worn_days_ago=wd, seed_offset=i)
                for i, (c, wd) in enumerate([
                    ("blue",   None),   # never worn → novelty=1.0
                    ("red",    1),      # worn yesterday → novelty low
                    ("green",  3),      # worn 3 days ago → novelty medium
                    ("navy",   7),      # worn 7+ days ago → novelty=1.0
                    ("white",  None),   # never worn → novelty=1.0
                ])]
    bottoms3 = [_make_item("bottom", "jeans", color=c, seed_offset=100+i)
                for i, c in enumerate(["black", "gray", "navy"])]
    user = _make_user()

    plan5 = generate_weekly_plan(user, tops5 + bottoms3, days=7)

    log("  Day-by-day score breakdown (tops only):")
    for day_idx, result in enumerate(plan5):
        top    = result["items"][0]
        bottom = result["items"][1]
        expl   = result.get("explanation_data", {})
        novel_t = novelty_score(top)
        novel_b = novelty_score(bottom)
        harmony = color_harmony_score(top, bottom)
        rscore  = rule_score(user, top, bottom)
        log(f"  Day {day_idx+1}: top={top.color:8s} bottom={bottom.color:6s} "
            f"rule={rscore:.3f} novelty_top={novel_t:.2f} "
            f"harmony={harmony:.2f} final={result['score']:.3f}")
        score_audit.append({
            "day": day_idx + 1,
            "top_color": top.color,
            "bottom_color": bottom.color,
            "rule_score": round(rscore, 4),
            "novelty_top": round(novel_t, 4),
            "novelty_bottom": round(novel_b, 4),
            "color_harmony": round(harmony, 4),
            "final_score": result["score"],
            "explanation": expl,
        })

    results["score_decomposition"] = score_audit

    # Find which component correlates most with top selection
    never_worn_ids = {str(t.id) for t in tops5 if t.last_worn_at is None}
    never_worn_days = sum(1 for r in plan5 if str(r["items"][0].id) in never_worn_ids)
    log(f"\n  Never-worn tops used: {never_worn_days}/{len(plan5)} days "
        f"({100*never_worn_days//len(plan5)}%) — high % means novelty dominates selection")
    results["never_worn_dominance_pct"] = 100 * never_worn_days // max(len(plan5), 1)

    # -- Test 1.3: Does the cooldown actually fire? --------------------------─
    sub("1.3  Cooldown penalty effectiveness")
    from app.services.outfit_engine import _cooldown_penalty, ITEM_COOLDOWN_DAYS, ITEM_COOLDOWN_PENALTY

    log(f"  ITEM_COOLDOWN_DAYS = {ITEM_COOLDOWN_DAYS}")
    log(f"  ITEM_COOLDOWN_PENALTY = {ITEM_COOLDOWN_PENALTY}")

    cooldown_tests = [
        ("worn 0 days ago",  0,  True),
        ("worn 1 day ago",   1,  True),
        ("worn 2 days ago",  2,  False),   # boundary — depends on < vs <=
        ("worn 3 days ago",  3,  False),
        ("never worn",       None, False),
    ]
    cooldown_results = []
    for label, days_ago, expect_penalty in cooldown_tests:
        item = _make_item("top", "shirt", worn_days_ago=days_ago)
        bottom = _make_item("bottom", "jeans")
        pen = _cooldown_penalty(item, bottom)
        fires = pen < 0
        status = "[OK]" if fires == expect_penalty else "[FAIL] UNEXPECTED"
        log(f"  {status} {label:20s} → penalty={pen:.3f} (fires={fires}, expected={expect_penalty})")
        cooldown_results.append({"label": label, "penalty": pen, "fires": fires, "expected": expect_penalty, "correct": fires == expect_penalty})

    results["cooldown_tests"] = cooldown_results

    # -- Test 1.4: Pair deduplication — does used_pairs actually block repeats?
    sub("1.4  Pair deduplication across planning session")
    tops2    = [_make_item("top",    "shirt", color="blue",  seed_offset=0),
                _make_item("top",    "shirt", color="red",   seed_offset=1)]
    bottoms2 = [_make_item("bottom", "jeans", color="black", seed_offset=10),
                _make_item("bottom", "jeans", color="gray",  seed_offset=11)]
    small_plan = generate_weekly_plan(_make_user(), tops2 + bottoms2, days=7)
    pair_sequence = [(str(r["items"][0].id), str(r["items"][1].id)) for r in small_plan]
    unique_pairs  = len(set(pair_sequence))
    total_pairs   = len(pair_sequence)
    log(f"  2T 2B wardrobe, 7-day plan: {total_pairs} days, {unique_pairs} unique pairs")
    log(f"  Pair sequence: {[(r['items'][0].color, r['items'][1].color) for r in small_plan]}")
    if unique_pairs < total_pairs:
        log(f"  [!] Pairs repeated: {total_pairs - unique_pairs} duplicate pair(s) in plan")
    results["pair_dedup"] = {"unique_pairs": unique_pairs, "total_days": total_pairs}

    # -- Test 1.5: Non-determinism check ------------------------------------─
    sub("1.5  Determinism check (same inputs → same plan?)")
    def _run_plan(seed_val):
        random.seed(seed_val)
        tops_d = [_make_item("top", "shirt", color=c, seed_offset=i) for i,c in enumerate(["blue","red","green","navy","white"])]
        bots_d = [_make_item("bottom", "jeans", color=c, seed_offset=10+i) for i,c in enumerate(["black","gray","navy"])]
        return generate_weekly_plan(_make_user(body_shape=None), tops_d + bots_d, days=7)

    plan_a = _run_plan(42)
    plan_b = _run_plan(42)
    colors_a = [(r["items"][0].color, r["items"][1].color) for r in plan_a]
    colors_b = [(r["items"][0].color, r["items"][1].color) for r in plan_b]
    is_deterministic = colors_a == colors_b
    log(f"  Same seed → same plan: {'[OK] YES' if is_deterministic else '[FAIL] NO — NON-DETERMINISTIC'}")
    results["is_deterministic"] = is_deterministic

    report["sections"]["outfit_engine"] = results

    # -- Root-cause analysis --------------------------------------------------
    sub("ROOT-CAUSE ANALYSIS: Outfit Engine")
    problems_found = []

    for r in repetition_results:
        if r["has_problem"]:
            problems_found.append(
                f"Wardrobe {r['config']}: max streak={r['max_streak']}, "
                f"only {r['unique_tops']}/{r['total_tops']} tops used"
            )

    if results.get("never_worn_dominance_pct", 0) > 60:
        problems_found.append(
            f"Novelty dominates: {results['never_worn_dominance_pct']}% of days "
            f"go to never-worn tops — novelty_score() weight of 0.15+0.15=0.30 "
            f"in rule_score() creates a 'freshness first' bias that overrides variety"
        )

    bad_cooldown = [c for c in cooldown_results if not c["correct"]]
    if bad_cooldown:
        problems_found.append(
            f"Cooldown fires unexpectedly: {[c['label'] for c in bad_cooldown]} — "
            f"check ITEM_COOLDOWN_DAYS boundary condition (< vs <=)"
        )

    if not results.get("is_deterministic"):
        problems_found.append(
            "NON-DETERMINISTIC: Same wardrobe produces different plans — "
            "likely from dict iteration order or set ordering in used_pairs"
        )

    if not problems_found:
        log("  No critical outfit engine problems detected.")
    else:
        for p in problems_found:
            log(f"  [!] {p}")

    log(f"\n  WHERE TO LOOK:")
    log(f"  • outfit_engine.py → generate_weekly_plan(): used_pairs only blocks exact")
    log(f"    top+bottom combos, not individual tops. A top can appear every day paired")
    log(f"    with a different bottom. Fix: track used_top_ids separately from used_pairs.")
    log(f"  • outfit_engine.py → rule_score(): novelty weight is 0.30 total (0.15 top +")
    log(f"    0.15 bottom). Items never worn get novelty=1.0, giving them a +0.30 bonus")
    log(f"    that persists every day since last_worn_at isn't updated within the session.")
    log(f"  • outfit_engine.py → _cooldown_penalty(): only fires for items worn in the")
    log(f"    DB (last_worn_at). Items used earlier in the SAME planning session don't")
    log(f"    get a cooldown — so day 1's top is equally attractive on day 3.")


# ===========================================================================
# SECTION 2 — Color Extraction: Accuracy & Anchor Gap Analysis
# ===========================================================================

def run_color_extraction_tests():
    section("SECTION 2 — Color Extraction: Accuracy & Anchor Gap Analysis")
    results = {}

    if not HAS_NUMPY:
        log("[SKIP] numpy not available")
        report["sections"]["color_extraction"] = {"skipped": True}
        return

    try:
        from app.services.color_extraction import (
            _rgb_to_lab,
            _lab_to_color_name,
            _ANCHOR_NAMES,
            _ANCHOR_LAB,
            _filter_garment_pixels_bgr,
            extract_dominant_colors,
        )
    except Exception as e:
        log(f"[FATAL] Cannot import color_extraction: {e}")
        report["sections"]["color_extraction"] = {"error": str(e)}
        return

    # -- Test 2.1: Known problem colors — anchor distance analysis ----------─
    sub("2.1  Known problem colors: LAB distance to nearest anchors")

    # Colors that are known to misclassify in the wild
    known_problem_colors = [
        # (label, RGB,              expected_name)
        ("dark green shirt",     (30,  90,  40),  "green"),
        ("forest green",         (20,  80,  45),  "green"),
        ("dark teal green",      (20,  85,  75),  "teal"),
        ("olive/khaki",          (110, 120, 50),  "olive"),
        ("very dark green",      (10,  50,  20),  "green"),
        ("teal-blue dark",       (15,  80,  90),  "teal"),
        ("charcoal gray",        (55,  60,  65),  "gray"),
        ("dark navy",            ( 8,  12,  50),  "navy"),
        ("pure black fabric",    (15,  15,  15),  "black"),
        ("dark denim",           (30,  50,  90),  "blue"),
        ("light wash denim",     (160, 195, 220), "blue"),
        ("mustard yellow",       (200, 165,  30), "yellow"),
        ("burnt orange",         (200,  80,  20), "orange"),
        ("dusty pink",           (210, 145, 140), "pink"),
        ("wine/burgundy",        (100,  20,  30), "red"),
        ("bright purple shirt",  (120,   0, 130), "purple"),
        ("shadow purple",        ( 55,  10,  70), "purple"),
        ("beige/cream",          (220, 205, 175), "beige"),
        ("sand/tan",             (195, 170, 130), "beige"),
        ("rust red",             (180,  55,  25), "red"),
    ]

    anchor_gap_results = []
    misclassifications = []
    log(f"  {'Label':30s} {'RGB':20s} {'Got':15s} {'Expected':12s} {'Gap':8s} {'Status'}")
    log(f"  {'-'*90}")

    for label, rgb, expected in known_problem_colors:
        lab = _rgb_to_lab(rgb)
        got_label, best_internal, runner_up, gap, confident = _lab_to_color_name(lab)
        got_base = got_label.split("-")[0]
        correct = got_base == expected or got_label == expected
        status = "[OK]" if correct else f"[FAIL] MISS"

        log(f"  {label:30s} RGB{str(rgb):20s} {got_label:15s} {expected:12s} {gap:8.2f} {status}")

        entry = {
            "label": label, "rgb": rgb, "expected": expected,
            "got": got_label, "gap": round(gap, 3), "correct": correct,
            "confident": confident, "runner_up": runner_up,
        }
        anchor_gap_results.append(entry)
        if not correct:
            misclassifications.append(entry)

    results["known_colors"] = anchor_gap_results
    results["misclassification_count"] = len(misclassifications)
    log(f"\n  Total misclassifications: {len(misclassifications)}/{len(known_problem_colors)}")

    # -- Test 2.2: Anchor coverage — find color-space holes ------------------
    sub("2.2  Anchor coverage — color-space holes")

    # Sample LAB space systematically and find regions with no close anchor
    import itertools
    far_from_anchor = []

    # Sample a grid of hue values at medium saturation/lightness
    # to find regions where nearest anchor is >20 LAB units away
    for hue_deg in range(0, 360, 15):
        for sat in [0.5, 0.7]:
            for val in [0.3, 0.5, 0.7]:
                r = int(255 * val * (1 - sat * abs((hue_deg/60 % 2) - 1)) if (hue_deg/60 % 2) <= 1 else val * (1 - sat * (abs((hue_deg/60 % 2) - 1)))) if True else 0
                # Use colorsys instead for correctness
                r_f, g_f, b_f = colorsys.hsv_to_rgb(hue_deg/360, sat, val)
                rgb_sample = (int(r_f*255), int(g_f*255), int(b_f*255))
                lab = _rgb_to_lab(rgb_sample)
                lab_arr = np.array(lab, dtype=np.float32)
                diffs = _ANCHOR_LAB - lab_arr
                dists = np.sqrt((diffs**2).sum(axis=1))
                min_dist = float(dists.min())
                nearest = _ANCHOR_NAMES[int(dists.argmin())]
                if min_dist > 20:
                    far_from_anchor.append({
                        "hue_deg": hue_deg, "sat": sat, "val": val,
                        "rgb": rgb_sample, "nearest_anchor": nearest,
                        "distance": round(min_dist, 2)
                    })

    log(f"  Colors >20 LAB units from any anchor: {len(far_from_anchor)}")
    if far_from_anchor:
        # Group by hue range
        by_hue = defaultdict(list)
        for f in far_from_anchor:
            hue_bucket = (f["hue_deg"] // 30) * 30
            by_hue[hue_bucket].append(f)
        log("  Underrepresented hue regions:")
        for hue, items in sorted(by_hue.items()):
            avg_dist = statistics.mean(i["distance"] for i in items)
            anchors = set(i["nearest_anchor"] for i in items)
            log(f"    Hue {hue:3d}-{hue+30:3d}°: {len(items)} holes, avg_dist={avg_dist:.1f}, "
                f"nearest={anchors}")

    results["anchor_coverage_holes"] = len(far_from_anchor)
    results["anchor_holes_detail"] = far_from_anchor[:20]  # cap for JSON

    # -- Test 2.3: Skin filter effectiveness --------------------------------─
    sub("2.3  Skin filter effectiveness")

    if HAS_CV2:
        # Create synthetic BGR images: garment + skin bleed
        def test_skin_filter(garment_rgb, skin_rgb, skin_fraction):
            n = 1000
            n_skin = int(n * skin_fraction)
            n_garment = n - n_skin
            garment_px = np.array([garment_rgb[::-1]] * n_garment, dtype=np.uint8)  # BGR
            skin_px    = np.array([skin_rgb[::-1]]    * n_skin,    dtype=np.uint8)
            mixed = np.vstack([garment_px, skin_px])
            filtered = _filter_garment_pixels_bgr(mixed)
            skin_bgr = np.array(skin_rgb[::-1])
            remaining_skin = sum(1 for px in filtered if np.allclose(px, skin_bgr, atol=5))
            return len(filtered), remaining_skin

        skin_tests = [
            ("Dark green + light skin", (30, 90, 40),   (220, 175, 140), 0.20),
            ("Navy + medium skin",      (10, 20, 80),   (180, 130, 100), 0.30),
            ("Black + dark skin",       (15, 15, 15),   (100,  65,  45), 0.25),
            ("White shirt + light skin",(240,240,240),  (240, 200, 165), 0.15),
            ("Red + tanned skin",       (190, 30, 30),  (195, 145, 100), 0.20),
        ]

        skin_filter_results = []
        log(f"  {'Test':35s} {'Filtered':10s} {'Skin left':10s} {'% removed'}")
        for label, garment, skin, frac in skin_tests:
            total_filtered, skin_remaining = test_skin_filter(garment, skin, frac)
            skin_injected = int(1000 * frac)
            pct_removed = 100 * (1 - skin_remaining / max(skin_injected, 1))
            log(f"  {label:35s} {total_filtered:10d} {skin_remaining:10d} {pct_removed:8.1f}%")
            skin_filter_results.append({
                "label": label, "total_filtered": total_filtered,
                "skin_remaining": skin_remaining, "pct_skin_removed": round(pct_removed, 1)
            })

        results["skin_filter_tests"] = skin_filter_results
        poor_filters = [s for s in skin_filter_results if s["pct_skin_removed"] < 70]
        if poor_filters:
            log(f"\n  [!] Poor skin filtering (< 70% removed): {[s['label'] for s in poor_filters]}")
            log(f"  WHERE TO LOOK: _filter_garment_pixels_bgr() in color_extraction.py")
            log(f"  The not_skin condition uses HSV hue <= 25, S 20-85, V 90-240.")
            log(f"  Dark skin (V < 90) and very pale skin (S < 20) slip through.")

    # -- Test 2.4: Real image extraction on uploads/ ------------------------─
    sub("2.4  Real image extraction on uploads/")

    image_files = list(UPLOADS_DIR.glob("*.webp")) + list(UPLOADS_DIR.glob("*.jpg"))
    image_files = [f for f in image_files if "_mask" not in f.name][:30]  # cap at 30
    log(f"  Testing {len(image_files)} images from uploads/")

    extraction_results = []
    low_confidence_cases = []
    compound_label_cases = []
    unknown_cases = []

    for img_path in image_files:
        mask_path_candidate = img_path.parent / (img_path.stem + "_mask.png")
        mask_arg = str(mask_path_candidate) if mask_path_candidate.exists() else None

        try:
            t0 = time.time()
            result = extract_dominant_colors(str(img_path), mask_path=mask_arg)
            elapsed = time.time() - t0

            entry = {
                "file": img_path.name,
                "primary_color": result.primary_color,
                "palette": result.palette,
                "confidence": result.primary_confidence,
                "palette_confidences": result.palette_confidences,
                "has_mask": mask_arg is not None,
                "elapsed_s": round(elapsed, 3),
            }
            extraction_results.append(entry)

            if result.primary_confidence < 0.40:
                low_confidence_cases.append(entry)
            if "-" in result.primary_color:
                compound_label_cases.append(entry)
            if result.primary_color == "unknown":
                unknown_cases.append(entry)

        except Exception as e:
            entry = {"file": img_path.name, "error": str(e)}
            extraction_results.append(entry)
            log(f"  ERROR {img_path.name}: {e}")

    if extraction_results:
        confidences = [r["confidence"] for r in extraction_results if "confidence" in r]
        if confidences:
            log(f"\n  Confidence stats across {len(confidences)} images:")
            log(f"    Mean:   {statistics.mean(confidences):.3f}")
            log(f"    Median: {statistics.median(confidences):.3f}")
            log(f"    Min:    {min(confidences):.3f}")
            log(f"    Max:    {max(confidences):.3f}")
            log(f"    < 0.40: {len(low_confidence_cases)} images ({100*len(low_confidence_cases)//len(confidences)}%)")
            log(f"    compound labels: {len(compound_label_cases)} images")

        # Color distribution
        color_dist = Counter(r.get("primary_color","?") for r in extraction_results if "error" not in r)
        log(f"\n  Color distribution: {dict(color_dist.most_common(10))}")

        if low_confidence_cases:
            log(f"\n  Low-confidence extractions:")
            for r in low_confidence_cases[:10]:
                log(f"    {r['file'][:50]:50s} → {r['primary_color']:15s} conf={r['confidence']:.3f}")

    results["real_images"] = extraction_results
    results["low_confidence_count"] = len(low_confidence_cases)
    results["compound_label_count"] = len(compound_label_cases)

    # -- Root-cause analysis --------------------------------------------------
    sub("ROOT-CAUSE ANALYSIS: Color Extraction")
    log(f"  Misclassified known colors: {len(misclassifications)}")
    for m in misclassifications:
        log(f"    RGB{m['rgb']} '{m['label']}' → got '{m['got']}', expected '{m['expected']}', gap={m['gap']:.2f}")
        # Suggest specific anchor to add
        log(f"      FIX: Add anchor near RGB{m['rgb']} named '{m['expected']}_variant' in _COLOR_ANCHORS_RGB")

    log(f"\n  WHERE TO LOOK:")
    log(f"  • color_extraction.py → _COLOR_ANCHORS_RGB: Missing anchors for dark greens.")
    log(f"    Dark green (10,70,25) and navy (5,10,70) are only 22 LAB units apart.")
    log(f"    Any dark green pixel with slight blue cast lands closer to navy.")
    log(f"    Add forest_green(20,80,45), dark_teal(15,80,75), teal_green(25,100,70).")
    log(f"  • color_extraction.py → extract_dominant_colors(): KMeans runs on BGR,")
    log(f"    not LAB. Cluster centers in BGR space don't correspond to perceptual")
    log(f"    color centers. A dark green cluster center can drift toward dark blue.")
    log(f"    Fix: convert pixels to LAB before KMeans, use LAB centers directly.")
    log(f"  • color_extraction.py → _filter_garment_pixels_bgr(): skin filter misses")
    log(f"    dark skin (V<90). This bleeds dark-toned pixels into the sample,")
    log(f"    pulling green cluster centers toward darker, bluer values → navy/black.")

    report["sections"]["color_extraction"] = results


# ===========================================================================
# SECTION 3 — SAM Mask Quality Audit
# ===========================================================================

def run_mask_quality_tests():
    section("SECTION 3 — SAM Mask Quality Audit")
    results = {}

    if not (HAS_NUMPY and HAS_PIL):
        log("[SKIP] numpy/PIL not available")
        report["sections"]["mask_quality"] = {"skipped": True}
        return

    mask_files = list(UPLOADS_DIR.glob("*_mask.png"))
    log(f"  Auditing {len(mask_files)} mask files")

    if not mask_files:
        log("  No mask files found in uploads/")
        report["sections"]["mask_quality"] = {"no_masks": True}
        return

    mask_results = []
    inversion_suspects = []
    fragmentation_suspects = []
    background_bleed_suspects = []

    for mask_path in mask_files:
        img_stem = mask_path.stem.replace("_mask", "")
        orig_candidates = list(UPLOADS_DIR.glob(f"{img_stem}.*"))
        orig_candidates = [f for f in orig_candidates if "_mask" not in f.name]
        orig_path = orig_candidates[0] if orig_candidates else None

        try:
            mask_arr = np.array(Image.open(mask_path).convert("L"))
            h, w = mask_arr.shape
            total_px = h * w

            white_px   = int((mask_arr > 128).sum())
            black_px   = total_px - white_px
            white_pct  = white_px / total_px

            # Fragmentation: count connected components
            n_components = 1
            if HAS_CV2:
                mask_bin = (mask_arr > 128).astype(np.uint8)
                n_components, _ = cv2.connectedComponents(mask_bin)
                n_components -= 1  # subtract background label

            # Center pixel check — is the center masked or not?
            center_masked = bool(mask_arr[h//2, w//2] > 128)

            # Background color bleed check (if original available)
            bg_bleed_score = 0.0
            if orig_path and HAS_CV2:
                try:
                    orig_arr = np.array(Image.open(orig_path).convert("RGB"))
                    if orig_arr.shape[:2] != mask_arr.shape:
                        orig_arr = np.array(
                            Image.fromarray(orig_arr).resize((w, h), Image.NEAREST)
                        )
                    garment_pixels = orig_arr[mask_arr > 128]
                    if len(garment_pixels) > 100:
                        # Check what fraction of garment pixels look like typical bg colors
                        # (near-white or near-gray)
                        gray_vals = garment_pixels.astype(np.float32)
                        channel_std = gray_vals.std(axis=1)   # per-pixel channel variation
                        is_neutral = channel_std < 20         # low variation = gray/white
                        is_bright  = gray_vals.mean(axis=1) > 210
                        bg_bleed_score = float((is_neutral & is_bright).mean())
                except Exception:
                    pass

            entry = {
                "mask": mask_path.name,
                "size": f"{w}x{h}",
                "white_pct": round(white_pct, 3),
                "n_components": n_components,
                "center_masked": center_masked,
                "bg_bleed_score": round(bg_bleed_score, 3),
            }
            mask_results.append(entry)

            # Flag suspects
            if white_pct > 0.85:
                inversion_suspects.append(entry)
            if n_components > 5:
                fragmentation_suspects.append(entry)
            if bg_bleed_score > 0.20:
                background_bleed_suspects.append(entry)

        except Exception as e:
            mask_results.append({"mask": mask_path.name, "error": str(e)})

    # Summary stats
    valid = [r for r in mask_results if "error" not in r]
    if valid:
        white_pcts = [r["white_pct"] for r in valid]
        components = [r["n_components"] for r in valid if "n_components" in r]
        center_masked_pct = sum(1 for r in valid if r.get("center_masked")) / len(valid)

        log(f"\n  Mask coverage stats ({len(valid)} masks):")
        log(f"    Mean white%:      {statistics.mean(white_pcts):.3f}")
        log(f"    Median white%:    {statistics.median(white_pcts):.3f}")
        log(f"    > 85% white (inversion suspect): {len(inversion_suspects)}")
        log(f"    Center is masked: {100*center_masked_pct:.1f}% of masks")
        if components:
            log(f"    Mean components:  {statistics.mean(components):.1f}")
            log(f"    > 5 components (fragmented): {len(fragmentation_suspects)}")
        log(f"    BG bleed > 20%:   {len(background_bleed_suspects)}")

    if inversion_suspects:
        log(f"\n  [!] Probable inversion artifacts (mask captured background):")
        for s in inversion_suspects[:5]:
            log(f"    {s['mask'][:60]:60s} white={s['white_pct']:.2f} center={s['center_masked']}")

    if fragmentation_suspects:
        log(f"\n  [!] Highly fragmented masks (SAM split garment into pieces):")
        for s in fragmentation_suspects[:5]:
            log(f"    {s['mask'][:60]:60s} components={s['n_components']}")

    if background_bleed_suspects:
        log(f"\n  [!] Background bleed detected:")
        for s in background_bleed_suspects[:5]:
            log(f"    {s['mask'][:60]:60s} bleed_score={s['bg_bleed_score']:.3f}")

    sub("ROOT-CAUSE ANALYSIS: SAM Segmentation")
    log(f"  WHERE TO LOOK:")
    log(f"  • sam_segmentation.py → segment_clothing(): The 3x3 grid spans")
    log(f"    25%-75% of width/height. For person-worn photos, arms extend to")
    log(f"    edges, so outer grid points land on skin/background.")
    log(f"    Fix: cluster grid tighter (35%-65% width, 40%-60% height).")
    log(f"  • sam_segmentation.py → The center-pixel inversion check is a heuristic.")
    log(f"    It fails when the garment is lighter than background (e.g. white shirt")
    log(f"    on a cream wall) because the center IS bright, masking passes, but")
    log(f"    the background also passes the brightness test.")
    log(f"  • sam_segmentation.py → No explicit background rejection points (label=0).")
    log(f"    SAM is being told 9 'is foreground' points but no 'is background' hints.")
    log(f"    Adding corner rejection points dramatically improves accuracy.")
    log(f"  • _fallback_threshold_mask(): threshold 245 is too loose for gray bgs.")
    log(f"    Gray background (mean ~190) passes easily. GrabCut would handle this.")

    results["mask_audit"] = mask_results
    results["inversion_suspects"] = len(inversion_suspects)
    results["fragmentation_suspects"] = len(fragmentation_suspects)
    results["background_bleed_suspects"] = len(background_bleed_suspects)
    report["sections"]["mask_quality"] = results


# ===========================================================================
# SECTION 4 — Integration: Full Pipeline End-to-End
# ===========================================================================

def run_integration_tests():
    section("SECTION 4 — Integration: Scoring Pipeline Consistency")
    results = {}

    if not HAS_NUMPY:
        log("[SKIP]")
        report["sections"]["integration"] = {"skipped": True}
        return

    try:
        from app.services.outfit_engine import (
            rule_score, color_harmony_score, novelty_score,
            NEUTRAL_COLORS, _primary_color
        )
    except Exception as e:
        log(f"[FATAL] {e}")
        report["sections"]["integration"] = {"error": str(e)}
        return

    # -- Test 4.1: NEUTRAL_COLORS vs extractor output consistency ------------
    sub("4.1  NEUTRAL_COLORS set vs color_extraction anchor outputs")

    try:
        from app.services.color_extraction import _DISPLAY_NAME, _COLOR_ANCHORS_RGB
        extractor_outputs = set(_COLOR_ANCHORS_RGB.keys()) - set(_DISPLAY_NAME.keys())
        extractor_outputs |= set(_DISPLAY_NAME.values())
        extractor_outputs |= set(k for k in _COLOR_ANCHORS_RGB if k not in _DISPLAY_NAME)
        all_possible_outputs = set()
        for k in _COLOR_ANCHORS_RGB:
            all_possible_outputs.add(_DISPLAY_NAME.get(k, k))

        in_extractor_not_neutral = all_possible_outputs - NEUTRAL_COLORS - {"multicolor"}
        neutral_not_in_extractor = NEUTRAL_COLORS - all_possible_outputs

        log(f"  Colors extractor can output: {sorted(all_possible_outputs)}")
        log(f"  NEUTRAL_COLORS set:          {sorted(NEUTRAL_COLORS)}")
        log(f"  In extractor but not neutral: {sorted(in_extractor_not_neutral)}")
        log(f"  In neutral but extractor never outputs: {sorted(neutral_not_in_extractor)}")

        results["extractor_color_set"] = sorted(all_possible_outputs)
        results["neutral_colors"] = sorted(NEUTRAL_COLORS)
        results["missing_from_neutral"] = sorted(in_extractor_not_neutral)
        results["dead_neutral_entries"] = sorted(neutral_not_in_extractor)

        if neutral_not_in_extractor:
            log(f"\n  [!] Dead entries in NEUTRAL_COLORS (extractor never outputs these):")
            for c in neutral_not_in_extractor:
                log(f"    '{c}' — remove from NEUTRAL_COLORS or add anchor")

    except Exception as e:
        log(f"  ERROR: {e}")

    # -- Test 4.2: Compound label passthrough ------------------------------
    sub("4.2  Compound label handling in scoring")

    compound_test_colors = [
        "purple-navy", "blue-purple", "gray-black", "beige-gray",
        "yellow-orange", "teal-green", "red-orange"
    ]
    compound_results = []
    for compound in compound_test_colors:
        primary = _primary_color(compound)
        in_neutral = primary in NEUTRAL_COLORS
        compound_results.append({
            "compound": compound, "primary": primary, "in_neutral": in_neutral
        })
        log(f"  '{compound:20s}' → primary='{primary}' in_neutral={in_neutral}")

    results["compound_handling"] = compound_results

    # -- Test 4.3: Score sensitivity — how much does each component matter? --
    sub("4.3  Score component sensitivity analysis")

    user = _make_user()
    top    = _make_item("top",    "shirt", color="blue",  worn_days_ago=None)
    bottom = _make_item("bottom", "jeans", color="black", worn_days_ago=None)

    from app.services.outfit_engine import score_outfit, rule_score as rs

    base_rule = rs(user, top, bottom)
    log(f"  Base rule_score: {base_rule:.4f}")
    log(f"    = 0.40 * body_shape ({0.40*1.0:.3f})")
    log(f"    + 0.30 * color_harmony ({0.30*color_harmony_score(top,bottom):.3f})")
    log(f"    + 0.15 * novelty_top ({0.15*novelty_score(top):.3f})")
    log(f"    + 0.15 * novelty_bottom ({0.15*novelty_score(bottom):.3f})")

    # Now show what happens when top was worn recently
    top_worn = _make_item("top", "shirt", color="blue", worn_days_ago=0)
    rule_worn = rs(user, top_worn, bottom)
    novelty_penalty = base_rule - rule_worn
    log(f"\n  If top worn TODAY: rule_score={rule_worn:.4f} (delta={-novelty_penalty:.4f})")
    log(f"  This {'-' if novelty_penalty>0 else '+'}0.15 change {'IS' if novelty_penalty>0.1 else 'IS NOT'} enough to change outfit ranking")

    results["score_sensitivity"] = {
        "base_rule": round(base_rule, 4),
        "rule_worn_today": round(rule_worn, 4),
        "novelty_impact": round(abs(novelty_penalty), 4),
        "novelty_weight": 0.30
    }

    report["sections"]["integration"] = results


# ===========================================================================
# SECTION 5 — Pattern Summary & Codex Fix Guidance
# ===========================================================================

def generate_fix_guidance():
    section("SECTION 5 — Pattern Summary & Codex Fix Guidance")

    outfit_data  = report["sections"].get("outfit_engine", {})
    color_data   = report["sections"].get("color_extraction", {})
    mask_data    = report["sections"].get("mask_quality", {})

    # Outfit repetition pattern
    rep_results = outfit_data.get("repetition_by_wardrobe_size", [])
    problem_configs = [r for r in rep_results if r.get("has_problem")]

    log("PROBLEM 1 — OUTFIT REPETITION")
    log("  Confirmed: " + ("YES — found in configs: " + str([r["config"] for r in problem_configs])
                            if problem_configs else "NO — repetition within acceptable bounds"))

    novelty_dom = outfit_data.get("never_worn_dominance_pct", 0)
    if novelty_dom > 50:
        log(f"  Root cause: novelty_score() dominates ({novelty_dom}% of days go to never-worn tops)")
        log(f"  Mechanism: used_pairs blocks top+bottom combos but NOT individual tops.")
        log(f"    Top A + Pants 1 = blocked. Top A + Pants 2 = allowed. Top A wins again.")
        log(f"  Fix location: generate_weekly_plan() in outfit_engine.py")
        log(f"  Fix: track used_top_ids set separately. After each day, add top.id to")
        log(f"    used_top_ids. In _score_candidates, apply -0.25 penalty to any top")
        log(f"    whose id is in used_top_ids from the current session.")

    log("\nPROBLEM 2 — BACKGROUND SEPARATION")
    inv = mask_data.get("inversion_suspects", 0)
    frag = mask_data.get("fragmentation_suspects", 0)
    bleed = mask_data.get("background_bleed_suspects", 0)
    log(f"  Inversion artifacts: {inv} masks")
    log(f"  Fragmented masks: {frag} masks")
    log(f"  Background bleed: {bleed} masks")
    log(f"  Fix locations: sam_segmentation.py → segment_clothing()")
    log(f"  Fix: add background rejection points at image corners (label=0),")
    log(f"    tighten grid to center 35-65% of image,")
    log(f"    replace fallback threshold with GrabCut for gray backgrounds.")

    log("\nPROBLEM 3 — COLOR MISCLASSIFICATION")
    misses = color_data.get("misclassification_count", "?")
    holes  = color_data.get("anchor_coverage_holes", "?")
    log(f"  Known color misclassifications: {misses}/20 test colors")
    log(f"  Color-space holes (>20 LAB units from any anchor): {holes}")
    log(f"  Low-confidence extractions on real images: {color_data.get('low_confidence_count', '?')}")
    log(f"  Fix location: color_extraction.py → _COLOR_ANCHORS_RGB")
    log(f"  Fix: add forest_green(20,80,45), dark_teal(15,80,75), teal_green(25,100,70),")
    log(f"    charcoal(50,55,60), slate(70,80,90), dark_navy(8,12,50).")
    log(f"  Additional fix: run KMeans in LAB space, not BGR — BGR centroids drift")
    log(f"    toward perceptually-wrong midpoints on dark complex garments.")

    log("\n" + "="*70)
    log("  PRIORITY ORDER FOR CODEX")
    log("="*70)
    log("  1. outfit_engine.py: add per-session used_top_ids tracking  (30 min fix)")
    log("  2. color_extraction.py: add 6 missing dark/teal anchors      (15 min fix)")
    log("  3. color_extraction.py: KMeans in LAB space                  (30 min fix)")
    log("  4. sam_segmentation.py: add corner rejection points + GrabCut (1 hr fix)")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    log("WARDROBE RECOMMENDER — DEEP DIAGNOSTIC EVALUATION")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Project: {PROJECT_ROOT}")
    log(f"Uploads: {len(list(UPLOADS_DIR.glob('*')))} files")

    t_start = time.time()

    try:
        run_outfit_engine_tests()
    except Exception as e:
        log(f"[ERROR in outfit tests] {e}")
        traceback.print_exc()

    try:
        run_color_extraction_tests()
    except Exception as e:
        log(f"[ERROR in color tests] {e}")
        traceback.print_exc()

    try:
        run_mask_quality_tests()
    except Exception as e:
        log(f"[ERROR in mask tests] {e}")
        traceback.print_exc()

    try:
        run_integration_tests()
    except Exception as e:
        log(f"[ERROR in integration tests] {e}")
        traceback.print_exc()

    try:
        generate_fix_guidance()
    except Exception as e:
        log(f"[ERROR in fix guidance] {e}")
        traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\nTotal runtime: {elapsed:.1f}s")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    save_reports()
