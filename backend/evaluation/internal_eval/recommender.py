from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from evaluation.simulate_users import (
    _generate_wardrobe_seed,
    _load_embedding_library,
    _mean,
    _std,
    run_user_mode,
)


def _shannon_entropy(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    acc = 0.0
    for count in counts.values():
        p = float(count) / total
        if p <= 1e-12:
            continue
        acc -= p * math.log(p, 2)
    return acc


def _max_streak(values: list[str]) -> int:
    if not values:
        return 0
    best = 1
    cur = 1
    for prev, current in zip(values, values[1:]):
        if prev == current:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _extra_metrics(day_rows: list[dict[str, Any]], total_items: int) -> dict[str, float]:
    item_ids = [str(item_id) for row in day_rows for item_id in (row.get("item_ids") or [])]
    outfit_ids = ["|".join(str(item_id) for item_id in (row.get("item_ids") or [])) for row in day_rows]
    top_ids = [str((row.get("item_ids") or [None])[0]) for row in day_rows if row.get("item_ids")]
    bottom_ids = [str((row.get("item_ids") or [None, None])[1]) for row in day_rows if len(row.get("item_ids") or []) > 1]
    item_counts = Counter(item_ids)
    coverage = len(item_counts) / float(max(total_items, 1))
    unused_rate = 1.0 - coverage
    exact_repeat_rate = 1.0 - (len(set(outfit_ids)) / float(max(len(outfit_ids), 1)))
    exposure_entropy = _shannon_entropy(item_counts)
    normalized_entropy = exposure_entropy / math.log(max(len(item_counts), 2), 2) if item_counts else 0.0
    forgotten_threshold = max(1, len(day_rows) - 21)
    late_ids = {str(item_id) for row in day_rows[forgotten_threshold:] for item_id in (row.get("item_ids") or [])}
    forgotten_rate = 1.0 - (len(late_ids) / float(max(len(item_counts), 1)))

    return {
        "wardrobe_coverage": round(coverage, 4),
        "unused_item_rate": round(unused_rate, 4),
        "exact_outfit_repeat_rate": round(exact_repeat_rate, 4),
        "item_exposure_entropy": round(normalized_entropy, 4),
        "forgotten_item_rate_21d": round(forgotten_rate, 4),
        "max_top_repeat_streak": float(_max_streak(top_ids)),
        "max_bottom_repeat_streak": float(_max_streak(bottom_ids)),
    }


def run_recommender_evaluation(
    out_dir: Path,
    *,
    days: int = 60,
    start_date: str = "2026-01-01",
    seed: int = 1337,
    noise: float = 0.12,
    window: int = 7,
    replicates: int = 3,
    embedding_source_db: Path | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    eval_dir = out_dir / "recommender"
    eval_dir.mkdir(parents=True, exist_ok=True)

    users_path = root / "evaluation" / "synthetic_users.json"
    users = json.loads(users_path.read_text(encoding="utf-8"))
    emb_db = embedding_source_db or (root / "wardrobe.db")
    emb_groups = _load_embedding_library(emb_db)
    start_d = date.fromisoformat(start_date)

    users_out: list[dict[str, Any]] = []
    score_lifts: list[float] = []
    diversity_lifts: list[float] = []
    repetition_lifts: list[float] = []
    coverage_lifts: list[float] = []
    forgotten_lifts: list[float] = []

    for user_def in users:
        wardrobe_seed = _generate_wardrobe_seed(
            user_id=user_def["user_id"],
            style_bias=user_def["style_bias"],
            emb_groups=emb_groups,
            n_tops=16,
            n_bottoms=8,
        )
        total_items = len(wardrobe_seed)

        base_runs = []
        pers_runs = []
        for idx in range(replicates):
            rep_seed = seed + idx * 101
            base_runs.append(
                run_user_mode(
                    user_def=user_def,
                    wardrobe_seed=wardrobe_seed,
                    mode="baseline",
                    start_date=start_d,
                    days=days,
                    seed=rep_seed,
                    noise_rate=noise,
                    window=window,
                )
            )
            pers_runs.append(
                run_user_mode(
                    user_def=user_def,
                    wardrobe_seed=wardrobe_seed,
                    mode="personalized",
                    start_date=start_d,
                    days=days,
                    seed=rep_seed,
                    noise_rate=noise,
                    window=window,
                )
            )

        base0 = base_runs[0]
        pers0 = pers_runs[0]
        base_extra = _extra_metrics(base0.day_rows, total_items=total_items)
        pers_extra = _extra_metrics(pers0.day_rows, total_items=total_items)

        lift = {
            "score_lift": round(float(pers0.metrics["mean_score"]) - float(base0.metrics["mean_score"]), 4),
            "diversity_lift": round(float(pers0.metrics["diversity"]) - float(base0.metrics["diversity"]), 4),
            "repetition_rate_lift": round(float(pers0.metrics["repetition_rate"]) - float(base0.metrics["repetition_rate"]), 4),
            "wear_through_lift": round(float(pers0.metrics["wear_through"]) - float(base0.metrics["wear_through"]), 4),
            "coverage_lift": round(float(pers_extra["wardrobe_coverage"]) - float(base_extra["wardrobe_coverage"]), 4),
            "forgotten_item_rate_lift": round(float(pers_extra["forgotten_item_rate_21d"]) - float(base_extra["forgotten_item_rate_21d"]), 4),
            "exact_outfit_repeat_lift": round(float(pers_extra["exact_outfit_repeat_rate"]) - float(base_extra["exact_outfit_repeat_rate"]), 4),
        }

        score_lifts.append(float(lift["score_lift"]))
        diversity_lifts.append(float(lift["diversity_lift"]))
        repetition_lifts.append(float(lift["repetition_rate_lift"]))
        coverage_lifts.append(float(lift["coverage_lift"]))
        forgotten_lifts.append(float(lift["forgotten_item_rate_lift"]))

        users_out.append(
            {
                "user_id": user_def["user_id"],
                "baseline": base0.metrics,
                "personalized": pers0.metrics,
                "baseline_extra": base_extra,
                "personalized_extra": pers_extra,
                "lift": lift,
                "baseline_series": base0.series,
                "personalized_series": pers0.series,
                "baseline_stability": base0.stability,
                "personalized_stability": pers0.stability,
                "replicate_score_lift_mean": round(_mean([float(p.metrics["mean_score"]) - float(b.metrics["mean_score"]) for b, p in zip(base_runs, pers_runs)]), 4),
                "replicate_score_lift_std": round(_std([float(p.metrics["mean_score"]) - float(b.metrics["mean_score"]) for b, p in zip(base_runs, pers_runs)]), 4),
            }
        )

    summary = {
        "days": days,
        "start_date": start_date,
        "seed": seed,
        "noise": noise,
        "window": window,
        "replicates": replicates,
        "users": users_out,
        "aggregate": {
            "avg_score_lift": round(_mean(score_lifts), 4),
            "avg_diversity_lift": round(_mean(diversity_lifts), 4),
            "avg_repetition_rate_lift": round(_mean(repetition_lifts), 4),
            "avg_coverage_lift": round(_mean(coverage_lifts), 4),
            "avg_forgotten_item_rate_lift": round(_mean(forgotten_lifts), 4),
        },
    }

    (eval_dir / "recommender_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "user_id,score_lift,diversity_lift,repetition_rate_lift,coverage_lift,forgotten_item_rate_lift,exact_outfit_repeat_lift",
    ]
    for user in users_out:
        lift = user["lift"]
        lines.append(
            ",".join(
                [
                    user["user_id"],
                    str(lift["score_lift"]),
                    str(lift["diversity_lift"]),
                    str(lift["repetition_rate_lift"]),
                    str(lift["coverage_lift"]),
                    str(lift["forgotten_item_rate_lift"]),
                    str(lift["exact_outfit_repeat_lift"]),
                ]
            )
        )
    (eval_dir / "recommender_lifts.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
