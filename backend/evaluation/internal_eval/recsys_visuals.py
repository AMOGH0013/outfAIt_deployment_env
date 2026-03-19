from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _metric_bar(summary: dict[str, Any], out_dir: Path) -> None:
    users = summary.get("users", [])
    if not users:
        return

    metrics = [
        ("mean_score", "Mean Score", False),
        ("diversity", "Diversity", False),
        ("repetition_rate", "Repetition Rate", True),
        ("wear_through", "Wear Through", False),
    ]
    baseline = []
    personalized = []
    names = []
    for key, label, _lower_is_better in metrics:
        b = np.mean([float(u["baseline"][key]) for u in users])
        p = np.mean([float(u["personalized"][key]) for u in users])
        names.append(label)
        baseline.append(b)
        personalized.append(p)

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bars1 = ax.bar(x - width / 2.0, baseline, width, label="Baseline", color="#7f8aa3")
    bars2 = ax.bar(x + width / 2.0, personalized, width, label="Personalized", color="#2f9e44")
    ax.set_title("Recommender Metrics: Baseline vs Personalized")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    _save(fig, out_dir / "recsys_baseline_vs_personalized.png")


def _lift_bar(summary: dict[str, Any], out_dir: Path) -> None:
    agg = summary.get("aggregate", {})
    metrics = [
        ("avg_score_lift", "Score lift"),
        ("avg_diversity_lift", "Diversity lift"),
        ("avg_repetition_rate_lift", "Repetition lift"),
        ("avg_coverage_lift", "Coverage lift"),
        ("avg_forgotten_item_rate_lift", "Forgotten-item lift"),
    ]
    labels = [label for _key, label in metrics]
    values = [float(agg.get(key, 0.0)) for key, _label in metrics]
    colors = ["#2f9e44" if v >= 0 else "#d9480f" for v in values]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#555", linewidth=1)
    ax.set_title("Average Lift vs Baseline (8 Synthetic Users)")
    ax.set_ylabel("Lift")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bar, v in zip(bars, values):
        y = v + (0.004 if v >= 0 else -0.009)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{v:+.4f}", ha="center", va="bottom", fontsize=9)

    _save(fig, out_dir / "recsys_avg_lifts.png")


def _user_lift_heatmap(summary: dict[str, Any], out_dir: Path) -> None:
    users = summary.get("users", [])
    if not users:
        return

    cols = [
        ("score_lift", "Score"),
        ("diversity_lift", "Diversity"),
        ("repetition_rate_lift", "Repetition"),
        ("coverage_lift", "Coverage"),
        ("forgotten_item_rate_lift", "Forgotten"),
    ]
    row_labels = [str(u.get("user_id", "u")) for u in users]
    matrix = np.array(
        [[float(u["lift"].get(key, 0.0)) for key, _label in cols] for u in users],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 1e-3)
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title("Per-User Lift Heatmap (Personalized - Baseline)")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _key, label in cols], fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=7, color="#111")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lift")
    _save(fig, out_dir / "recsys_user_lift_heatmap.png")


def _coverage_repeat_scatter(summary: dict[str, Any], out_dir: Path) -> None:
    users = summary.get("users", [])
    if not users:
        return

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for u in users:
        uid = str(u.get("user_id", "u"))
        bx = float(u["baseline_extra"]["wardrobe_coverage"])
        by = float(u["baseline_extra"]["exact_outfit_repeat_rate"])
        px = float(u["personalized_extra"]["wardrobe_coverage"])
        py = float(u["personalized_extra"]["exact_outfit_repeat_rate"])
        ax.plot([bx, px], [by, py], color="#7f8aa3", alpha=0.45, linewidth=1.2)
        ax.scatter([bx], [by], color="#d9480f", s=36)
        ax.scatter([px], [py], color="#2f9e44", s=36)
        ax.text(px + 0.002, py + 0.002, uid.replace("synthetic_", ""), fontsize=7, alpha=0.75)

    ax.set_xlabel("Wardrobe coverage (higher is better)")
    ax.set_ylabel("Exact outfit repeat rate (lower is better)")
    ax.set_title("Each line = one user (red: baseline -> green: personalized)")
    ax.grid(True, linestyle="--", alpha=0.25)
    _save(fig, out_dir / "recsys_coverage_vs_repeat.png")


def _trend_plot(summary: dict[str, Any], out_dir: Path) -> None:
    users = summary.get("users", [])
    if not users:
        return

    # Align on available horizon
    n = min(len(u.get("baseline_series", [])) for u in users)
    if n <= 0:
        return

    def avg_series(mode_key: str, metric: str) -> np.ndarray:
        arrs = []
        for u in users:
            series = u.get(mode_key, [])[:n]
            arrs.append(np.array([float(day.get(metric, 0.0)) for day in series], dtype=np.float32))
        return np.mean(np.stack(arrs, axis=0), axis=0)

    days = np.arange(1, n + 1)
    b_rep = avg_series("baseline_series", "ma_repetition_rate")
    p_rep = avg_series("personalized_series", "ma_repetition_rate")
    b_div = avg_series("baseline_series", "ma_diversity")
    p_div = avg_series("personalized_series", "ma_diversity")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    ax1, ax2 = axes
    ax1.plot(days, b_rep, label="Baseline", color="#d9480f", linewidth=2)
    ax1.plot(days, p_rep, label="Personalized", color="#2f9e44", linewidth=2)
    ax1.set_title("Rolling Repetition Rate Over Time")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Repetition rate")
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.legend()

    ax2.plot(days, b_div, label="Baseline", color="#d9480f", linewidth=2)
    ax2.plot(days, p_div, label="Personalized", color="#2f9e44", linewidth=2)
    ax2.set_title("Rolling Diversity Over Time")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Diversity")
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.legend()

    _save(fig, out_dir / "recsys_trends_over_time.png")


def _readme(summary: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Recommender Visual Pack",
        "",
        "Use these charts in PPT to show recommender behavior clearly.",
        "",
        "## Chart list",
        "",
        "1. `recsys_baseline_vs_personalized.png`",
        "2. `recsys_avg_lifts.png`",
        "3. `recsys_user_lift_heatmap.png`",
        "4. `recsys_coverage_vs_repeat.png`",
        "5. `recsys_trends_over_time.png`",
        "",
        "## One-line talk tracks",
        "",
        "- Baseline vs personalized: personalized improves diversity and reduces repetition.",
        "- Average lifts: diversity and coverage go up, repeats go down.",
        "- User heatmap: improvements are consistent across most users, not just one case.",
        "- Coverage vs repeat: users move toward better coverage and lower repeats.",
        "- Trends over time: personalized remains less repetitive while preserving diversity.",
        "",
        "## Context",
        "",
        f"- Simulation horizon: {summary.get('days')} days",
        f"- Replicates: {summary.get('replicates')}",
        f"- Users: {len(summary.get('users', []))}",
    ]
    (out_dir / "RECSYS_VISUALS_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate recommender-only presentation visuals.")
    parser.add_argument(
        "--summary",
        type=str,
        default=r"backend/evaluation_outputs/internal_eval_20260318_220859/recommender/recommender_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"backend/evaluation_outputs/internal_eval_20260318_220859/charts",
    )
    args = parser.parse_args()

    summary = _load_summary(Path(args.summary))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _metric_bar(summary, out_dir)
    _lift_bar(summary, out_dir)
    _user_lift_heatmap(summary, out_dir)
    _coverage_repeat_scatter(summary, out_dir)
    _trend_plot(summary, out_dir)
    _readme(summary, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
