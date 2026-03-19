from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_chart(path: Path, vision_summary: dict[str, Any]) -> None:
    color_metric = float(
        vision_summary.get(
            "color_stability_pass_rate",
            vision_summary.get("color_success_rate", 0.0),
        )
    )
    metrics = {
        "Detection": float(vision_summary.get("detection_accuracy_proxy", 0.0)),
        "Segmentation": float(vision_summary.get("segmentation_success_rate", 0.0)),
        "Color Stability": color_metric,
        "End-to-End": float(vision_summary.get("pipeline_success_rate", 0.0)),
    }
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = list(metrics.keys())
    values = [metrics[name] * 100.0 for name in names]
    bars = ax.bar(names, values, color=["#3a7be0", "#2ea86f", "#ef7d22", "#7850b2"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Pipeline Stage Success Rates")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _failure_chart(path: Path, vision_summary: dict[str, Any]) -> None:
    failures = vision_summary.get("failure_counts", {}) or {}
    names = list(failures.keys()) or ["none"]
    values = [int(failures.get(name, 0)) for name in names] or [0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(names, values, color="#c98793")
    ax.set_ylabel("Image count")
    ax.set_title("Failure Category Counts")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.2, str(value), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _recommender_chart(path: Path, recommender_summary: dict[str, Any]) -> None:
    aggregate = recommender_summary.get("aggregate", {}) or {}
    metrics = {
        "Score lift": float(aggregate.get("avg_score_lift", 0.0)),
        "Diversity lift": float(aggregate.get("avg_diversity_lift", 0.0)),
        "Coverage lift": float(aggregate.get("avg_coverage_lift", 0.0)),
        "Repeat lift": float(aggregate.get("avg_repetition_rate_lift", 0.0)),
        "Forgotten lift": float(aggregate.get("avg_forgotten_item_rate_lift", 0.0)),
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#3a7be0" if value >= 0 else "#c98793" for value in metrics.values()]
    bars = ax.bar(metrics.keys(), list(metrics.values()), color=colors)
    ax.axhline(0.0, color="#444", linewidth=1)
    ax.set_ylabel("Lift vs baseline")
    ax.set_title("Synthetic User Evaluation")
    for bar, value in zip(bars, metrics.values()):
        offset = 0.01 if value >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + offset, f"{value:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _executive_findings(vision_summary: dict[str, Any], recommender_summary: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    color_metric = float(
        vision_summary.get(
            "color_stability_pass_rate",
            vision_summary.get("color_success_rate", 0.0),
        )
    )
    if float(vision_summary.get("detection_accuracy_proxy", 0.0)) < 0.90:
        findings.append("Detection remains the main reliability bottleneck on weak-label checks.")
    if color_metric < 0.85:
        findings.append("Color extraction still needs attention on unstable or leakage-heavy masks (stability pass metric).")
    if float(vision_summary.get("mean_hsv_improvement_pct", 0.0)) > 0:
        findings.append("LAB-based color extraction is measurably more stable than the HSV baseline.")
    if float(recommender_summary.get("aggregate", {}).get("avg_repetition_rate_lift", 0.0)) < 0:
        findings.append("Personalization reduces repetition versus the baseline recommender.")
    if float(recommender_summary.get("aggregate", {}).get("avg_coverage_lift", 0.0)) > 0:
        findings.append("Personalization improves wardrobe coverage instead of ignoring the long tail.")
    if not findings:
        findings.append("No major regression dominated the run; the remaining failures are distributed across edge cases.")
    return findings


def generate_evaluation_report(
    out_dir: Path,
    *,
    dataset_summary: dict[str, Any],
    vision_result: dict[str, Any],
    recommender_summary: dict[str, Any],
    fixes_applied: list[str],
) -> Path:
    charts_dir = _ensure_dir(out_dir / "charts")
    report_path = out_dir / "evaluation_report.md"

    vision_summary = vision_result["summary"]
    _stage_chart(charts_dir / "stage_success.png", vision_summary)
    _failure_chart(charts_dir / "failure_counts.png", vision_summary)
    _recommender_chart(charts_dir / "synthetic_user_lifts.png", recommender_summary)

    executive_findings = _executive_findings(vision_summary, recommender_summary)
    worst_images = vision_summary.get("top_20_worst_images", [])[:10]
    weak_dist = dataset_summary.get("weak_label_distribution", {})

    lines = [
        "# Evaluation Report",
        "",
        "## Executive Summary",
        "",
        f"- Dataset scanned: `{dataset_summary.get('dataset_dir')}`",
        f"- Images evaluated: `{dataset_summary.get('image_count')}`",
        f"- Detection proxy accuracy: `{vision_summary.get('detection_accuracy_proxy')}`",
        f"- Segmentation success rate: `{vision_summary.get('segmentation_success_rate')}`",
        f"- Color stability pass rate: `{vision_summary.get('color_stability_pass_rate', vision_summary.get('color_success_rate'))}`",
        f"- End-to-end pipeline success rate: `{vision_summary.get('pipeline_success_rate')}`",
        "",
        "### Key Findings",
        "",
    ]
    lines.extend([f"- {item}" for item in executive_findings])
    lines.extend(
        [
            "",
            "## Fixes Applied In This Build",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in fixes_applied])
    lines.extend(
        [
            "",
            "## Dataset Profile",
            "",
            "| Label bucket | Count |",
            "|---|---:|",
        ]
    )
    for label, count in weak_dist.items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Stage success](charts/stage_success.png)",
            "",
            "![Failure counts](charts/failure_counts.png)",
            "",
            "![Synthetic user lifts](charts/synthetic_user_lifts.png)",
            "",
            "## Vision Evaluation",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mean mask quality score | {vision_summary.get('mean_mask_quality_score')} |",
            f"| Mean color stability score | {vision_summary.get('mean_color_stability_score')} |",
            f"| Color stability pass rate | {vision_summary.get('color_stability_pass_rate', vision_summary.get('color_success_rate'))} |",
            f"| Mean LAB drift | {vision_summary.get('mean_lab_drift')} |",
            f"| Mean LAB improvement over HSV (%) | {vision_summary.get('mean_hsv_improvement_pct')} |",
            "",
            "### Failure Breakdown",
            "",
            "| Failure | Count |",
            "|---|---:|",
        ]
    )
    for label, count in (vision_summary.get("failure_counts", {}) or {}).items():
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "### Worst Images To Review",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in worst_images])

    aggregate = recommender_summary.get("aggregate", {}) or {}
    lines.extend(
        [
            "",
            "## Synthetic User Recommendation Evaluation",
            "",
            f"- Simulation horizon: `{recommender_summary.get('days')} days`",
            f"- Replicates: `{recommender_summary.get('replicates')}`",
            f"- Avg score lift: `{aggregate.get('avg_score_lift')}`",
            f"- Avg diversity lift: `{aggregate.get('avg_diversity_lift')}`",
            f"- Avg repetition-rate lift: `{aggregate.get('avg_repetition_rate_lift')}`",
            f"- Avg coverage lift: `{aggregate.get('avg_coverage_lift')}`",
            f"- Avg forgotten-item-rate lift: `{aggregate.get('avg_forgotten_item_rate_lift')}`",
            "",
            "## Generated Artifacts",
            "",
            f"- Vision JSON: `{(out_dir / 'vision' / 'vision_summary.json').relative_to(out_dir)}`",
            f"- Vision records: `{(out_dir / 'vision' / 'vision_records.json').relative_to(out_dir)}`",
            f"- Failure folders: `{(out_dir / 'vision' / 'failures').relative_to(out_dir)}`",
            f"- Worst images: `{(out_dir / 'vision' / 'top_20_worst').relative_to(out_dir)}`",
            f"- Recommender summary: `{(out_dir / 'recommender' / 'recommender_summary.json').relative_to(out_dir)}`",
            "",
            "## How To Use This Report",
            "",
            "- Use the stage success chart to explain where reliability drops first.",
            "- Use the failure folders to show concrete examples of missed detection, poor segmentation, and color mistakes.",
            "- Use the synthetic-user lifts to justify that the recommender is not just accurate, but also diverse and less repetitive.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report_index.json").write_text(
        json.dumps(
            {
                "dataset_summary": dataset_summary,
                "vision_summary": vision_summary,
                "recommender_summary": recommender_summary,
                "report": str(report_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path
