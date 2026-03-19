from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.internal_eval.dataset import collect_image_cases
from evaluation.internal_eval.recommender import run_recommender_evaluation
from evaluation.internal_eval.reporting import generate_evaluation_report
from evaluation.internal_eval.vision import run_vision_evaluation


FIXES_APPLIED = [
    "Stopped single-garment scans from duplicating one shirt into both top and bottom results.",
    "Switched scan ingestion to single-item mode (one detected garment per upload).",
    "Added deeper color extraction guards: strong mask erosion, center-weighted sampling, LAB histogram mode fallback, and anchor correction rules.",
    "Redirected the profile save flow back to the homepage after a successful save.",
    "Added visible color swatches next to detected color names and palette entries in the scan UI.",
    "Improved the scan page layout so controls, status, and result cards use space more effectively.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent internal evaluation framework for Outfit AI.")
    parser.add_argument("--dataset-dir", type=str, default=r"C:\Users\amogh\Desktop\clothes")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "evaluation_outputs"))
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--start-date", type=str, default="2026-01-01")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--label-manifest",
        type=str,
        default=str(ROOT / "evaluation" / "manifests" / "clothes_manifest.json"),
        help="Optional JSON filename->label map for reliable detection evaluation.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests.")
    parser.add_argument("--skip-recommender", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root).expanduser().resolve() / f"internal_eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest_path = Path(args.label_manifest).expanduser().resolve()
    if not manifest_path.exists():
        manifest_path = None
    cases = collect_image_cases(dataset_dir, label_manifest=manifest_path)
    if int(args.limit) > 0:
        cases = cases[: int(args.limit)]
    dataset_summary = {
        "dataset_dir": str(dataset_dir),
        "label_manifest": (str(manifest_path) if manifest_path is not None else None),
        "image_count": len(cases),
        "weak_label_distribution": {
            key: sum(1 for case in cases if case.weak_label == key)
            for key in sorted({case.weak_label for case in cases})
        },
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")

    vision_result = run_vision_evaluation(cases, out_dir=out_dir)
    recommender_summary = {
        "days": int(args.days),
        "replicates": int(args.replicates),
        "aggregate": {},
        "users": [],
    }
    if not args.skip_recommender:
        recommender_summary = run_recommender_evaluation(
            out_dir=out_dir,
            days=int(args.days),
            start_date=args.start_date,
            seed=int(args.seed),
            replicates=int(args.replicates),
        )
    report_path = generate_evaluation_report(
        out_dir=out_dir,
        dataset_summary=dataset_summary,
        vision_result=vision_result,
        recommender_summary=recommender_summary,
        fixes_applied=FIXES_APPLIED,
    )

    print(f"Evaluation completed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
