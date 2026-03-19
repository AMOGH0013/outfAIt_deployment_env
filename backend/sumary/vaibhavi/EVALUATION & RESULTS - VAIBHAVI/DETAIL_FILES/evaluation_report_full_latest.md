# Detailed Evaluation Report (Latest Saved Artifacts)

Last updated: 2026-03-19
Purpose: detailed reference for viva and technical Q&A.

## 1) Evaluation scope

This report combines:

1. Vision pipeline quality checks
2. Recommender baseline vs personalized simulation

Data source files used:

1. `recommender_summary_latest.json`
2. `vision_summary_latest.json`

---

## 2) Vision metrics (from `vision_summary_latest.json`)

Dataset:

1. Images evaluated: `34`
2. Label mix:
   - `upper_only`: `19`
   - `lower_only`: `14`
   - `full_outfit`: `1`

Main metrics:

1. Detection proxy accuracy: `0.7647`
2. Segmentation success rate: `1.0000`
3. Color success rate: `0.5161`
4. End-to-end pipeline success rate: `0.4118`

Extra diagnostics:

1. Mean mask quality: `0.9871`
2. Mean color stability: `63.2`
3. Mean LAB drift: `4.0433`
4. Mean LAB vs HSV improvement: `-7.83%`

Failure counts:

1. `missed_detection`: `4`
2. `wrong_class`: `4`
3. `color_misclassification`: `14`

Simple interpretation:

1. Segmentation is stable.
2. Detection is decent but not perfect.
3. Color is still the largest vision issue.

---

## 3) Recommender metrics (from `recommender_summary_latest.json`)

Simulation config:

1. Users: `8` synthetic users
2. Horizon: `60 days`
3. Replicates: `3`
4. Start date: `2026-01-01`

Aggregate lifts (personalized minus baseline):

1. Avg score lift: `+0.1284`
2. Avg diversity lift: `+0.0103`
3. Avg repetition-rate lift: `-0.0042` (good)
4. Avg coverage lift: `+0.0208`
5. Avg forgotten-item-rate lift: `-0.1738` (good)

Simple interpretation:

1. Personalized recommender is clearly better than baseline.
2. It is less repetitive and uses wardrobe items more fairly.

---

## 4) Product and code changes already integrated after earlier reports

These changes are implemented in code and aligned with presentation freeze:

1. Single-garment scan behavior (no duplicate top-bottom from one shirt).
2. Profile save returns to home.
3. Profile page now has body-shape icons and skin-tone color swatches.
4. Scan result page now shows useful user information, not raw mask filepath text.
5. Outfit generation no longer hard-fails only because `item_type` is unknown.
6. Added calendar, palette, and scan history pages.

Important note:

- Metric values above are from latest saved evaluation artifacts.
- If you need metrics that include every latest patch, run a fresh internal evaluation and replace only numeric sections.

---

## 5) Honest current risks

1. Color extraction accuracy is still below target.
2. Detection on difficult real images still has misses.
3. End-to-end success depends on those two stages.

---

## 6) Files for evidence during Q&A

1. `recommender_summary_latest.json`
2. `vision_summary_latest.json`
3. `COLOR_DEEP_PASS_COMPARISON_20260319.md`
4. `../MAIN_VISUALS/*`
