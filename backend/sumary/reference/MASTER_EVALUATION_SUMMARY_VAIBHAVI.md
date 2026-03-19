# VAIBHAVI - MASTER EVALUATION SUMMARY (Freeze Version)

Last updated: 2026-03-19
Audience: non-technical and technical presentation members.

## 1) One-line project status

The recommender is working better than baseline, but color and some detection cases still need improvement.

---

## 2) Exact numbers to present (from latest evaluation files)

Source files:

1. `DETAIL_FILES/recommender_summary_latest.json`
2. `DETAIL_FILES/vision_summary_latest.json`

### Recommender results (Personalized vs Baseline)

1. Avg score lift: `+0.1284`
2. Avg diversity lift: `+0.0103`
3. Avg repetition-rate lift: `-0.0042` (good, because lower repetition is better)
4. Avg wardrobe coverage lift: `+0.0208`
5. Avg forgotten-item-rate lift: `-0.1738` (good)

Simple meaning:

1. Recommendations improved.
2. System repeats less.
3. System uses wardrobe items more fairly.

### Vision results

1. Dataset size: `34`
2. Detection proxy accuracy: `0.7647`
3. Segmentation success rate: `1.0000`
4. Color success rate: `0.5161`
5. End-to-end success rate: `0.4118`

Simple meaning:

1. Segmentation is very strong.
2. Detection and color are the current weak spots.

---

## 3) What changed recently in product/code (important for viva)

1. Scan now handles single-garment upload mode properly.
2. Scan result UI no longer shows raw mask filepath text to users.
3. Profile page now shows:
   - body-shape visual icons
   - skin-tone color swatches
4. Outfit generation was fixed so it is not blocked only because `item_type` is unknown.
5. Added pages: calendar, palette, history.

Note:

- These are real code changes.
- Evaluation numbers above are from the latest saved evaluation artifacts; rerun if you want numbers that include every very latest patch.

---

## 4) Visuals to use in PPT

Use these files from `MAIN_VISUALS/`:

1. `recsys_baseline_vs_personalized.png`
2. `recsys_avg_lifts.png`
3. `recsys_user_lift_heatmap.png`
4. `recsys_coverage_vs_repeat.png`
5. `recsys_trends_over_time.png`

How to explain simply:

1. Personalized model gives better recommendation behavior.
2. Repeats go down.
3. Coverage and diversity go up.

---

## 5) Simple explanation for class topics

### Is cosine similarity used?

Yes.

- It is used in embedding similarity/diversity logic.

### Is collaborative filtering used?

Yes.

- Matrix factorization (SVD style) is used as part of the hybrid recommender.

### Is Pearson correlation used?

No.

- Pearson is not in the current pipeline.

---

## 6) What still needs work (say this honestly)

1. Color success is still low relative to target.
2. Detection on hard real images can still fail.
3. End-to-end reliability depends on those two parts.

---

## 7) 45-second speaking script

1. We tested with synthetic users for 60 days and compared baseline vs personalized recommendations.
2. Personalized mode improved score, diversity, and wardrobe coverage, and reduced repetition.
3. So the recommender part is clearly stronger than baseline.
4. On vision side, segmentation is strong, but detection and color still need more work.
5. This gives a clear roadmap: keep recommender gains, improve vision reliability next.

---

## 8) Files that matter most for freeze

1. `MASTER_EVALUATION_SUMMARY_VAIBHAVI.md`
2. `README_START_HERE.md`
3. `DETAIL_FILES/evaluation_report_full_latest.md`
4. `DETAIL_FILES/recommender_summary_latest.json`
5. `DETAIL_FILES/vision_summary_latest.json`
6. `MAIN_VISUALS/*`
