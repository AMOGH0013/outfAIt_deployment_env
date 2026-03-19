# Vaibhavi Presentation Summary

Owner: Vaibhavi (Evaluation + testing + evidence)
Target duration: 3.5 minutes

## 1) What to present

1. How we tested the system.
2. Which numbers improved.
3. What is still weak.
4. Why results are credible for a recommender systems class.

---

## 2) Evaluation method (simple)

We used two evaluation layers:

1. Vision checks:
   - detection,
   - segmentation,
   - color,
   - end-to-end success.

2. Recommender checks:
   - baseline vs personalized comparison,
   - 8 synthetic users,
   - 60-day simulation,
   - 3 replicates.

---

## 3) Numbers to say clearly

From latest saved evaluation artifacts:

Recommender:

1. Avg score lift: `+0.1284`
2. Avg diversity lift: `+0.0103`
3. Avg repetition-rate lift: `-0.0042` (good, lower repetition)
4. Avg coverage lift: `+0.0208`
5. Avg forgotten-item-rate lift: `-0.1738` (good)

Vision:

1. Detection proxy accuracy: `0.7647`
2. Segmentation success: `1.0000`
3. Color success: `0.5161`
4. End-to-end success: `0.4118`

---

## 4) Honest interpretation

1. Recommender part is clearly better than baseline.
2. Segmentation is strong.
3. Color is still the biggest vision weakness.
4. Detection is better than before but not perfect.

This is important to say:

- We improved a lot, but we are not claiming perfect performance.

---

## 5) Visuals to use

Use charts in `MAIN_VISUALS/`:

1. `recsys_baseline_vs_personalized.png`
2. `recsys_avg_lifts.png`
3. `recsys_user_lift_heatmap.png`
4. `recsys_coverage_vs_repeat.png`
5. `recsys_trends_over_time.png`

How to explain each:

1. Personalized mode improves behavior.
2. Repetition goes down.
3. Coverage and diversity go up.

---

## 6) Smart moves and struggles to mention

1. We found weak points through metrics, not guessing.
2. We built failure categories and worst-image lists for debugging.
3. We separated recommender evaluation from vision evaluation so we could see real bottlenecks.
4. We used repeated simulations to reduce one-run noise.

---

## 7) 45-second close line

"Our evidence shows the recommender is working better than baseline. We also openly show where vision still needs work, especially color. This is a real engineering learning process, with measurable progress and clear next steps."
