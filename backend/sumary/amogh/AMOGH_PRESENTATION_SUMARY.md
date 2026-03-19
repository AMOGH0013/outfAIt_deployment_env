# Amogh Presentation Summary

Owner: Amogh (Technical Lead)
Target duration: 5 minutes

## 1) What to present

1. System architecture (frontend, backend, database, services).
2. Vision pipeline and why each stage exists.
3. Key bugs/failures we faced and exact fixes.
4. Recommender internals at engineering level.
5. Honest limits + next technical steps.

---

## 2) Technical flow to explain

Use this line:

`Upload -> YOLO -> SAM2 -> LAB color -> WardrobeItem -> Outfit engine -> Feedback loop`

Short meaning:

1. YOLO finds garment region.
2. SAM2 gives accurate mask.
3. LAB gives color name + palette.
4. Item saved in DB with metadata.
5. Recommender creates pairings with repeat control.
6. Feedback updates future ranking behavior.

---

## 3) Smart moves and struggles (important)

Mention these as real learning points:

1. Problem: shirt got classified as both top and bottom.
   Fix: moved to single-garment scan logic + overlap conflict resolution.

2. Problem: generation blocked because `item_type` was unknown.
   Fix: removed hard gate in generation and added scan-time auto item-type assignment with confidence threshold.

3. Problem: scan result UI showed raw mask filepath.
   Fix: replaced with user-useful details (detected type, family, palette, actions).

4. Problem: first profile UI was too plain and unclear.
   Fix: body-shape visual icons + skin-tone swatch references.

5. Problem: confidence in recommender quality needed proof.
   Fix: added measurable synthetic-user evaluation + visual reports.

---

## 4) Engineering choices to justify

1. YOLO + SAM2 instead of only one model:
   - YOLO gives semantics (upper/lower),
   - SAM2 gives precise boundaries.

2. LAB-based color extraction:
   - more stable for shade comparison than fixed HSV thresholds.

3. Hybrid recommender:
   - rule-based + collaborative signal + cosine-based diversity.
   - practical and explainable for this project stage.

---

## 5) Likely viva questions and compact answers

Q: Why not a fully deep recommender?
A: We prioritized explainability and controllability with measurable behavior gains.

Q: Is collaborative filtering used?
A: Yes, SVD-style matrix factorization is included in the hybrid stack.

Q: Is cosine similarity used?
A: Yes, in embedding similarity/diversity control.

Q: Is Pearson used?
A: No, not in current pipeline.

---

## 6) Final close line

"We improved the system by identifying real failure points, fixing them one by one, and validating progress with measurable outputs, not assumptions."
