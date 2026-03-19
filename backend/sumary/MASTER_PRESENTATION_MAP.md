# OutfitAI Presentation Master Map

Last updated: 2026-03-19
Purpose: one file that tells everyone what to present, in what order, and what proof to show.

## 1) Ownership (recovered and reassigned cleanly)

Source used for ownership recovery:

- `backend/recovery.md`

Final presenter ownership:

1. Amogh: technical architecture, pipeline internals, major fixes, engineering decisions.
2. Dhanuja: recommender logic story, user flow, explainability, demo flow.
3. Vaibhavi: evaluation/testing, metrics, visuals, conclusion quality.

---

## 2) Recommended flow (12 minutes total)

1. Amogh - 5 minutes
2. Dhanuja - 3.5 minutes
3. Vaibhavi - 3.5 minutes

---

## 3) What to emphasize as team story

Do not present this as a perfect one-shot build.

Present it as an engineering learning journey:

1. Early stage had scan/recommendation reliability issues.
2. We diagnosed failures with logs + evaluation framework.
3. We replaced weak parts step by step.
4. We measured improvements and kept known limitations visible.

Good examples to mention:

1. Shirt being treated as both top and bottom -> fixed with single-garment mode and detection conflict logic.
2. Outfit generation failing due to strict unknown-type gate -> fixed by removing hard eligibility block.
3. Scan UI showing technical mask path -> replaced with user-relevant info.
4. Profile UX too plain -> upgraded with body-shape visuals and skin-tone swatches.

---

## 4) Core pipeline to explain clearly

`Upload image -> YOLO detection -> SAM2 segmentation -> LAB color extraction -> save to wardrobe -> generate outfits -> collect feedback -> improve future ranking`

What each stage gives:

1. YOLO: where the garment is (upper/lower).
2. SAM2: exact garment area (clean mask).
3. LAB color extractor: dominant color + palette.
4. Wardrobe DB: persistent item memory.
5. Recommender: top-bottom matching with diversity + repeat control.
6. Feedback loop: behavior adaptation over time.

---

## 5) Presenter file map

Use these files directly:

1. `amogh/AMOGH_PRESENTATION_SUMARY.md`
2. `dhanuja/DHANUJA_PRESENTATION_SUMARY.md`
3. `vaibhavi/VAIBHAVI_PRESENTATION_SUMARY.md`

---

## 6) Freeze references

Main references copied/indexed in `reference/`:

1. project report
2. vaibhavi master evaluation summary
3. detailed evaluation report
4. visual readme

If a teacher asks for proof, open those files and matching charts.
