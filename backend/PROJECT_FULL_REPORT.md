# OutfitAI - Final Project Report (Presentation Freeze)

Last updated: 2026-03-19
Freeze intent: this version is written for presentation use.

## 1) What this project does

OutfitAI is a wardrobe + outfit recommendation system.

It does 4 practical things:

1. User uploads one clothing image (shirt or pant) from scan page.
2. System detects and segments the garment, then extracts color.
3. Garment is saved in wardrobe with metadata.
4. Recommender generates top-bottom outfits and adapts with feedback.

---

## 2) Current architecture (simple view)

Frontend:

- `frontend/scan.html`
- `frontend/wardrobe.html`
- `frontend/outfits.html`
- `frontend/profile.html`
- `frontend/calendar.html`
- `frontend/palette.html`
- `frontend/history.html`

Backend:

- FastAPI + SQLAlchemy + SQLite (MVP)
- Main API modules:
  - `app/api/scan.py`
  - `app/api/wardrobe.py`
  - `app/api/outfits.py`
  - `app/api/feedback.py`
  - `app/api/body_profile.py`

Core pipeline:

`Upload -> YOLO (upper/lower) -> SAM2 segmentation -> LAB color extraction -> save wardrobe item -> recommender`

---

## 3) What is implemented now (important for demo)

### Vision and scan flow

1. Single-garment scan mode is active (no forced top+bottom duplication).
2. YOLO + SAM2 pipeline is integrated and working.
3. Scan UI now shows useful details (type/color/family/palette), not raw mask file paths.
4. Scan auto-assigns `item_type` from suggestions when confidence is good and category matches.

### Recommender flow

1. Recommender uses hybrid logic:
   - rule-based scoring
   - collaborative signal (MF/SVD)
   - cosine similarity controls for diversity
2. Repeat control and rotation penalties are active.
3. Feedback updates ranking behavior.
4. Outfit generation endpoint no longer hard-blocks on `item_type="unknown"` (active top+bottom is enough).

### UI/product features

1. Persistent navbar + active page highlight.
2. Dark/light toggle with localStorage persistence.
3. Toast notifications.
4. Keyboard shortcuts (`S`, `W`, `O`, `?`).
5. Outfit of the Day widget + mark as worn.
6. Wardrobe masonry layout + filters + stats sidebar.
7. Calendar page, palette page, and history page.
8. Profile page includes:
   - body-shape visual icons
   - skin-tone color swatches
   - skip-friendly flow
   - save redirects to homepage.

---

## 4) API endpoints currently used in UI

Scan:

- `POST /scan/start`
- `POST /scan/upload/{scan_id}`
- `GET /scan/history`

Wardrobe:

- `GET /wardrobe`
- `PUT /wardrobe/{item_id}`
- `DELETE /wardrobe/{item_id}`

Outfits:

- `POST /outfits/generate`
- `GET /outfits/{outfit_date}`
- `GET /outfits/history?month=YYYY-MM`
- `POST /outfits/mark-worn`

Feedback/Profile:

- `POST /feedback`
- `GET /body-profile`
- `PUT /body-profile`

---

## 5) Evaluation status (latest measured numbers)

Important: these are from the latest completed evaluation artifacts, not from a rerun after the most recent UI bug fixes.

Source files:

- `EVALUATION & RESULTS - VAIBHAVI/DETAIL_FILES/vision_summary_latest.json`
- `EVALUATION & RESULTS - VAIBHAVI/DETAIL_FILES/recommender_summary_latest.json`

### Vision summary

Dataset size: `34`

- Detection proxy accuracy: `0.7647`
- Segmentation success: `1.0000`
- Color success: `0.5161`
- End-to-end pipeline success: `0.4118`

Interpretation:

1. Segmentation is strong.
2. Detection and color are still the weak points.

### Recommender summary (baseline vs personalized)

- Avg score lift: `+0.1284`
- Avg diversity lift: `+0.0103`
- Avg repetition-rate lift: `-0.0042` (lower repetition is better)
- Avg coverage lift: `+0.0208`
- Avg forgotten-item-rate lift: `-0.1738` (lower is better)

Interpretation:

1. Recommender quality is better than baseline.
2. It repeats less and uses more wardrobe variety.

---

## 6) Known limits (honest section)

1. Vision metrics still show color instability on difficult images.
2. Detection on weak-label internet images is improved but not perfect.
3. SQLite is fine for demo, not ideal for real multi-user production load.
4. After major logic changes, metrics should be rerun for final publication numbers.

---

## 7) Presentation script (short, clear)

Use this structure:

1. Problem: users want outfit suggestions that are useful and not repetitive.
2. Pipeline: upload -> detect -> segment -> color -> store -> recommend.
3. Recommender claim: personalized mode improves score, diversity, and coverage while reducing repeats.
4. Honest gap: color and some detection cases still need work.
5. Result: complete working system with measurable progress and clear next steps.

---

## 8) Freeze checklist

Before PPT submission, keep these exact files together:

1. `PROJECT_FULL_REPORT.md` (this file)
2. `EVALUATION & RESULTS - VAIBHAVI/MASTER_EVALUATION_SUMMARY_VAIBHAVI.md`
3. `EVALUATION & RESULTS - VAIBHAVI/DETAIL_FILES/evaluation_report_full_latest.md`
4. `EVALUATION & RESULTS - VAIBHAVI/MAIN_VISUALS/*`

Recommended final step after freeze:

1. Run one final internal evaluation and replace only metric blocks if numbers change.
