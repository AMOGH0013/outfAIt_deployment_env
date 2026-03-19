# Wardrobe Recommendation System (MVP)

FastAPI backend + server-served dev UI for a wardrobe/outfit recommender:

- Scan clothing image -> YOLO (upper/lower) -> SAM2 mask -> color palette (LAB + KMeans) + CLIP embedding
- Wardrobe CRUD + manual semantics correction (category/item_type)
- Outfit generation with explainable scoring (rules + feedback + diversity + fit)
- Offline evaluation harness (`evaluation/`) with synthetic users (A/B baseline vs personalized)

## Quickstart

1) Install deps:

```bash
pip install -r requirements.txt
```

Notes:
- `ultralytics` is used for YOLO inference.
- `sam2` (Segment Anything 2) is installed from GitHub; on Windows you may need WSL/Linux for a smooth install.
- If SAM2 is not available at runtime, `/scan/upload` falls back to a bbox rectangle mask (so the pipeline still runs).

2) Run the API + dev UI:

```bash
uvicorn app.main:app --reload
```

Open:
- Dev UI: `http://127.0.0.1:8000/`
- Swagger: `http://127.0.0.1:8000/docs`

## Vision weights (YOLO + SAM/SAM2)

This repo expects the following weights in the backend root (or set env vars):

- YOLO (upper/lower detector): `best_stage2_FINAL.pt`
  - Override with `YOLO_MODEL_PATH=/absolute/path/to/best_stage2_FINAL.pt`
  - Optional class override: `YOLO_CLASS_MAP="upper=0,lower=1"`

- SAM v1 (fallback): `sam_models/sam_vit_b.pth` (ignored by git)

- SAM2 (ROI segmentation): `sam2_hiera_large.pt` (very large; keep local)
  - Override with `SAM2_CHECKPOINT_PATH=/absolute/path/to/sam2_hiera_large.pt`
  - Optional config override: `SAM2_MODEL_CFG=/absolute/path/to/sam2_hiera_l.yaml`

## Offline evaluation (synthetic)

Run the simulator from the repo root, for example:

```bash
python -m evaluation\simulate_users --days 100 --replicates 5
python -m evaluation\report
```
