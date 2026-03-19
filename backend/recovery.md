# OutfitAI Full Project Report (End to End)

## Presenter Ownership Index (Section + Line)

- **Amogh (Technical Lead)**: `3`, `4`, `5`, `6`, `7`, `9`, `10`, `13`, `14.1`, `17`, `18` (lines: `35`, `83`, `148`, `223`, `251`, `363`, `387`, `503`, `518`, `640`, `664`)
- **Dhanuja (Rule-Based + Product Flow)**: `2`, `8`, `14.2`, `16` (lines: `20`, `309`, `545`, `625`)
- **Vaibhavi (Evaluation + Testing + Results)**: `11`, `12`, `14.3` (lines: `417`, `491`, `583`)
- **Shared / Team Context**: `1`, `14`, `15`, `19`, `20` (lines: `10`, `514`, `617`, `679`, `710`)

## [Owner: Shared] 1) Project Identity

- **Project**: Wardrobe recommendation system (MVP)
- **Stack**: FastAPI + SQLAlchemy + SQLite + vanilla HTML/CSS/JS
- **Vision/CV stack**: SAM (segmentation), HSV+KMeans (color extraction), CLIP ViT-B/32 (embeddings + type suggestions)
- **Recommendation type**: Hybrid rule-based recommender with bounded personalization signals
- **Personalization style**: Explainable, deterministic scoring (no neural ranker training)

---

## [Owner: Dhanuja] 2) Problem Statement

Most wardrobe apps fail because they:

1. do not isolate clothing from background (bad color/features),
2. do not capture user preference loops (feedback is ignored),
3. produce repetitive combinations (same pair resurfacing),
4. cannot explain recommendations.

This project addresses all 4 with a full loop:

`scan -> segment -> extract features -> store semantics -> generate explainable outfit -> take feedback -> adapt future rankings`

---

## [Owner: Amogh] 3) Evolution Timeline (Start to Current State)

### Stage 0: Core CRUD and Data Plumbing

- Models for `User`, `WardrobeItem`, `Outfit`, `OutfitItem`, `Feedback`, `ScanSession`
- APIs for scan, wardrobe CRUD, outfits, feedback
- Basic frontend pages for scan, wardrobe, outfits

### Stage 1: Vision-Powered Ingestion

- SAM-based mask generation added (`app/services/sam_segmentation.py`)
- Mask-aware color extraction integrated (`app/services/color_extraction.py`)
- CLIP embedding generation integrated (`app/services/embedding_service.py`)
- One uploaded image now creates exactly one wardrobe item (no fake top+bottom duplication)

### Stage 2: Semantics and Safety

- Wardrobe validation for:
  - `category` in `{top, bottom}`
  - `item_type` allowlist
  - category <-> item_type consistency checks
- Outfit eligibility gate:
  - active items only
  - `item_type != "unknown"`
  - must have at least one top and one bottom

### Stage 3: Assisted Intelligence

- CLIP text-image suggestion service for item types (`app/services/item_type_suggester.py`)
- Suggestions stored as metadata (`suggested_item_type`, confidence)
- User remains source of truth (manual confirmation in wardrobe UI)

### Stage 4: Personal Relevance + Stability

- Body profile API and schema extended (`height_cm`, `weight_kg`, body/fit/sex/age/skin_tone)
- Fit preference is used in scoring (bounded contribution)
- Style profile derived from feedback history (`app/services/style_profile.py`)
- Confidence score + challenger labeling added in explanation data
- Repetition control + cooldown + negative feedback amplification added

### Stage 5: Evaluation Harness

- Offline synthetic user simulation added under `evaluation/`
- A/B mode: baseline vs personalized
- 100-day / 5-replicate runs with CI and trend analytics

---

## [Owner: Amogh] 4) Full Arrowed Pipeline (Detailed)

## 4.1 Ingestion Pipeline (Scan to Wardrobe)

`User uploads image`  
`-> POST /scan/start` (creates ScanSession)  
`-> POST /scan/upload/{scan_id}`  
`-> save file in uploads/`  
`-> SAM segmentation (center-point prompt)`  
`-> save mask as /uploads/*_mask.png`  
`-> extract dominant colors using mask`  
`-> compute CLIP embedding from masked image`  
`-> compute item_type suggestions from CLIP text prompts`  
`-> create WardrobeItem (item_type="unknown", category="top" placeholder)`  
`-> store image_url, mask_url, color, palette, embedding, suggestions`  
`-> return debug payload (mask_url, extracted_color, confidence, suggestions)`

## 4.2 Semantics Confirmation Pipeline (Wardrobe Page)

`GET /wardrobe`  
`-> render table of wardrobe items`  
`-> user sets category top/bottom`  
`-> user sets item_type from allowlist`  
`-> PUT /wardrobe/{item_id}`  
`-> backend validates category and item_type consistency`  
`-> item becomes eligible for recommendations`

## 4.3 Outfit Generation Pipeline

`User clicks Generate outfits`  
`-> frontend refreshes wardrobe map`  
`-> POST /outfits/generate?start_date=...&days=...`  
`-> backend loads eligible items (active + known item_type)`  
`-> reject request if no top/bottom eligible`  
`-> build pair feedback biases (time-decayed)`  
`-> fetch recent embeddings`  
`-> fetch recent pairs for repeat blocking`  
`-> derive hard banned ids + embedding neighbor penalties from disliked feedback`  
`-> compute style profile (colors/types + centroids)`  
`-> call outfit_engine.generate_weekly_plan(...)`  
`-> create Outfit + OutfitItems per date (idempotent by date)`  
`-> return outfits with explanation_data`

## 4.4 Feedback Loop Pipeline

`User clicks hero/list action` (`liked`, `disliked`, `worn`, `skipped`)  
`-> POST /feedback`  
`-> backend normalizes aliases`  
`-> if worn: idempotent record + wear_count++ + last_worn_at update`  
`-> if preference: upsert latest preference feedback for outfit`  
`-> future generation uses this feedback in pair_bias + dislike bans/penalties + style profile`

## 4.5 Offline Evaluation Pipeline

`Load synthetic archetypes`  
`-> seed deterministic wardrobes (real sampled embeddings)`  
`-> run BASELINE mode for N days`  
`-> run PERSONALIZED mode for same N days`  
`-> simulate deterministic behavior + noise`  
`-> compute metrics + moving averages + stability`  
`-> aggregate replicates, CI95`  
`-> emit summary.json + report.md`

---

## [Owner: Amogh] 5) Current Architecture and Responsibilities

## 5.1 Backend App Composition

- `app/main.py`
  - app init
  - static mounts (`/uploads`, `/frontend`)
  - root route to `frontend/index.html`
  - startup schema ensure and dev-user bootstrap

- `app/api/scan.py`
  - scan session start/upload
  - pipeline orchestration for SAM + color + embedding + suggestions

- `app/api/wardrobe.py`
  - list, update, soft delete
  - semantic validations

- `app/api/outfits.py`
  - generation + date fetch
  - builds pair biases, bans, recent embeddings, style context
  - stores explanation payload

- `app/api/feedback.py`
  - feedback ingestion
  - worn action updates usage counters

- `app/api/body_profile.py`
  - auto-create profile on GET
  - partial update with strict validation
  - safe handling of empty strings for numeric fields

- `app/api/style_profile.py`
  - exposes derived user style fingerprint

## 5.2 Perception Services

- `app/services/sam_segmentation.py`
  - loads SAM once at import
  - center-point prompt segmentation
  - saves binary mask PNG

- `app/services/color_extraction.py`
  - deterministic HSV + KMeans
  - optional mask support
  - returns primary color + palette + confidences

- `app/services/embedding_service.py`
  - loads CLIP once at import
  - applies mask and computes normalized 512-dim embedding

- `app/services/item_type_suggester.py`
  - CLIP image-text similarity against fixed prompts
  - returns ranked suggestions with confidence

## 5.3 Recommendation Services

- `app/services/outfit_engine.py`
  - candidate generation top x bottom
  - score composition
  - confidence score
  - challenger logic
  - repetition/cooldown control
  - explanation payload creation

- `app/services/style_profile.py`
  - derives preferences and centroids from feedback history
  - no model training, on-the-fly derivation

- `app/services/dev_event_log.py`
  - JSONL per-user event logging for debugging and optimization
  - redacts sensitive keys

---

## [Owner: Amogh] 6) Data Model Snapshot

## 6.1 Key Entities

1. `users`
2. `scan_sessions`
3. `wardrobe_items`
4. `outfits`
5. `outfit_items`
6. `feedback`
7. `body_profiles`

## 6.2 WardrobeItem Fields Used in Recommendations

- `item_type`, `category`, `color`, `color_palette`
- `embedding`
- `is_active`
- `wear_count`, `last_worn_at`
- `suggested_item_type`, `suggested_item_type_confidence`

## 6.3 Outfit Storage

- One outfit per user/date (idempotent behavior in generation route)
- `score` persisted
- `explanation` persisted as JSON string

---

## [Owner: Amogh] 7) Recommendation Logic (Exact, Explainable)

## 7.1 Candidate Construction

- Filter wardrobe:
  - active only
  - non-forbidden
  - non-banned (from negative feedback)
- Split into top and bottom
- Create all pair combinations

## 7.2 Core Rule Score

`rule_score = 0.40 * body_shape_score + 0.30 * color_harmony + 0.15 * novelty(top) + 0.15 * novelty(bottom)`

## 7.3 Added Personalization Terms

- `pair_bias` from feedback (time-decayed, capped)
- `embedding_diversity_score` vs recent worn embeddings
- `fit_score` from body profile fit preference
- `style_affinity_score` from derived style profile
- `cooldown_penalty` if item worn too recently
- `dislike_penalty` from nearest disliked embeddings
- `repeat_penalty` for forced repeat override

## 7.4 Final Score

`final = clamp(rule + pair_bias + embedding + fit + style + cooldown + dislike + repeat, 0, 1)`

## 7.5 Confidence Score

`confidence = sigmoid(w1*rule + w2*fit - w3*diversity_penalty - w4*repetition_risk)`

- Stored in `explanation_data`
- Clamped `[0,1]`

## 7.6 Challenger Tag

- Candidate labeled challenger when:
  - novelty push condition is met (`embedding_score < -0.1` OR exploration quota hit),
  - AND confidence below cutoff (`< 0.55`)

## 7.7 Repetition Control

- Recent pair blocking window: 4 days
- If no alternatives:
  - allow repeat as safety override
  - add repeat penalty
  - set `repetition_blocked_reason="override_no_alternatives"`
- Item cooldown penalty if worn in last 2 days

## 7.8 Negative Feedback Amplification

- Disliked items become hard banned IDs in future candidate set
- Similar neighbors receive penalty by embedding similarity bands

---

## [Owner: Dhanuja] 8) Frontend System (Developer UI)

## 8.1 Pages

1. `index.html`  
   navigation hub

2. `scan.html`  
   scan start/upload + visual debug:
   - original image
   - mask preview
   - extracted color and confidence
   - palette
   - created IDs

3. `wardrobe.html`  
   semantics correction and hygiene:
   - category dropdown
   - item_type dropdown with lock/change behavior
   - suggested type apply button
   - active toggle and soft delete

4. `profile.html`  
   wizard for body profile:
   - identity + age + height + weight
   - body shape visual cards
   - fit preference cards
   - skin tone swatches
   - review + raw JSON

5. `outfits.html`  
   recommendation visualizer:
   - personalization level indicator
   - style summary
   - hero "Outfit of the Day"
   - confidence bar
   - challenger/confident badge
   - powerful feedback buttons
   - detailed list cards with score breakdown and debug JSON

## 8.2 Shared JS and UX Components

- `frontend/app.js`:
  - API helper and error handling
  - explanation summarizer
  - personalization status calculator
  - toast notifications

- `frontend/style.css`:
  - dark editorial dev theme
  - hero styling, badges, progress bars, cards, toasts

---

## [Owner: Amogh] 9) Body Profile Behavior and Validation

## 9.1 API Rules

- `GET /body-profile`
  - auto-creates empty profile if missing

- `PUT /body-profile`
  - partial updates
  - strict enum validation
  - `height_cm` bounds: `50..250`
  - `weight_kg` bounds: `20..400`
  - empty strings are normalized to `None` and ignored (no overwrite)

## 9.2 Current Use in Recommender

- Directly used:
  - `fit_preference` -> `fit_score`

- Stored but not used for scoring yet:
  - `sex`, `age`, `skin_tone`, expanded `body_shape` classes

---

## [Owner: Amogh] 10) API Surface (Current)

## 10.1 Scan

- `POST /scan/start`
- `POST /scan/upload/{scan_id}`

## 10.2 Wardrobe

- `GET /wardrobe?include_inactive=true|false`
- `PUT /wardrobe/{item_id}`
- `DELETE /wardrobe/{item_id}` (soft delete)

## 10.3 Outfits

- `POST /outfits/generate?start_date=YYYY-MM-DD&days=N`
- `GET /outfits/{outfit_date}`

## 10.4 Feedback

- `POST /feedback` with action in `{liked, disliked, worn, skipped}` (+aliases)

## 10.5 Profile and Derived Signals

- `GET /body-profile`
- `PUT /body-profile`
- `GET /style-profile`

---

## [Owner: Vaibhavi] 11) Evaluation and Testing (How Correctness Was Checked)

## 11.1 Offline A/B Evaluation Harness

Location: `evaluation/`

- `synthetic_users.json`: 8 archetypes
- `simulate_users.py`: day-wise simulation engine
- `metrics.py`: metric calculations + trends + stability
- `report.py`: markdown report generation

### Evaluation design

- Same wardrobe and date horizon for both modes
- Baseline mode disables personalization context
- Personalized mode enables:
  - feedback/pair bias
  - fit context
  - style profile
  - embedding diversity effects
- Deterministic seeds + controlled noise
- Replicates for confidence intervals

## 11.2 Metrics Collected

1. `mean_score`
2. `diversity`
3. `repetition_rate`
4. `color_entropy`
5. `wear_through`
6. time series moving averages
7. stability (tail volatility)

## 11.3 Latest Reported Run

From `evaluation/report.md` and `evaluation/summary.json`:

- horizon: 100 days
- replicates: 5
- window: 21 days

### Aggregate lift summary

- Avg score lift: `-0.0049`
- Avg diversity lift: `+0.0002`
- Avg repetition lift: `-0.0036` (good direction overall)
- Avg wear-through lift: `-0.0087` (high variance)

### Interpretation

- Personalization does **not** inflate engine score.
- Personalization **does** change ranking behavior by user archetype.
- Effects are selective, not universally positive.
- System remains stable without collapse/explosion patterns.

## 11.4 System Behavioral Harness

Location: `tests/system_behavioral_test.py`

Covers:

1. hard invariants,
2. scenario tests (minimalist, explorer, rage, forgetful),
3. fake-feature detection,
4. 30-day regression checks,
5. edge safety.

Important practical note:

- This harness resets and seeds DB for deterministic runs.
- Failures like "no outfits available" are usually candidate-pool constraints, not random crashes.

---

## [Owner: Vaibhavi] 12) What Is Working Well (Current MVP Strengths)

1. End-to-end pipeline is complete and integrated.
2. Segmentation-aware color extraction fixed background leakage.
3. Embedding generation and storage are in place and reusable.
4. Recommendation engine is deterministic and explainable.
5. Feedback loop updates behavior, not just UI state.
6. Offline evaluation is academically defensible.
7. UI exposes internal logic clearly for debugging and presentation.

---

## [Owner: Amogh] 13) Known Constraints and Honest Limitations

1. Single-user auth placeholder (`get_current_user` returns first user).
2. SQLite + local files are dev-oriented, not production scaling setup.
3. "Never show" currently maps to `disliked` action semantics.
4. Small wardrobe can cause no-outfit situations under strict constraints.
5. Some profile fields are stored for future use, not yet scoring inputs.
6. Frontend includes some text encoding artifacts in labels (cosmetic cleanup needed).

---

## [Owner: Shared] 14) Presentation Split for 3 Members

This section is designed so each person can present independently without opening code files.

## [Owner: Amogh] 14.1 Member 1: Amogh (Technical Lead, Deep System)

### Topics to explain

1. Full architecture (API + services + DB + frontend)
2. Vision pipeline internals (SAM, mask-aware HSV+KMeans, CLIP embeddings)
3. Recommendation engine formula and constraints
4. Confidence/challenger/repetition logic
5. Feedback amplification and style profile derivation
6. Trade-offs and future production hardening

### Suggested slide order

1. Architecture diagram
2. Scan pipeline arrow flow
3. Data model + schema evolution
4. Scoring equation breakdown
5. Explanation payload example
6. Failure modes and fallback policy

### Technical speaking script (short)

- "Our system is a hybrid recommender where every score term is bounded and explainable."
- "Perception quality is handled first by SAM mask, then color and CLIP features."
- "Recommendation is not random. It is candidate generation plus deterministic scoring."
- "Feedback changes future ranking through pair bias, bans, and style centroid effects."

## [Owner: Dhanuja] 14.2 Member 2: Dhanuja (Rule-Based Recommender Focus + Product Logic)

### Mandatory included topic

- How rule-based recommender systems work in general
- How those rules are implemented in this project

### Simple explanation points

1. Rule-based recommender means:
   - no black-box neural ranker needed,
   - explicit business/user rules drive output.
2. In our project, rules include:
   - valid top+bottom combination,
   - color harmony,
   - novelty,
   - repetition restrictions,
   - fit preference.
3. Feedback is a rule input:
   - liked/worn boosts,
   - skipped/disliked reduces priority.
4. Explainability:
   - each outfit has reason fields and score parts,
   - user can understand "why this outfit".

### Suggested slide order

1. "What is rule-based recommendation?"
2. "Our rule stack in sequence"
3. "Example score breakdown from UI"
4. "Why this is safer for MVP and college project"

### Easy script lines

- "The engine first filters invalid items, then ranks valid pairs."
- "Rules are additive and bounded, so one noisy signal cannot dominate."
- "This makes outputs consistent, debuggable, and explainable."

## [Owner: Vaibhavi] 14.3 Member 3: Vaibhavi (Evaluation + Testing + Result Meaning)

### Mandatory included topic

- How we tested and evaluated correctness
- What results mean
- Is the system working

### Simple explanation points

1. We used offline synthetic users because live users are unavailable.
2. We ran A/B:
   - baseline (no personalization),
   - personalized (all user signals on).
3. We measured objective metrics:
   - diversity, repetition, wear-through, score, entropy.
4. We added replicates + CI to separate signal vs noise.
5. We also used API behavioral tests for invariants and scenarios.

### Suggested slide order

1. "Evaluation methodology"
2. "Synthetic user archetypes"
3. "Metric table and CI summary"
4. "Key findings and honest conclusion"

### Easy script lines

- "Personalization does not always increase score, and that is expected."
- "It changes selection patterns based on user style."
- "Our tests prove stability and functional behavior, not cosmetic output."

---

## [Owner: Shared] 15) Suggested 12-Minute Presentation Flow

1. **Amogh** (5 min): architecture + engine + technical novelty
2. **Dhanuja** (3.5 min): rule-based recommender and UI interpretation
3. **Vaibhavi** (3.5 min): evaluation/testing proof and conclusion

---

## [Owner: Dhanuja] 16) Demo Walkthrough Script (Live)

1. Open `/frontend/scan.html`
2. Start scan and upload one clothing image
3. Show original + mask + extracted color + suggestions
4. Open `/frontend/wardrobe.html`
5. Set category and item_type for eligibility
6. Open `/frontend/profile.html`
7. Set fit preference and profile details
8. Open `/frontend/outfits.html`
9. Generate outfits, show hero confidence + explanation
10. Click feedback actions and regenerate to show adaptation

---

## [Owner: Amogh] 17) Viva Defense Q&A (Ready Answers)

### Q1: Is this neural recommendation?

- **Answer**: No. It is a hybrid rule-based recommender with embedding similarity features. Embeddings are feature extractors, not a trained ranking model.

### Q2: Where is cosine similarity used?

- **Answer**: In embedding diversity scoring and dislike-neighbor penalty calculations.

### Q3: Why no Pearson similarity?

- **Answer**: Pearson is more suitable for user-item rating vectors. Our current MVP primarily compares visual vectors and behavior signals, so cosine similarity is the right fit.

### Q4: Is this random recommendation?

- **Answer**: No. Candidate generation and ranking are deterministic with explicit score terms and constraints.

### Q5: Is it production ready?

- **Answer**: It is MVP-ready for demo and controlled usage. Full production needs auth, stronger infra, security hardening, and scaling.

---

## [Owner: Amogh] 18) Commercial Extension (Commission-Based New Clothes Recommendation)

Current system can be extended with an affiliate recommendation layer:

`style profile + wardrobe gaps + color/type preferences -> query catalog API -> rank products -> attach affiliate links`

Safe rollout plan:

1. Keep existing outfit recommender unchanged.
2. Add separate "shop similar / fill gap" endpoint.
3. Use same style and fit signals for ranking external products.
4. Track clicks/conversions to evaluate business lift.

---

## [Owner: Shared] 19) Repository Map (Important Files)

- Backend core:
  - `app/main.py`
  - `app/database.py`
  - `app/models/*.py`
  - `app/api/*.py`
  - `app/services/*.py`

- Frontend:
  - `frontend/index.html`
  - `frontend/scan.html`
  - `frontend/wardrobe.html`
  - `frontend/outfits.html`
  - `frontend/profile.html`
  - `frontend/app.js`
  - `frontend/style.css`

- Evaluation:
  - `evaluation/synthetic_users.json`
  - `evaluation/simulate_users.py`
  - `evaluation/metrics.py`
  - `evaluation/report.py`
  - `evaluation/summary.json`
  - `evaluation/report.md`

- System tests:
  - `tests/system_behavioral_test.py`

---

## [Owner: Shared] 20) Final Conclusion

The project is a complete recommender MVP with:

1. robust perception input (SAM + color + CLIP),
2. strict semantic data controls,
3. explainable hybrid scoring,
4. closed feedback loop,
5. offline A/B evaluation with statistical discipline,
6. usable developer UI for demonstration and debugging.

For college-level recommender systems, this is beyond a basic CRUD demo and demonstrates full lifecycle thinking: data quality -> ranking logic -> feedback adaptation -> validation.
