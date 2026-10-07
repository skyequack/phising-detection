# Progress

Checklist against the implementation plan (`~/.claude/plans/okay-now-make-an-mellow-oasis.md`). Last updated 2026-10-07. `[x]` done and verified, `[~]` partly done or needs a decision, `[ ]` not started.

Latest commits (calibration is the newest, see `git log`): Phase 1 `1e4d6f8`, Phase 2 `9c87d07` / `93529b1`, Phase 3 `b8f2017`, shortcut fix and retrain `58084cf`, threshold change `85f65be`. 84 tests pass (`.venv\Scripts\python -m pytest -q`).

## Phase 0: Environment
- [x] Python 3.12 `.venv`, `pyproject.toml` (packages `ml`, `backend`), pinned `requirements.txt`
- [x] `.gitignore`, `.env.example` (`ADMIN_TOKEN`, `URL_HASH_KEY`, `DATABASE_URL`)
- [x] `git init`; imports of sklearn, xgboost, fastapi verified

## Phase 1: Data cleaning (`ml/data_prep.py`)
- [x] Cleaning, canonical dedupe, conflict removal, stratified 80/20 split (seed 42)
- [x] `tests/test_data_prep.py`; counts match `docs/dataset/README.md`

## Phase 2: Feature extractor (`ml/features.py`)
- [x] 30 features, `FEATURE_NAMES`, `FEATURE_GROUPS` (final groups), `extract_matrix`
- [x] `InvalidURLError` (422), malformed-URL fallback, 4096 limit
- [x] `normalize_url` (scheme, `www.`, trailing slash), schema version 2
- [x] `tests/test_features.py`; `docs/design/feature-schema.md`
- [ ] Re-measure extraction throughput for schema v2
- [ ] UI wording: show the probability as a "model score", not "confidence" (check `frontend/components/Gauge.tsx`)

## Phase 3: Training and evaluation
- [x] `ml/preprocessing.py`, `ml/train.py`, `ml/evaluate.py`
- [x] 5 algorithms tuned (5-fold stratified, random search); SVM on 50k subsample; RF 4 iterations
- [x] Model selected (Random Forest), threshold chosen on out-of-fold data (max F1, now 0.41 after calibration), model frozen with SHA-256
- [x] Test metrics, confusion matrices, ROC/PR curves, ablation in `reports/`
- [x] `tests/test_train.py`
- [x] Calibration (`ml/calibration.py`, `python -m ml.train --calibrate`): isotonic fitted on out-of-fold train predictions; test ECE 0.0246 to 0.0032; threshold re-picked on calibrated scores (0.41); `tests/test_calibration.py`
- [ ] Re-run `python -m ml.train` with the same seed and confirm identical metrics (about 1 hour)
- [ ] **Decide on the generalization problem** (see below)

## Phase 4: Prediction library
- [~] Prediction and explanation live in `backend/service.py`; `ml/predict.py` was not created (decision recorded in `architecture.md` section 12)
- [x] `tests/test_model.py` (artifact schema, scheme/www invariance, obvious phishing flagged, no network)
- [x] `ml/sanity_check.py` and `tests/data/sanity_urls.json`

## Phase 5: Backend
- [x] FastAPI app: `/health`, `POST /predict`, `GET /models`, `GET /metrics`, CORS for the dashboard origin, generic 422/503 errors
- [x] `tests/test_api.py` (predict shape, bad input, error bodies)
- [ ] SQLite via SQLAlchemy: `predictions`, `feature_snapshots`, `models`, `audit_logs`
- [ ] HMAC-SHA256 URL hashing; `BackgroundTasks` write on `/predict`
- [ ] `GET /predictions` (paged); `GET /metrics/{model_version}` with an allow-list regex
- [ ] Admin bearer-token auth (`ADMIN_TOKEN`, `secrets.compare_digest`), `POST /admin/train`, `POST /admin/models/{version}/activate`, audit log
- [ ] Optional rate limit (`slowapi`)

## Phase 6: Security and performance tests
- [ ] `tests/test_security.py` (SQLi, XSS, path traversal, unauthorized admin, no internals in errors)
- [ ] `tests/test_performance.py` and `reports/metrics/performance.json` (measured latency p50/p95, memory). A first spot check measured about 50 ms per `/predict` call; not a reported figure.

## Phase 7: Frontend
- [x] Next.js app scaffolded; typed API client; Detector page with result, gauge, contribution chart and model panel; type-checks (`tsc --noEmit`)
- [~] Not yet verified in a browser against the live backend (start both with `.claude/launch.json`: `backend` on 8000, `frontend` on 3000)
- [ ] History, Model Info and Metrics pages (confusion matrix, ROC, PR, comparison table, ablation)

## Phase 8: Documentation
- [x] `docs/architecture.md` updated (4-entity schema, auth, SVM subsample, threshold, groups, status)
- [x] `docs/dataset/README.md` (shortcut signals), `docs/design/feature-schema.md`
- [ ] `docs/design/api.md`, `docs/design/database.md`
- [ ] `docs/testing/` results; `docs/adr/` ADRs 001 SQLite, 002 static token, 003 SVM subsample, 004 heuristic explanation, 005 URL normalization
- [ ] `README.md` with setup, run and reproduce commands
- [ ] Report fixes: dataset citation and CC BY 4.0 attribution, PGD-vs-MCA wording, "Chapter 7" to "Chapter 8", placeholders, single-source limitation

## Where to pick up

1. **Generalization.** Held-out test: F1 0.831, recall 0.826, precision 0.836 (`reports/metrics/test_metrics.json`). On 40 real legitimate URLs the model flags 10 as phishing (25%), and catches 19 of 20 synthetic phishing URLs (`reports/metrics/sanity_check.json`). The single-source legitimate class is narrow. Choose: add a second legitimate source (Tranco) and a second phishing feed and retrain, or keep the dataset and document the limitation. Either way, quote metrics as dataset-specific.
2. **Backend persistence and admin** (Phase 5 unchecked items), then the History / Model Info / Metrics pages.
3. Security and performance tests, then the remaining docs and report fixes.

## Disclose in the report
- First models learned a scheme/`www.` shortcut (F1 about 0.97); fixed by URL normalization; honest F1 about 0.83.
- SVM trained on a 50k subsample; Random Forest searched 4 iterations vs 8 for the others.
- Threshold rule changed from max F1 with FNR <= 5% (0.14) to max F1 after test results were seen (0.48, then 0.41 on calibrated scores); recorded in `metadata.json`.
- Probabilities are calibrated in-distribution only (test ECE 0.003) and are overconfident on real-world URLs the dataset does not represent.
- Subdomain counting uses a small suffix list, not a public-suffix database.
