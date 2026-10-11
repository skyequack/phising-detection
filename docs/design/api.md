# API reference

Base URL `http://localhost:8000`. JSON in and out. Source: `backend/main.py` (public routes), `backend/admin.py` (admin).
Every response carries `X-Content-Type-Options: nosniff`. CORS allows only the origins in `CORS_ORIGINS`, methods
`GET` and `POST`, and the `Content-Type` header (not `Authorization`, so a browser page on another origin cannot call
admin routes).

## Errors

Error bodies are fixed strings. They never contain the submitted value, file paths, stack traces or library names.

| Status | Body | When |
|---|---|---|
| 401 | `{"detail": "Unauthorized"}` + `WWW-Authenticate: Bearer` | Admin route without a valid token (same body whatever the reason) |
| 404 | `{"detail": "Unknown model version"}` | Version not registered |
| 409 | `{"detail": "A training run is already in progress"}` / `{"detail": "Model artifacts are not available"}` | Training busy / activation could not load the model |
| 422 | `{"detail": "Invalid URL"}` | URL is only whitespace (rejected by the feature extractor) |
| 422 | `{"detail": "Invalid request"}` | Body is not JSON, wrong types, missing fields, URL empty or over 4096 characters, paging value out of range |
| 422 | `{"detail": "Invalid model version"}` | Version does not match `^[A-Za-z0-9._-]{1,64}$` |
| 503 | `{"detail": "Model not available yet"}` | No trained model could be loaded |

## Public routes

### `GET /health`
`{"status": "ok"}` or `{"status": "model_not_ready"}`.

### `POST /predict`
Request: `{"url": "<1 to 4096 characters>"}`. Other fields are ignored.

Response 200:

```json
{
  "label": "phishing",
  "probability": 0.856,
  "threshold": 0.43,
  "model_version": "rf-2026-10-08",
  "algorithm": "Random Forest",
  "features": { "<30 feature names>": 0.0 },
  "explanation": {
    "top_features": [ { "feature": "num_special_chars", "group": "char_count", "value": 12, "legit_median": 3, "contribution": 0.07 } ],
    "group_contributions": { "char_count": 0.1, "entropy": 0.0 }
  },
  "url_parts": { "scheme": "http", "host": "example.ru", "path": "/signin", "query": "", "fragment": "" }
}
```

- `probability` is the model's calibrated score, not a measured chance on real traffic (see the README). `label` is
  `phishing` when it is at or above `threshold`.
- `contribution` values are estimates from swapping a feature for the typical legitimate value (and the reverse); they do
  not add up exactly. See [ADR 004](../adr/004-heuristic-explanation.md).
- `url_parts` is for display only and is empty strings if the URL cannot be split.
- The URL is only parsed. Nothing is fetched.
- After the response is sent, the prediction is stored (hash, label, score, features). A failed write is logged and does
  not affect the response.

### `GET /predictions?limit=20&offset=0`
Newest first. `limit` 1 to 100, `offset` 0 to 10,000,000, otherwise 422.

```json
{ "total": 4, "limit": 20, "offset": 0,
  "items": [ { "id": 4, "url_hash": "<64 hex>", "label": "legitimate", "probability": 0.28,
               "model_version": "rf-2026-10-08", "created_at": "2026-10-11T06:43:00.689713+00:00" } ] }
```
URLs are not stored, so they cannot be returned. `created_at` is UTC with an explicit offset.

### `GET /models`
The active model: `version`, `algorithm`, `trained_at`, `feature_schema_version`, `threshold` (value, rule and its
out-of-fold precision/recall/F1), `hyperparameters`, `train_rows`, `feature_importance`, `group_importance`,
`test_metrics`, `candidates` (cross-validation scores of all five algorithms).

### `GET /metrics` and `GET /metrics/{model_version}`
For the active model, the contents of its report files; any part whose file is missing is `null` or `[]`:
`test` (test metrics and confusion matrix), `threshold_sweep`, `comparison`, `ablation`,
`calibration` (`summary`, `reliability`), `curves` (thinned ROC and precision-recall points per algorithm) and `external`
(results on data never used for training or tuning, earlier model vs current). For an older registered version only
`test` (its stored test metrics, under `selected_at_chosen_threshold`) and an empty `threshold_sweep` are returned.
Unknown version: 404.

## Admin routes

All need `Authorization: Bearer <ADMIN_TOKEN>`. Admin is disabled (every call 401) unless `ADMIN_TOKEN` is at least 16
characters and not `change-me`. Every call, allowed or denied, is written to `audit_logs`
([database](database.md)). There is no rate limit on failed attempts.

| Route | Purpose |
|---|---|
| `GET /admin/models` | Registered models: `version`, `algorithm`, `trained_at`, `active`, `activatable` |
| `POST /admin/models/{version}/activate` | Load that model's artifacts and serve it; persists across restarts. 404 unknown, 409 if its files are gone, 422 bad version |
| `POST /admin/train` | Start a training run in a child process (train, then calibrate) using `TRAIN_DATA_DIR`. 202 `{"status": "running", "run": "<timestamp>"}`, 409 if one is running. Output goes to `ml/runs/<timestamp>/`; the new model is registered inactive |
| `GET /admin/train/status` | `status` (`idle`, `running`, `succeeded`, `failed`), `run`, `started_at`, `finished_at`, `message` |
| `GET /admin/audit-logs?limit=50&offset=0` | Audit trail, newest first; `limit` 1 to 200 |

Training notes: a run that produces a version already registered (same algorithm, same day) fails and is reported as
such. A run takes about an hour and rewrites the out-of-fold cache inside `TRAIN_DATA_DIR`.

## Not implemented

Rate limiting, per-user accounts, token rotation without a restart, HTTPS (put a reverse proxy in front for any
non-local use).
