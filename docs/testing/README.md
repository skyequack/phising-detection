# Testing

Run date 2026-10-11: **280 tests, all passing, about 20 seconds** (`.venv\Scripts\python -m pytest -q`), on Windows 11
with Python 3.12. Frontend: `npx tsc --noEmit` and `npm run lint` pass with no warnings.

Tools: pytest, FastAPI `TestClient`, and a real uvicorn process for the benchmark. API tests use a small logistic-regression
model built in `tests/conftest.py`, so they do not depend on the trained artifacts; the model, sanity and performance tests
do use the real files in `ml/models` and skip if they are absent.

## Results by file

| File | Tests | What it covers |
|---|---|---|
| `test_data_prep.py` | 17 | Cleaning rules, canonical dedupe and conflict removal, stratified split, no train/test overlap |
| `test_features.py` | 36 | 30 features and their order, URL normalization, malformed and oversized input, determinism |
| `test_train.py` | 11 | Threshold selection, model selection rule, a quick end-to-end training smoke test |
| `test_calibration.py` | 6 | Calibration metrics and the calibrated wrapper |
| `test_model.py` | 5 | Saved model: artifact schema, scheme and `www.` invariance, obvious phishing flagged, no network access |
| `test_api.py` | 9 | `/predict` shape, labels, bad input, error bodies, `/models`, `/metrics` keys, missing model |
| `test_persistence.py` | 18 | Storage without raw URLs, hash normalization and keys, paging, failed write isolation, model registry, version allow-list |
| `test_admin.py` | 40 | Token auth (missing, wrong, weak, unset), audit of denials, activation, background training, one run at a time, failure handling, restart behaviour |
| `test_security.py` | 131 | See below |
| `test_performance.py` | 7 | Loose regression budgets, see below |

### Security sweep (`test_security.py`)
- 29 hostile URLs (SQL and script strings, `javascript:` and `file:` schemes, traversal, NUL and CRLF, over-long hosts, lone
  surrogates, bidi characters) through `/predict`: status 200 or 422, JSON content type, no stack traces, paths or library
  names in the body, nothing reflected into history, database intact.
- Wrong body shapes (null, numbers, lists, objects, non-JSON, invalid UTF-8, 3 MB, deeply nested): 4xx, and the submitted
  value is not echoed back.
- Unknown routes and wrong methods, paging values that overflow integers, traversal in version parameters on both public
  and admin routes.
- Admin bypass attempts: token in query string, cookie or alternate headers, near-miss tokens, a 100 kB token, weak configured
  token. Every admin route answers 401 with an identical body.
- CORS does not reflect unknown origins and cannot be used to authorize `Authorization`; `nosniff` on every response.
- Source checks: no raw SQL in the backend, no network or shell imports on the prediction path, the single subprocess call
  uses a fixed argument list, no hard-coded secrets, `.env` ignored by git.

### Performance (`test_performance.py` and `python -m ml.benchmark`)
The tests are guards with at least 5x headroom: extraction above 3,000 URLs/s, median prediction under 150 ms, model load
under 20 s, memory under 2.5 GB, served forest single-threaded, and 80 concurrent requests all succeed with none of their
stored rows lost. The reported numbers come from the benchmark and are in `reports/metrics/performance.json` (one machine,
loopback, one worker): prediction p50 30 ms, p95 32 ms, p99 85 ms; about 20,000 URLs/s for feature extraction; 1.1 GB model
file, about 1.34 GB resident; 8 concurrent clients reach only about 13 requests/s (p95 312 ms).

## Defects the tests found
- Invalid UTF-8 request bodies caused HTTP 500: FastAPI's default 422 handler crashed encoding the input.
- Validation errors echoed the submitted value back to the caller.
- A very large `offset` caused HTTP 500 (SQLite integer overflow).
- Timestamps were returned without a timezone, so a browser in another timezone showed them shifted.
- The served forest used all CPU cores per request, doubling latency (not a test failure, found by the benchmark).

## Checks that the checks work
Auth was deliberately broken once (the token check replaced by `if True`): 31 admin tests failed, then the file was restored.
The first run of the security sweep had 14 failures: three real defects (the first three above) and the rest mistakes in the tests themselves, fixed in the tests.

## Known gaps
- **No automated frontend tests.** The four pages were checked by hand in a browser (desktop and a 375 px phone width),
  with `tsc` and `eslint`.
- **`POST /admin/train` is tested with a stand-in runner.** A real run takes about an hour; the command it launches is the
  same one used by hand, but the full path through the endpoint has not been run.
- **Retraining reproducibility** (same seed, identical metrics) has not been re-verified end to end.
- **No load test beyond 8 concurrent clients,** and none over a real network or with HTTPS.
- **The test split is not pristine for v2.** It was read when evaluating the first model, before v2 was designed; see
  [PROGRESS.md](../PROGRESS.md) for how this is disclosed.
- Model-quality results (metrics, held-out comparisons, calibration, ablation) live in `reports/metrics` and on the
  dashboard's Metrics page; they are measurements, not pass/fail tests.
