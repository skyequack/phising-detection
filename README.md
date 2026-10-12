# Phishing URL detector

Classifies a URL as phishing or legitimate from the characters in the URL alone (30 lexical features, no page fetch),
and shows which features drove each verdict. A FastAPI backend serves the model, a Next.js dashboard shows predictions,
history, model details and measured metrics.

**Read this before relying on it.** On the held-out test split from the original dataset the model scores F1 0.81.
On URLs from other sources it does worse: about 77% of newly reported phishing URLs are caught, and about 15% of real
links from popular sites are wrongly flagged. The Metrics page and [docs/PROGRESS.md](docs/PROGRESS.md) list every
number and its caveats. It is a research project, not a security product.

## Requirements

- Python 3.12 (the project is pinned to `>=3.12,<3.13`)
- Node.js 20 or newer, for the dashboard
- At least 2 GB of free memory to serve the model (the Random Forest is 1.1 GB on disk and about 1.3 GB resident)

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Windows; on Linux/macOS use .venv/bin/python
cd frontend && npm install && cd ..
```

Copy `.env.example` to `.env` and set real values. Variables the code reads:

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLite database | `sqlite:///./phishing.db` |
| `URL_HASH_KEY` | Key for hashing URLs before storing them. Unset or `change-me` gives a random key per process, so hashes change on restart | none |
| `ADMIN_TOKEN` | Bearer token for `/admin/*`. Admin stays disabled unless this is at least 16 characters | none |
| `CORS_ORIGINS` | Allowed dashboard origins, comma separated | `http://localhost:3000,http://127.0.0.1:3000` |
| `PHISHING_MODELS_DIR`, `PHISHING_METRICS_DIR` | Where the served model and its reports are read from | `ml/models`, `reports/metrics` |
| `TRAIN_DATA_DIR` | Data used by `POST /admin/train` | `data/training` |
| `NEXT_PUBLIC_API_URL` | Backend URL used by the dashboard | `http://localhost:8000` |

Generate a token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. The shell does not load `.env`
for you; export the variables or set them in your launcher.

## Run

```bash
.venv\Scripts\python -m uvicorn backend.main:app --port 8000
cd frontend && npm run dev          # http://localhost:3000
```

The trained model files (`ml/models/*.joblib`) are not in git because of their size. Without them the API starts but
`/predict` answers 503; train or restore them first (see below). In the Claude desktop app, `.claude/launch.json`
defines `backend` and `frontend` servers.

Quick check:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"url\": \"http://paypal-secure-login.example.ru/signin\"}"
```

## Tests

```bash
.venv\Scripts\python -m pytest -q          # 280 tests, about 20 seconds
cd frontend && npx tsc --noEmit && npm run lint
```

Details and known gaps: [docs/testing/README.md](docs/testing/README.md).

## Reproduce the model

Data is not in git. You need the raw dataset at `data/raw/URL dataset.csv` (Mendeley Data, DOI 10.17632/vfszbj9b36.1,
CC BY 4.0, see [docs/dataset/README.md](docs/dataset/README.md)) and, for v2, the three external downloads under
`data/external/` described there (PhishTank, Tranco, OpenPhish) plus the link crawl.

```bash
.venv\Scripts\python -m ml.data_prep                      # clean and split the original data
.venv\Scripts\python -m ml.phishtank_prep                 # PhishTank cleaning and time split
.venv\Scripts\python -m ml.crawl_links                    # optional: re-run the link crawl (about 1,000 requests)
.venv\Scripts\python -m ml.build_training_set             # build data/training
.venv\Scripts\python -u -m ml.train --verbose            # about 1 hour
.venv\Scripts\python -m ml.train --calibrate
.venv\Scripts\python -m ml.evaluate                    # add --no-ablation to skip the slow part
.venv\Scripts\python -m ml.heldout_eval                   # score every held-out set
.venv\Scripts\python -m ml.benchmark                      # latency, throughput, memory
```

Fixed seed 42 throughout. A rerun with the same data should give the same metrics; this has not been re-verified end to
end (see [docs/PROGRESS.md](docs/PROGRESS.md)).

## Layout

| Path | What |
|---|---|
| `ml/` | Cleaning, feature extraction (`features.py`), training, calibration, evaluation, benchmarks |
| `backend/` | FastAPI app (`main.py`), model and explanation (`service.py`), database (`db.py`), admin (`admin.py`) |
| `frontend/` | Next.js dashboard (Detector, History, Model, Metrics) |
| `tests/` | pytest suite |
| `reports/metrics`, `reports/figures` | Measured results for the model in use; `*_v1` hold the earlier model |
| `docs/` | [architecture](docs/architecture.md), [API](docs/design/api.md), [database](docs/design/database.md), [feature schema](docs/design/feature-schema.md), [decision records](docs/adr), [dataset](docs/dataset/README.md), [progress and disclosures](docs/PROGRESS.md) |

## Privacy and safety

URLs are only parsed, never fetched or opened. Raw URLs are not stored; the database keeps a keyed hash, the label,
the score and the 30 feature values. The one outbound-network tool, the link crawler, is a separate offline script and
is not imported by the server.
