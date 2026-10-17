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
- About 2 GB of free memory to serve the model (the Random Forest is 1.1 GB on disk and about 1.3 GB resident), and
  about 2.5 GB of disk for all trained model files
- Training the model takes about 1 hour on a laptop-class CPU

Commands below use Windows paths (`.venv\Scripts\python`). On Linux or macOS use `.venv/bin/python` instead.

## What is in the repository

| Included | Not included (you generate it) |
|---|---|
| Source code, tests, docs, measured results in `reports/` | Trained models `ml/models/*.joblib` (2.3 GB) |
| Input data in `data/raw/` and `data/external/` (about 45 MB) | Cleaned and combined data in `data/processed/` and `data/training/` |

Everything not included is rebuilt by the steps under [Reproduce the results](#reproduce-the-results).

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Copy `.env.example` to `.env` and set real values. The shell does not load `.env` for you, so export the variables
(or set them in the terminal that starts the backend) before running the API. Everything has a working default except
the two secrets, which are only needed for the history hashing and the admin routes.

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLite database | `sqlite:///./phishing.db` |
| `URL_HASH_KEY` | Key for hashing URLs before storing them. Unset or `change-me` gives a random key per process, so hashes change on restart | none |
| `ADMIN_TOKEN` | Bearer token for `/admin/*`. Admin stays disabled unless this is at least 16 characters | none |
| `CORS_ORIGINS` | Allowed dashboard origins, comma separated | `http://localhost:3000,http://127.0.0.1:3000` |
| `PHISHING_MODELS_DIR`, `PHISHING_METRICS_DIR` | Where the served model and its reports are read from | `ml/models`, `reports/metrics` |
| `TRAIN_DATA_DIR` | Data used by `POST /admin/train` | `data/training` |
| `NEXT_PUBLIC_API_URL` | Backend URL used by the dashboard | `http://localhost:8000` |

Generate a token with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

## Reproduce the results

Run these from the repository root, in order. Fixed seed 42 throughout.

```bash
# 1. Clean and split the original dataset  ->  data/processed/   (seconds)
.venv\Scripts\python -m ml.data_prep

# 2. Clean the PhishTank download and split it by time  ->  data/processed/
.venv\Scripts\python -m ml.phishtank_prep

# 3. Combine everything into the training set  ->  data/training/
.venv\Scripts\python -m ml.build_training_set

# 4. Tune and train five algorithms, select and freeze the best  ->  ml/models/   (about 1 hour)
.venv\Scripts\python -u -m ml.train --verbose

# 5. Calibrate the selected model and re-pick the threshold  ->  ml/models/
.venv\Scripts\python -m ml.train --calibrate

# 6. Test-set metrics, figures, curves  ->  reports/metrics, reports/figures
.venv\Scripts\python -m ml.evaluate --no-ablation        # drop --no-ablation to also redo the ablation (slow)

# 7. Score every held-out set (PhishTank newest 15%, OpenPhish, crawled links, Tranco, hand-made URLs)
.venv\Scripts\python -m ml.heldout_eval
.venv\Scripts\python -m ml.sanity_check

# 8. Optional: latency, throughput and memory of the served model
.venv\Scripts\python -m ml.benchmark
```

Notes:

- Step 1 is deterministic: it reproduces `data/processed/train.parquet` and `test.parquet` byte for byte
  (checked). The expected row counts are in [docs/dataset/README.md](docs/dataset/README.md).
- The link crawl (`ml.crawl_links`) needs live web access and cannot be repeated exactly, so its output is shipped in
  `data/external/crawl/` and used as is. You do not need to run it.
- Steps 4 to 7 overwrite the files in `reports/`. Compare your output with the committed copies (`git diff reports/`)
  to check the reproduction. Small differences in the last digits can come from a different CPU or library build.
- A same-seed rerun of step 4 on a second machine has not been done end to end, so identical metrics are expected but
  not verified (see [docs/PROGRESS.md](docs/PROGRESS.md)).
- For a quick smoke test of the training code, add `--quick` to step 4 (20k rows, 2 iterations). It writes to the same
  `ml/models/` folder, so rerun the full step afterwards.

## Run the application

Needs the trained model from step 4 and 5. Without it the API starts but `/predict` answers 503.

```bash
# terminal 1: backend on http://localhost:8000
.venv\Scripts\python -m uvicorn backend.main:app --port 8000

# terminal 2: dashboard on http://localhost:3000
cd frontend && npm run dev
```

Quick check of the API:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"url\": \"http://paypal-secure-login.example.ru/signin\"}"
```

The interactive API docs are at http://localhost:8000/docs. The dashboard has four pages: Detector, History, Model and
Metrics.

## Tests

```bash
.venv\Scripts\python -m pytest -q          # 280 tests, about 20 seconds
cd frontend && npx tsc --noEmit && npm run lint
```

Some tests load the served model, so run them after training. Details and known gaps:
[docs/testing/README.md](docs/testing/README.md).

## Layout

| Path | What |
|---|---|
| `ml/` | Cleaning, feature extraction (`features.py`), training, calibration, evaluation, benchmarks |
| `backend/` | FastAPI app (`main.py`), model and explanation (`service.py`), database (`db.py`), admin (`admin.py`) |
| `frontend/` | Next.js dashboard (Detector, History, Model, Metrics) |
| `tests/` | pytest suite |
| `data/raw`, `data/external` | Input data. See [docs/dataset/README.md](docs/dataset/README.md) for sources, checksums and cleaning results |
| `reports/metrics`, `reports/figures` | Measured results for the model in use |
| `docs/` | [architecture](docs/architecture.md), [API](docs/design/api.md), [database](docs/design/database.md), [feature schema](docs/design/feature-schema.md), [decision records](docs/adr), [dataset](docs/dataset/README.md), [testing](docs/testing/README.md), [progress and disclosures](docs/PROGRESS.md) |

## Privacy and safety

URLs are only parsed, never fetched or opened. Raw URLs are not stored; the database keeps a keyed hash, the label,
the score and the 30 feature values. The one outbound-network tool, the link crawler, is a separate offline script and
is not imported by the server. The input data in `data/external/` is for academic use within this project and should
not be redistributed.
