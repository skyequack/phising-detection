# Database

SQLite through SQLAlchemy 2 (`backend/db.py`). Every query goes through the ORM, so values are always bound parameters;
a test scans the backend source for raw SQL. The file defaults to `./phishing.db` (git-ignored); set `DATABASE_URL` to
change it. Tables are created on startup with `create_all`.

```mermaid
erDiagram
    models ||--o{ predictions : produced_by
    predictions ||--|| feature_snapshots : has
    models { int id PK  string version UK  string algorithm  string trained_at  bool active  json metrics  string artifact_dir  string metrics_dir }
    predictions { int id PK  string url_hash  string label  float probability  int model_id FK  datetime created_at }
    feature_snapshots { int id PK  int prediction_id FK_UK  json features }
    audit_logs { int id PK  string actor  string action  string outcome  datetime created_at }
```

## Tables

**`models`**: one row per model version the server has served or trained.
- `version` (unique, 64) is `<algorithm>-<training date>`, for example `rf-2026-10-08`.
- `active` is true for exactly one row; activation clears the others.
- `metrics` holds the test metrics recorded at registration.
- `artifact_dir` and `metrics_dir` are paths (relative to the repository when inside it) where that model's files live.
  Only server code writes them, never a request, so activating a model cannot be pointed at an arbitrary path. A model
  registered without them cannot be activated.

**`predictions`**: one row per `POST /predict`.
- `url_hash`: HMAC-SHA256 hex digest (64 characters) of the canonical URL, indexed. The raw URL is never stored.
- `label`, `probability`, `model_id` (foreign key), `created_at` (indexed, UTC).

**`feature_snapshots`**: one row per prediction (`prediction_id` is unique), holding the 30 feature values as JSON. It lets
a stored prediction be re-examined without keeping the URL. Deleting a prediction deletes its snapshot.

**`audit_logs`**: append-only record of admin activity. No foreign keys, and no raw tokens or request bodies.

| `actor` | `action` | `outcome` |
|---|---|---|
| `admin` | `train_start` | `ok`, `busy` |
| `admin` | `train_finished` | `ok`, `failed` |
| `admin` | `activate:<version>` | `ok`, `not_found`, `failed` |
| `anonymous` | `<METHOD> <route template>`, for example `POST /admin/models/{version}/activate` | `denied` |

The actor is a label, not a user: admin access is one static token ([ADR 002](../adr/002-static-admin-token.md)), so the
log cannot say which person acted.

## URL hashing

`url_hash = HMAC-SHA256(URL_HASH_KEY, canonical(url))`, where the canonical form is the extractor's `normalize_url`
(scheme forced to `http`, leading `www.` and a lone trailing slash removed) lowercased. So `https://www.Example.com/a`
and `http://example.com/a` give the same hash, while `/a/` and `/a` differ. The key makes the hashes impossible to
reverse by hashing a dictionary of known URLs without it.

- If `URL_HASH_KEY` is unset or still `change-me`, a random key is generated for the process and a warning is logged.
  Hashes then stop matching after a restart.
- Changing the key breaks matching between old and new rows. There is no re-keying tool.
- A hash does not protect a URL that someone can already guess and who also holds the key; treat the database file as
  sensitive.

## Time

`created_at` is stored as UTC. SQLite drops the timezone, so the API adds an explicit `+00:00` when it returns a value
(`iso_utc` in `backend/db.py`).

## Limits and decisions to know about

- **No migrations.** `create_all` only creates missing tables. Changing a column on an existing database needs a manual
  migration or deleting the file. Adding `artifact_dir` and `metrics_dir` to `models` was done before any database existed.
- **No retention policy.** Predictions accumulate without bound; add a cleanup job before running this for long.
- **Concurrency.** SQLite allows one writer at a time. Prediction writes run in background tasks, are small, and a failure
  is logged without failing the request. Eighty concurrent requests were tested with no lost writes.
- **PostgreSQL is untested.** `make_engine` accepts any SQLAlchemy URL, but only SQLite has been run.
- **Backups** are the operator's job: copy the file while the server is stopped.
