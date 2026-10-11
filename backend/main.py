"""FastAPI app: POST /predict, GET /predictions, GET /models, GET /metrics[/{version}], /admin/*.
Run: uvicorn backend.main:app"""

from __future__ import annotations

import os
import re
import threading
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend import admin
from backend import db as D
from backend.service import METRICS, MODELS, ModelNotReady, Predictor
from ml.features import MAX_URL_LENGTH, InvalidURLError

ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
MAX_OFFSET = 10_000_000  # larger values overflow SQLite integers
VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # allow-list for the {model_version} path parameter


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)


def load_predictor(factory) -> Predictor:
    """The model an admin last activated if its artifacts still load, otherwise the default folder."""
    with factory() as db:
        active = db.scalar(select(D.ModelRow).where(D.ModelRow.active, D.ModelRow.artifact_dir.is_not(None)))
        saved = (active.version, active.artifact_dir, active.metrics_dir) if active else None
    if saved:
        try:
            p = Predictor.load(D.resolve(saved[1]), D.resolve(saved[2]))
            if p.version == saved[0]:
                return p
        except ModelNotReady:
            pass
    p = Predictor.load()
    with factory() as db:
        D.register_model(db, p.model_info(), MODELS, METRICS)
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = D.make_engine()
    app.state.db = D.init_db(engine)
    app.state.hasher = D.UrlHasher()
    app.state.train_job = {"status": "idle", "lock": threading.Lock()}
    app.state.train_runner = admin.default_train_runner
    try:
        app.state.predictor = load_predictor(app.state.db)
    except ModelNotReady as e:
        app.state.predictor = None
        app.state.not_ready = str(e)
    yield
    engine.dispose()


app = FastAPI(title="Phishing URL detector", lifespan=lifespan)
app.include_router(admin.router)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def _predictor(request: Request) -> Predictor:
    p = request.app.state.predictor
    if p is None:
        raise HTTPException(503, "Model not available yet")
    return p


def _db(request: Request):
    factory = getattr(request.app.state, "db", None)
    if factory is None:
        raise HTTPException(503, "Database not available")
    return factory()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Generic body: FastAPI's default echoes the submitted value (and crashes on non-UTF-8 input)."""
    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


@app.exception_handler(InvalidURLError)
async def invalid_url(_: Request, exc: InvalidURLError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": "Invalid URL"})


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok" if request.app.state.predictor else "model_not_ready"}


@app.post("/predict")
def predict(body: PredictRequest, request: Request, background: BackgroundTasks) -> dict:
    result = _predictor(request).predict(body.url)
    if getattr(request.app.state, "db", None) is not None:  # absent when the app runs without its lifespan
        background.add_task(D.record_prediction, request.app.state.db, request.app.state.hasher, body.url, result)
    return result


@app.get("/predictions")
def predictions(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=MAX_OFFSET)) -> dict:
    """Newest first. Only hashes are stored, never the URLs themselves."""
    with _db(request) as db:
        total = db.scalar(select(func.count(D.Prediction.id)))
        rows = db.execute(
            select(D.Prediction, D.ModelRow.version).join(D.ModelRow, D.Prediction.model_id == D.ModelRow.id)
            .order_by(D.Prediction.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "limit": limit, "offset": offset, "items": [
        {"id": p.id, "url_hash": p.url_hash, "label": p.label, "probability": p.probability,
         "model_version": v, "created_at": D.iso_utc(p.created_at)} for p, v in rows]}


@app.get("/models")
def models(request: Request) -> dict:
    return _predictor(request).model_info()


@app.get("/metrics")
def metrics(request: Request) -> dict:
    return _predictor(request).metrics()


@app.get("/metrics/{model_version}")
def metrics_for(model_version: str, request: Request) -> dict:
    if not VERSION_RE.fullmatch(model_version):
        raise HTTPException(422, "Invalid model version")
    predictor = _predictor(request)
    if model_version == predictor.version:
        return predictor.metrics()
    with _db(request) as db:
        row = db.scalar(select(D.ModelRow).where(D.ModelRow.version == model_version))
    if row is None:
        raise HTTPException(404, "Unknown model version")
    return {"test": {"selected_at_chosen_threshold": row.metrics} if row.metrics else None, "threshold_sweep": []}
