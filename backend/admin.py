"""Admin endpoints: audited, static-bearer-token protected. Never part of the public prediction flow.

Auth is fail-closed: with ``ADMIN_TOKEN`` unset, shorter than 16 characters, or equal to the
.env.example placeholder, every admin call is refused. Training runs in a separate process
into its own folder under ``ml/runs/`` and never touches the served model; the new model is
registered as inactive and only a separate, audited activation call switches to it.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select

from backend import db as D
from backend.service import ModelNotReady, Predictor

log = logging.getLogger("phishing.admin")
ACTOR = "admin"  # one static token, so the audit actor is a label, not a user
MIN_TOKEN_LENGTH = 16
VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
RUNS_DIR = D.ROOT / "ml" / "runs"
TRAIN_DATA = Path(os.environ.get("TRAIN_DATA_DIR", D.ROOT / "data" / "processed_v2"))

router = APIRouter(prefix="/admin", tags=["admin"])
_bearer = HTTPBearer(auto_error=False)


def _token_ok(creds: HTTPAuthorizationCredentials | None) -> bool:
    expected = os.environ.get("ADMIN_TOKEN", "")
    if len(expected) < MIN_TOKEN_LENGTH or expected == "change-me" or creds is None:
        return False
    return secrets.compare_digest(creds.credentials.encode(), expected.encode())


def require_admin(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if _token_ok(creds):
        return
    route = request.scope.get("route")
    D.audit(request.app.state.db, "anonymous", f"{request.method} {getattr(route, 'path', 'admin')}", "denied")
    raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})


def _paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return run_dir / "models", run_dir / "metrics", run_dir / "figures"


def default_train_runner(run_dir: Path) -> None:
    """Train, then calibrate, in a child process with fixed arguments (nothing comes from the request)."""
    models, metrics, figures = _paths(run_dir)
    base = [sys.executable, "-u", "-m", "ml.train", "--processed", str(TRAIN_DATA), "--models", str(models),
            "--metrics", str(metrics), "--figures", str(figures)]
    with (run_dir / "train.log").open("w", encoding="utf-8") as logf:
        for extra in ([], ["--calibrate"]):
            subprocess.run(base + extra, cwd=D.ROOT, stdout=logf, stderr=subprocess.STDOUT, check=True)


def _run_training(app, run_dir: Path, runner: Callable[[Path], None]) -> None:
    job, factory = app.state.train_job, app.state.db
    outcome, note = "failed", "training failed"
    try:
        runner(run_dir)
        models, metrics, _ = _paths(run_dir)
        predictor = Predictor.load(models, metrics)  # validates artifacts and the feature schema
        with factory() as db:
            if db.scalar(select(D.ModelRow).where(D.ModelRow.version == predictor.version)):
                note = "a model with this version is already registered"
            else:
                D.register_model(db, predictor.model_info(), models, metrics, activate=False)
                outcome, note = "ok", predictor.version
    except Exception:  # noqa: BLE001 - details go to the server log, never to the caller
        log.exception("training run %s failed", run_dir.name)
    job.update(status="succeeded" if outcome == "ok" else "failed", finished_at=_now(), message=note)
    D.audit(factory, ACTOR, "train_finished", outcome)
    job["lock"].release()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@router.post("/train", status_code=202, dependencies=[Depends(require_admin)])
def train(request: Request) -> JSONResponse:
    app = request.app
    job = app.state.train_job
    if not job["lock"].acquire(blocking=False):
        D.audit(app.state.db, ACTOR, "train_start", "busy")
        raise HTTPException(409, "A training run is already in progress")
    run_dir = RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    try:
        for d in _paths(run_dir):
            d.mkdir(parents=True, exist_ok=True)
    except OSError:
        job["lock"].release()
        log.exception("could not create run directory")
        raise HTTPException(500, "Could not start training")
    job.update(status="running", run=run_dir.name, started_at=_now(), finished_at=None, message="")
    D.audit(app.state.db, ACTOR, "train_start", "ok")
    threading.Thread(target=_run_training, args=(app, run_dir, app.state.train_runner), daemon=True).start()
    return JSONResponse(status_code=202, content={"status": "running", "run": run_dir.name})


@router.get("/train/status", dependencies=[Depends(require_admin)])
def train_status(request: Request) -> dict:
    return {k: v for k, v in request.app.state.train_job.items() if k != "lock"}


@router.get("/models", dependencies=[Depends(require_admin)])
def list_models(request: Request) -> list[dict]:
    with request.app.state.db() as db:
        rows = db.scalars(select(D.ModelRow).order_by(D.ModelRow.id)).all()
    return [{"version": r.version, "algorithm": r.algorithm, "trained_at": r.trained_at, "active": r.active,
             "activatable": bool(r.artifact_dir)} for r in rows]


@router.post("/models/{version}/activate", dependencies=[Depends(require_admin)])
def activate(version: str, request: Request) -> dict:
    app = request.app
    if not VERSION_RE.fullmatch(version):
        raise HTTPException(422, "Invalid model version")
    action = f"activate:{version}"
    with app.state.db() as db:
        row = db.scalar(select(D.ModelRow).where(D.ModelRow.version == version))
        found = None if row is None else (row.artifact_dir, row.metrics_dir)
    if found is None:
        D.audit(app.state.db, ACTOR, action, "not_found")
        raise HTTPException(404, "Unknown model version")
    try:
        predictor = Predictor.load(D.resolve(found[0]), D.resolve(found[1]))
        if predictor.version != version:
            raise ModelNotReady("artifacts do not match the registered version")
    except Exception:  # noqa: BLE001 - missing path, corrupt file, schema mismatch: all the same to the caller
        log.exception("could not load model %s for activation", version)
        D.audit(app.state.db, ACTOR, action, "failed")
        raise HTTPException(409, "Model artifacts are not available")
    with app.state.db() as db:
        D.set_active(db, db.scalar(select(D.ModelRow).where(D.ModelRow.version == version)))
        db.commit()
    app.state.predictor = predictor
    D.audit(app.state.db, ACTOR, action, "ok")
    return {"active": version}


@router.get("/audit-logs", dependencies=[Depends(require_admin)])
def audit_logs(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=10_000_000)) -> dict:
    with request.app.state.db() as db:
        total = db.scalar(select(func.count(D.AuditLog.id)))
        rows = db.scalars(select(D.AuditLog).order_by(D.AuditLog.id.desc()).limit(limit).offset(offset)).all()
    return {"total": total, "limit": limit, "offset": offset, "items": [
        {"id": r.id, "actor": r.actor, "action": r.action, "outcome": r.outcome,
         "created_at": D.iso_utc(r.created_at)} for r in rows]}
