"""SQLite persistence (SQLAlchemy ORM only, no string-built SQL).

Four tables: models, predictions, feature_snapshots, audit_logs. Raw URLs are never
stored: ``url_hash`` is an HMAC-SHA256 of the canonical URL keyed by ``URL_HASH_KEY``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone

from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from ml.features import normalize_url

ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("phishing.db")
DEFAULT_URL = "sqlite:///./phishing.db"
_PLACEHOLDER_KEYS = {"", "change-me"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelRow(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    trained_at: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(default=False)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Where this model's artifacts live. Only server code writes these, never a request, so activation
    # cannot be pointed at an arbitrary path.
    artifact_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_created", "created_at"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    url_hash: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(16))
    probability: Mapped[float] = mapped_column(Float)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    snapshot: Mapped["FeatureSnapshot"] = relationship(back_populates="prediction", cascade="all, delete-orphan")


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    features: Mapped[dict] = mapped_column(JSON)
    prediction: Mapped[Prediction] = relationship(back_populates="snapshot")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def iso_utc(moment: datetime) -> str:
    """ISO 8601 with an explicit UTC offset. SQLite stores no timezone, so a value read back is naive UTC."""
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()


def make_engine(url: str | None = None) -> Engine:
    url = url or os.environ.get("DATABASE_URL", DEFAULT_URL)
    if url.startswith("sqlite"):
        kw: dict = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kw["poolclass"] = StaticPool  # one shared connection so tables survive across sessions
        return create_engine(url, **kw)
    return create_engine(url)


def init_db(engine: Engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


class UrlHasher:
    """HMAC-SHA256 over the canonical URL (normalized and lowercased).

    Without ``URL_HASH_KEY`` (or with the .env.example placeholder) an ephemeral random key
    is used: hashes are then stable only until the process restarts.
    """

    def __init__(self, key: str | None = None):
        key = os.environ.get("URL_HASH_KEY", "") if key is None else key
        if key in _PLACEHOLDER_KEYS:
            log.warning("URL_HASH_KEY not set; using an ephemeral key (hashes change on restart)")
            key = secrets.token_hex(32)
        self._key = key.encode()

    def __call__(self, url: str) -> str:
        canon = normalize_url(url.strip()).lower()
        return hmac.new(self._key, canon.encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()


def rel(path: Path) -> str:
    """Store paths inside the repo relative to it, so the database survives a move."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(stored: str | None) -> Path | None:
    return None if not stored else (ROOT / stored)


def register_model(db: Session, info: dict, artifact_dir: Path | None = None, metrics_dir: Path | None = None,
                   activate: bool = True) -> ModelRow:
    """Insert the model if its version is new; optionally make it the only active one."""
    row = db.scalar(select(ModelRow).where(ModelRow.version == info["version"]))
    if row is None:
        row = ModelRow(version=info["version"], algorithm=info["algorithm"], trained_at=info["trained_at"],
                       metrics=info.get("test_metrics"))
        db.add(row)
    if artifact_dir is not None:
        row.artifact_dir, row.metrics_dir = rel(artifact_dir), rel(metrics_dir or artifact_dir)
    if activate:
        set_active(db, row)
    db.commit()
    return row


def set_active(db: Session, row: ModelRow) -> None:
    for other in db.scalars(select(ModelRow).where(ModelRow.active, ModelRow.id != row.id)):
        other.active = False
    row.active = True


def audit(session_factory: sessionmaker[Session], actor: str, action: str, outcome: str) -> None:
    """Append one audit row. Failing to audit is logged but must not mask the action's own result."""
    try:
        with session_factory() as db:
            db.add(AuditLog(actor=actor[:64], action=action[:64], outcome=outcome[:16]))
            db.commit()
    except Exception:  # noqa: BLE001
        log.exception("could not write audit log")


def record_prediction(session_factory: sessionmaker[Session], hasher: UrlHasher, url: str, result: dict) -> None:
    """Store one prediction. Runs after the response is sent; failures are logged, never raised."""
    try:
        with session_factory() as db:
            model = db.scalar(select(ModelRow).where(ModelRow.version == result["model_version"]))
            if model is None:
                return
            db.add(Prediction(url_hash=hasher(url), label=result["label"], probability=result["probability"],
                              model_id=model.id, snapshot=FeatureSnapshot(features=result["features"])))
            db.commit()
    except Exception:  # noqa: BLE001 - persistence must not affect the prediction response
        log.exception("could not store prediction")
