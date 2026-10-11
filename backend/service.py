"""Model loading and prediction + explanation. No network access; the URL is only parsed."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd

from ml import features as F

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("PHISHING_MODELS_DIR", ROOT / "ml" / "models"))
METRICS = Path(os.environ.get("PHISHING_METRICS_DIR", ROOT / "reports" / "metrics"))

TOP_N = 8


class ModelNotReady(RuntimeError):
    """The frozen model or its metadata has not been produced yet (run ml.train)."""


def _single_threaded(model):
    """Serve with one thread per prediction.

    The forest was trained with ``n_jobs=-1``; kept at serving time, every request fans out across all
    cores for a 60-row batch, which costs about twice the latency and oversubscribes the CPU when
    requests overlap. ``n_jobs`` does not change predictions, and the saved file is not touched.
    """
    clf = getattr(getattr(model, "base", model), "named_steps", {}).get("clf")
    if clf is not None and hasattr(clf, "n_jobs"):
        clf.n_jobs = 1
    return model


@dataclass
class Predictor:
    model: object
    meta: dict
    metrics_dir: Path

    @classmethod
    def load(cls, models: Path = MODELS, metrics: Path = METRICS) -> "Predictor":
        model_path, meta_path = models / "phishing_model.joblib", models / "metadata.json"
        if not (model_path.exists() and meta_path.exists()):
            raise ModelNotReady("phishing_model.joblib / metadata.json not found; run `python -m ml.train`")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["feature_names"] != list(F.FEATURE_NAMES):
            raise ModelNotReady("model feature schema does not match ml.features")
        return cls(_single_threaded(joblib.load(model_path)), meta, metrics)

    @property
    def version(self) -> str:
        return f"{self.meta['algorithm']}-{self.meta['trained_at'][:10]}"

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"]["threshold"])

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, url: str) -> dict:
        """Raises ``ml.features.InvalidURLError`` for empty/oversized input."""
        feats = F.extract_features(url)
        names = list(F.FEATURE_NAMES)
        row = pd.DataFrame([[feats[n] for n in names]], columns=names).astype(float)
        # Model-agnostic explanation against the legitimate-class median profile. For each
        # feature, average two effects: (a) the probability drop when only that feature is
        # reset to the legit median, and (b) the rise when only that feature is set to its
        # real value on an otherwise all-legit row. (b) keeps the bars informative when the
        # model is saturated and no single reset changes the score.
        medians = self.meta["legit_medians"]
        base = pd.DataFrame([[medians[n] for n in names]], columns=names).astype(float)
        drop_out, add_in = [], []
        for n in names:
            v = row.copy()
            v.loc[0, n] = medians[n]
            drop_out.append(v)
            w = base.copy()
            w.loc[0, n] = row.loc[0, n]
            add_in.append(w)
        probs = self._proba(pd.concat([row, base, *drop_out, *add_in], ignore_index=True))
        p, p_base = float(probs[0]), float(probs[1])
        k = len(names)
        deltas = ((probs[0] - probs[2:2 + k]) + (probs[2 + k:] - p_base)) / 2

        contributions = [
            {"feature": n, "group": group_of(n), "value": float(feats[n]),
             "legit_median": float(medians[n]), "contribution": float(d)}
            for n, d in zip(names, deltas)
        ]
        groups = {g: float(sum(c["contribution"] for c in contributions if c["group"] == g))
                  for g in F.FEATURE_GROUPS}
        top = sorted(contributions, key=lambda c: -abs(c["contribution"]))[:TOP_N]
        return {
            "label": "phishing" if p >= self.threshold else "legitimate",
            "probability": p,
            "threshold": self.threshold,
            "model_version": self.version,
            "algorithm": self.meta["algorithm_name"],
            "features": {n: float(feats[n]) for n in names},
            "explanation": {"top_features": top, "group_contributions": groups},
            "url_parts": url_parts(url),
        }

    def model_info(self) -> dict:
        m = self.meta
        return {
            "version": self.version, "algorithm": m["algorithm_name"], "trained_at": m["trained_at"],
            "feature_schema_version": m["feature_schema_version"], "threshold": m["threshold"],
            "hyperparameters": m["hyperparameters"], "train_rows": m["train_rows"],
            "feature_importance": m["feature_importance"], "group_importance": m["group_importance"],
            "test_metrics": m.get("test_metrics"),
            "candidates": [{k: c[k] for k in ("algo", "cv_f1", "cv_ap", "cv_roc_auc")} for c in m["candidates"]],
        }

    def metrics(self) -> dict:
        """Everything the Metrics page shows. Every part is optional: a missing file gives None or an empty list."""
        d = self.metrics_dir
        test = _read_json(d / "test_metrics.json")
        return {
            "test": test,
            "threshold_sweep": _read_csv(d / "threshold_sweep.csv"),
            "comparison": _read_csv(d / "test_comparison.csv"),
            "ablation": _read_csv(d / "ablation.csv"),
            "calibration": {"summary": _read_json(d / "calibration.json"),
                            "reliability": _read_csv(d / "calibration_test.csv")},
            "curves": _read_json(d / "curves.json"),
            "external": _read_json(d / "heldout_compare.json"),
        }


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def group_of(feature: str) -> str:
    return next(g for g, names in F.FEATURE_GROUPS.items() if feature in names)


def url_parts(url: str) -> dict[str, str]:
    """Display-only split of the URL; falls back to empty parts if urllib rejects it."""
    try:
        s = urlsplit(url.strip())
        return {"scheme": s.scheme, "host": s.hostname or "", "path": s.path, "query": s.query, "fragment": s.fragment}
    except ValueError:
        return {"scheme": "", "host": "", "path": "", "query": "", "fragment": ""}
