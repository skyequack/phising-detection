"""Probability calibration fitted on out-of-fold training predictions (never on the test set)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

BINS = np.linspace(0, 1, 11)


class PrefitCalibrated:
    """A fitted model plus a monotone map from its raw score to a calibrated probability."""

    def __init__(self, base, calibrator, method: str):
        self.base, self.calibrator, self.method = base, calibrator, method

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return self.calibrator.predict_proba(_logit(raw))[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.calibrate(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(raw: np.ndarray, y: np.ndarray, method: str):
    if method == "isotonic":
        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
    if method == "platt":
        return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw), y)
    raise ValueError(method)


def reliability_table(y, p, bins: np.ndarray = BINS) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    b = pd.cut(p, bins, include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y, "bin": b}).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), actual_phishing=("y", "mean"))
    t = t.reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


def calibration_metrics(y, p) -> dict:
    t = reliability_table(y, p)
    ece = float((t["n"] * (t["mean_pred"] - t["actual_phishing"]).abs()).sum() / t["n"].sum())
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"ece": ece, "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}


def choose_method(raw: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[str, dict]:
    """Pick isotonic vs Platt by ECE on held-out halves of the out-of-fold data."""
    scores = {}
    for method in ("isotonic", "platt"):
        eces = []
        for fit_idx, ev_idx in StratifiedKFold(2, shuffle=True, random_state=seed).split(raw, y):
            cal = PrefitCalibrated(None, fit_calibrator(raw[fit_idx], y[fit_idx], method), method)
            eces.append(calibration_metrics(y[ev_idx], cal.calibrate(raw[ev_idx]))["ece"])
        scores[method] = float(np.mean(eces))
    return min(scores, key=scores.get), scores
