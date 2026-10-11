"""Pre-retrain baseline: frozen-model recall on the newest PhishTank URLs.

Usage: python -m ml.phishtank_baseline
Writes reports/metrics/phishtank_heldout_baseline.json. Not used for tuning.
"""

from __future__ import annotations

import json

import pandas as pd

from backend.service import METRICS, ROOT, Predictor
from ml import features as F

HELDOUT = ROOT / "data" / "processed" / "phishtank_heldout.parquet"


def run() -> dict:
    p, df = Predictor.load(), pd.read_parquet(HELDOUT)
    s = p._proba(F.extract_matrix(df["url"]).astype(float))
    df["flagged"] = s >= p.threshold
    month = df["submitted"].dt.strftime("%Y-%m")
    out = {"model_version": p.version, "threshold": p.threshold, "n": len(df),
           "recall": float(df["flagged"].mean()), "mean_score": float(s.mean()),
           "recall_by_month": {k: round(float(v), 4) for k, v in df.groupby(month)["flagged"].mean().items()}}
    (METRICS / "phishtank_heldout_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
