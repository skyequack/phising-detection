"""Frozen-model false-positive rate on crawled real legitimate links.

Usage: python -m ml.crawl_baseline
Writes reports/metrics/crawl_baseline.json. Not used for tuning.
"""

from __future__ import annotations

import json

import pandas as pd

from backend.service import METRICS, ROOT, Predictor
from ml import features as F


def run() -> dict:
    p, df = Predictor.load(), pd.read_parquet(ROOT / "data" / "external" / "crawl" / "links.parquet")
    df["score"] = p._proba(F.extract_matrix(df["url"]).astype(float))
    df["flagged"] = df["score"] >= p.threshold
    bucket = pd.cut(df["rank"], [0, 1_000, 10_000, 100_000, 1_000_000],
                    labels=["1-1k", "1k-10k", "10k-100k", "100k-1M"])
    out = {"model_version": p.version, "threshold": p.threshold, "n": len(df),
           "domains": int(df["domain"].nunique()), "false_positive_rate": float(df["flagged"].mean()),
           "by_rank": {str(k): {"n": int(len(g)), "fpr": round(float(g["flagged"].mean()), 4)}
                       for k, g in df.groupby(bucket, observed=True)},
           "flagged_examples": df[df["flagged"]].sort_values("score", ascending=False)["url"].head(15).tolist()}
    (METRICS / "crawl_baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
