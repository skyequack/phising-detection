"""Score the frozen model on URLs from outside the training dataset.

Usage: python -m ml.external_validation
Inputs (downloaded by hand, git-ignored): data/external/tranco/top-1m.csv.zip,
data/external/openphish/feed.txt. Writes reports/metrics/external_validation.json.
Not used for tuning. Tranco holds bare domains only (no paths), so the legitimate
side is host-level; compare it with the in-dataset bare-host rows, not the full test set.
"""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile

import numpy as np

from backend.service import METRICS, ROOT, Predictor

EXT = ROOT / "data" / "external"
BUCKETS = [(1, 1_000), (1_001, 10_000), (10_001, 100_000), (100_001, 1_000_000)]
PER_BUCKET, SEED = 500, 42


def tranco_sample() -> dict[str, list[str]]:
    with zipfile.ZipFile(EXT / "tranco" / "top-1m.csv.zip") as z, z.open(z.namelist()[0]) as f:
        rows = list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))
    rng = random.Random(SEED)
    return {f"{lo}-{hi}": [f"https://{r[1]}" for r in rng.sample(rows[lo - 1:hi], PER_BUCKET)]
            for lo, hi in BUCKETS}


def rates(p: Predictor, urls: list[str]) -> dict:
    s = np.array([p.predict(u)["probability"] for u in urls])
    return {"n": len(urls), "flagged": int((s >= p.threshold).sum()),
            "flag_rate": float((s >= p.threshold).mean()), "mean_score": float(s.mean())}


def run() -> dict:
    p = Predictor.load()
    phish = [u.strip() for u in (EXT / "openphish" / "feed.txt").read_text(encoding="utf-8").splitlines() if u.strip()]
    out = {"model_version": p.version, "threshold": p.threshold,
           "legitimate_tranco_by_rank": {k: rates(p, v) for k, v in tranco_sample().items()},
           "phishing_openphish": rates(p, phish)}
    out["phishing_recall"] = out["phishing_openphish"]["flag_rate"]
    (METRICS / "external_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
