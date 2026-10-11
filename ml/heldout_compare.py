"""Score two frozen models on every held-out set that never entered training.

Usage: python -m ml.heldout_compare [--a ml/models] [--b ml/models_v2] [--out reports/metrics_v2/heldout_compare.json]
Sets: PhishTank newest 15% (phishing), OpenPhish feed (phishing), crawled same-site links
(legitimate), Tranco validation sample (legitimate, bare domains), sanity set (both).
Each model uses its own frozen threshold. Not used for tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backend.service import ROOT, Predictor
from ml import features as F
from ml.external_validation import EXT, tranco_sample


def sets() -> dict[str, tuple[list[str], int]]:
    sanity = json.loads((ROOT / "tests" / "data" / "sanity_urls.json").read_text(encoding="utf-8"))
    out = {
        "phishtank_newest_15pct": (pd.read_parquet(ROOT / "data/processed/phishtank_heldout.parquet")["url"].tolist(), 1),
        "openphish_feed": ([u for u in (EXT / "openphish" / "feed.txt").read_text(encoding="utf-8").split()], 1),
        "crawl_links": (pd.read_parquet(EXT / "crawl" / "links.parquet")["url"].tolist(), 0),
        "sanity_phishing": (sanity["phishing"], 1), "sanity_legit": (sanity["legit"], 0),
    }
    for bucket, urls in tranco_sample().items():
        out[f"tranco_{bucket}"] = (urls, 0)
    return out


def score(model_dir: Path, data: dict) -> dict:
    p = Predictor.load(model_dir, ROOT / "reports" / "metrics")
    res = {"version": p.version, "threshold": p.threshold}
    for name, (urls, label) in data.items():
        flagged = float((p._proba(F.extract_matrix(urls).astype(float)) >= p.threshold).mean())
        res[name] = {"n": len(urls), "recall" if label else "false_positive_rate": round(flagged, 4)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=ROOT / "ml" / "models")
    ap.add_argument("--b", type=Path, default=ROOT / "ml" / "models_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "metrics_v2" / "heldout_compare.json")
    a = ap.parse_args()
    data = sets()
    out = {"v1": score(a.a, data), "v2": score(a.b, data)}
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k in data:
        m1, m2 = out["v1"][k], out["v2"][k]
        key = next(x for x in m1 if x != "n")
        print(f"{k:28s} n={m1['n']:6d} {key:20s} v1={m1[key]:.3f}  v2={m2[key]:.3f}")
