"""Score the frozen model on the hand-made sanity set (tests/data/sanity_urls.json).

Usage: python -m ml.sanity_check
Writes reports/metrics/sanity_check.json. Not used for tuning; it measures how
far the in-dataset test metrics carry over to URLs from outside the dataset.
"""

from __future__ import annotations

import json

from backend.service import METRICS, ROOT, Predictor

SANITY = ROOT / "tests" / "data" / "sanity_urls.json"


def run() -> dict:
    p, data = Predictor.load(), json.loads(SANITY.read_text(encoding="utf-8"))
    out = {"model_version": p.version, "threshold": p.threshold}
    for kind in ("legit", "phishing"):
        scores = {u: p.predict(u)["probability"] for u in data[kind]}
        flagged = [u for u, s in scores.items() if s >= p.threshold]
        out[kind] = {"n": len(scores), "flagged_phishing": len(flagged),
                     "flagged_urls": flagged if kind == "legit" else None,
                     "missed_urls": [u for u in scores if u not in flagged] if kind == "phishing" else None}
    out["legit_false_positive_rate"] = out["legit"]["flagged_phishing"] / out["legit"]["n"]
    out["phishing_recall"] = out["phishing"]["flagged_phishing"] / out["phishing"]["n"]
    (METRICS / "sanity_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = run()
    print({k: r[k] for k in ("legit_false_positive_rate", "phishing_recall")})
