"""Clean the PhishTank dump and split it by submission time.

Usage: python -m ml.phishtank_prep [--heldout-frac 0.15]
Input: data/external/phishtank/online-valid.csv.gz (git-ignored, never edited).
Outputs in data/processed/: phishtank_train.parquet (older URLs),
phishtank_heldout.parquet (the newest URLs, never used for training or tuning),
phishtank_report.json. Columns: url, label (always 1), submitted.
No network access.

URLs are cleaned with the same rules as ml.data_prep. A URL whose canonical key
already appears in the original train/test split is dropped (it is already
covered, or it conflicts with a legitimate label). Keys in the OpenPhish feed are
dropped too, so the OpenPhish set stays independent of training.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ml.data_prep import OUT_DIR, ROOT, _sha256, canonicalize, drop_reason

PT_FILE = ROOT / "data" / "external" / "phishtank" / "online-valid.csv.gz"
OP_FILE = ROOT / "data" / "external" / "openphish" / "feed.txt"
HELDOUT_FRAC = 0.15


def prepare(raw: pd.DataFrame, known: set[str], openphish: set[str], heldout_frac: float):
    df = pd.DataFrame({
        "url": raw["url"].astype(str).map(html.unescape).str.strip(),
        "submitted": pd.to_datetime(raw["submission_time"], utc=True),
    })
    df["reason"] = df["url"].map(drop_reason)
    df["canon"] = df["url"].map(canonicalize)
    df.loc[df["reason"].isna() & df["canon"].isin(known), "reason"] = "in_original_dataset"
    df.loc[df["reason"].isna() & df["canon"].isin(openphish), "reason"] = "in_openphish"
    # Oldest submission wins among canonical duplicates.
    df = df.sort_values("submitted", kind="stable").reset_index(drop=True)
    ok = df["reason"].isna()
    df.loc[ok & df.duplicated("canon", keep="first"), "reason"] = "duplicate"

    kept = df[df["reason"].isna()][["url", "submitted"]].reset_index(drop=True)
    kept["label"] = 1
    cut = int(len(kept) * (1 - heldout_frac))
    return kept.iloc[:cut].copy(), kept.iloc[cut:].copy(), df


def run(heldout_frac: float = HELDOUT_FRAC) -> dict:
    raw = pd.read_csv(PT_FILE, dtype=str, keep_default_na=False, encoding="utf-8")
    known = set()
    for name in ("train", "test"):
        known |= set(pd.read_parquet(OUT_DIR / f"{name}.parquet")["url"].map(canonicalize))
    openphish = {canonicalize(u.strip()) for u in OP_FILE.read_text(encoding="utf-8").splitlines() if u.strip()}

    train, held, df = prepare(raw, known, openphish, heldout_frac)
    train.to_parquet(OUT_DIR / "phishtank_train.parquet", index=False)
    held.to_parquet(OUT_DIR / "phishtank_heldout.parquet", index=False)
    span = lambda d: [d["submitted"].min().isoformat(), d["submitted"].max().isoformat()]
    report = {
        "input_file": PT_FILE.name, "input_sha256": _sha256(PT_FILE), "rows_in": len(raw),
        "dropped_by_reason": dict(sorted(Counter(df["reason"].dropna()).items())),
        "heldout_frac": heldout_frac,
        "train": {"rows": len(train), "submitted_span": span(train),
                  "by_year": {str(k): int(v) for k, v in train["submitted"].dt.year.value_counts().sort_index().items()}},
        "heldout": {"rows": len(held), "submitted_span": span(held),
                    "by_year": {str(k): int(v) for k, v in held["submitted"].dt.year.value_counts().sort_index().items()}},
    }
    (OUT_DIR / "phishtank_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--heldout-frac", type=float, default=HELDOUT_FRAC)
    print(json.dumps(run(p.parse_args().heldout_frac), indent=2))
