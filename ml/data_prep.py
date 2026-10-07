"""Phase 1: clean the raw URL dataset and write stratified train/test splits.

Usage: python -m ml.data_prep [--input CSV] [--out-dir DIR] [--seed N]

Outputs (in --out-dir): train.parquet, test.parquet (columns: url, label),
dropped.csv (every removed row with its reason) and cleaning_report.json.
The raw CSV is never modified. No network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "URL dataset.csv"
OUT_DIR = ROOT / "data" / "processed"
SEED = 42
TEST_SIZE = 0.2
LABELS = {"legitimate": 0, "phishing": 1}  # positive class = phishing

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_HTTP_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_WWW = re.compile(r"^www\.")


def canonicalize(url: str) -> str:
    """Key used only for dedupe and conflict detection, never for features.

    Ignores case, scheme (http/https), a leading ``www.`` and trailing slashes,
    so ``https://www.X.com/`` and ``http://x.com`` collapse to one key.
    """
    key = _HTTP_SCHEME.sub("", url.strip().lower())
    return _WWW.sub("", key).rstrip("/")


def drop_reason(url: str) -> str | None:
    """Return why a URL is unusable, or None if it is kept."""
    if not url:
        return "empty_url"
    if _CONTROL.search(url):
        return "control_chars"
    if not _HTTP_SCHEME.match(url):
        return "bad_scheme"  # httpss://, ftp://, scheme-less
    try:
        host = urlsplit(url).hostname
    except ValueError:  # e.g. 'http://bad[.]com/x' (invalid IPv6 bracket)
        return "unparseable"
    if not host:
        return "no_hostname"  # e.g. 'http://?', 'http://#NAME?'
    return None


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (clean rows, dropped rows with reasons, report fragment)."""
    df = pd.DataFrame(
        {
            "url": raw["url"].astype(str).str.strip(),
            "label": raw["type"].astype(str).str.strip().str.lower().map(LABELS),
        }
    )

    reasons = [
        "bad_label" if pd.isna(lab) else drop_reason(url)
        for url, lab in zip(df["url"], df["label"])
    ]
    df["reason"] = reasons
    dropped = [df[df["reason"].notna()]]
    kept = df[df["reason"].isna()].copy()
    kept["label"] = kept["label"].astype(int)

    # Canonical-form conflicts: same URL modulo scheme/www/slash/case, different
    # labels. Ambiguous ground truth, so the whole group is removed.
    kept["canon"] = kept["url"].map(canonicalize)
    conflict = kept.groupby("canon")["label"].transform("nunique") > 1
    conflict_keys = sorted(kept.loc[conflict, "canon"].unique())
    dropped.append(kept[conflict].assign(reason="label_conflict"))
    kept = kept[~conflict]

    # Remaining canonical duplicates share one label: keep the first occurrence.
    dup = kept.duplicated("canon", keep="first")
    dropped.append(kept[dup].assign(reason="duplicate"))
    kept = kept[~dup]

    dropped_df = pd.concat(dropped)[["url", "label", "reason"]].reset_index(drop=True)
    clean_df = kept[["url", "label"]].reset_index(drop=True)
    report = {
        "rows_in": int(len(raw)),
        "dropped_by_reason": dict(sorted(Counter(dropped_df["reason"]).items())),
        "rows_dropped": int(len(dropped_df)),
        "label_conflict_groups": len(conflict_keys),
        "label_conflict_keys": conflict_keys,
        "rows_clean": int(len(clean_df)),
        "class_counts_clean": _class_counts(clean_df),
    }
    return clean_df, dropped_df, report


def split(
    df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split (shuffled, reproducible for a given seed)."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _class_counts(df: pd.DataFrame) -> dict:
    n = len(df)
    phishing = int((df["label"] == 1).sum())
    return {
        "legitimate": n - phishing,
        "phishing": phishing,
        "phishing_fraction": round(phishing / n, 6) if n else 0.0,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(input_csv: Path = RAW_CSV, out_dir: Path = OUT_DIR, seed: int = SEED) -> dict:
    raw = pd.read_csv(input_csv, dtype=str, keep_default_na=False, encoding="utf-8")
    if not {"url", "type"} <= set(raw.columns):
        raise ValueError(f"expected columns url,type; got {list(raw.columns)}")

    clean_df, dropped_df, report = clean(raw)
    train, test = split(clean_df, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    dropped_df.to_csv(out_dir / "dropped.csv", index=False)

    report.update(
        {
            "input_file": input_csv.name,
            "input_sha256": _sha256(input_csv),
            "seed": seed,
            "test_size": TEST_SIZE,
            "label_encoding": LABELS,
            "canonical_key": "lowercase, scheme/www/trailing-slash removed (dedupe only)",
            "train": {"rows": len(train), **_class_counts(train)},
            "test": {"rows": len(test), **_class_counts(test)},
        }
    )
    (out_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, default=RAW_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    rep = run(args.input, args.out_dir, args.seed)
    print(json.dumps({k: v for k, v in rep.items() if k != "label_conflict_keys"}, indent=2))


if __name__ == "__main__":
    main()
