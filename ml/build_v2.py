"""Build the v2 training set: original data + PhishTank (older 85%) + Tranco legitimate URLs.

Usage: python -m ml.build_v2
Output (data/processed_v2/): train.parquet, test.parquet (columns url, label, source),
build_report.json. test.parquet is the ORIGINAL test split, unchanged, so v1 and v2
stay comparable. Held-out sets (PhishTank newest 15%, OpenPhish, the link crawl, the
Tranco validation sample, tests/data/sanity_urls.json) never enter training and their
canonical keys are removed from it. No network access.

The Tranco rows have no real paths, so one URL per sampled domain is built from
templates (see ``synth_url``). This is a stand-in for real legitimate paths; the crawl
held-out set measures whether it generalises. Template words deliberately include
login/account/verify-style terms and random-looking ids on the legitimate side, so
those cues do not map to phishing by construction.
"""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile

import pandas as pd

from ml.crawl_links import sample_domains
from ml.data_prep import OUT_DIR as V1, ROOT, canonicalize
from ml.external_validation import tranco_sample

OUT = ROOT / "data" / "processed_v2"
SEED = 42
BUCKET_QUOTA = [(1, 1_000, None), (1_001, 10_000, 4_000), (10_001, 100_000, 20_000), (100_001, 1_000_000, 35_000)]

WORDS = """about contact privacy terms help support products services news blog careers pricing docs download search
category events team faq shop cart store login signin signup account settings verify security billing update profile
orders checkout invoice password reset dashboard portal members community forum press investors partners solutions
features resources learn guides tutorials videos gallery photos reviews offers deals sale returns shipping policy
legal cookies sitemap status api developers downloads releases changelog jobs students teachers research publications
topics archive tags authors users view page item article post story events calendar tickets locations stores map""".split()
SLUG_WORDS = """how to best new guide introduction getting started tips review overview update annual report summer
winter spring release notes learn more design data cloud mobile security health travel food music sports""".split()
LANGS = ["en", "en-us", "en-gb", "de", "fr", "es", "it", "pt-br", "ja", "ru", "zh", "nl", "pl"]
SUBS = ["support", "blog", "shop", "docs", "m", "help", "news", "store", "app", "mail", "accounts", "my", "portal"]
EXTS = [".html", ".htm", ".php", ".aspx", ".jsp"]


def _slug(r: random.Random) -> str:
    return "-".join(r.choice(SLUG_WORDS) for _ in range(r.randint(2, 6)))


def _token(r: random.Random) -> str:
    return r.choice([lambda: str(r.randint(1, 9_999_999)), lambda: f"{r.getrandbits(r.choice((24, 32, 48, 64))):x}",
                     lambda: _slug(r) + "-" + f"{r.getrandbits(32):x}"])()


def synth_url(domain: str, r: random.Random) -> str:
    host = f"{r.choice(SUBS)}.{domain}" if r.random() < 0.10 else domain
    if r.random() < 0.30 and host == domain:  # bare host, optionally with a trailing slash
        return f"https://{host}" + r.choice(["", "/"])
    w1, w2 = r.choice(WORDS), r.choice(WORDS)
    path = r.choice([
        lambda: f"/{w1}", lambda: f"/{w1}/", lambda: f"/{w1}/{w2}", lambda: f"/{w1}/{w2}/{_slug(r)}",
        lambda: f"/{r.choice(LANGS)}/{w1}/", lambda: f"/{r.choice(LANGS)}/{w1}/{w2}",
        lambda: f"/{w1}{r.choice(EXTS)}", lambda: f"/{w1}/{w2}{r.choice(EXTS)}",
        lambda: f"/{w1}/{_token(r)}", lambda: f"/p/{_token(r)}", lambda: f"/@{_slug(r).split('-')[0]}/{_slug(r)}-{_token(r)}",
        lambda: f"/blog/{r.randint(2012, 2026)}/{r.randint(1, 12):02d}/{_slug(r)}",
        lambda: f"/{w1}/{r.randint(2012, 2026)}/{_slug(r)}/",
    ])()
    q = ""
    roll = r.random()
    if roll < 0.20:
        q = r.choice([f"?id={r.randint(1, 99999)}", f"?page={r.randint(1, 40)}&sort={r.choice(WORDS)}",
                      f"?q={r.choice(SLUG_WORDS)}+{r.choice(SLUG_WORDS)}", f"?utm_source={r.choice(WORDS)}&utm_medium=email",
                      f"?ref={r.choice(WORDS)}&lang={r.choice(LANGS)}", f"?token={_token(r)}"])
    elif roll < 0.25:
        q = f"#{r.choice(WORDS)}"
    return f"{r.choice(['https'] * 6 + ['http'])}://{host}{path}{q}"


def tranco_rows(exclude: set[str]) -> list[tuple[str, int]]:
    with zipfile.ZipFile(ROOT / "data" / "external" / "tranco" / "top-1m.csv.zip") as z, z.open(z.namelist()[0]) as f:
        rows = [(r[1], int(r[0])) for r in csv.reader(io.TextIOWrapper(f, encoding="utf-8"))]
    rng, picked = random.Random(SEED), []
    for lo, hi, quota in BUCKET_QUOTA:
        pool = [x for x in rows[lo - 1:hi] if x[0] not in exclude]
        picked += pool if quota is None else rng.sample(pool, min(quota, len(pool)))
    return picked


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_parquet(V1 / "train.parquet"), pd.read_parquet(V1 / "test.parquet")
    pt = pd.read_parquet(V1 / "phishtank_train.parquet")
    held = pd.read_parquet(V1 / "phishtank_heldout.parquet")
    crawl = pd.read_parquet(ROOT / "data" / "external" / "crawl" / "links.parquet")
    openphish = (ROOT / "data" / "external" / "openphish" / "feed.txt").read_text(encoding="utf-8").split()
    sanity = json.loads((ROOT / "tests" / "data" / "sanity_urls.json").read_text(encoding="utf-8"))

    # Domains reserved for validation/test must not feed training.
    reserved_domains = {u.split("//")[1] for v in tranco_sample().values() for u in v} | {d for d, _ in sample_domains(125)}
    blocked = {canonicalize(u) for u in [*held["url"], *crawl["url"], *openphish, *sanity["legit"], *sanity["phishing"]]}
    blocked |= set(te["url"].map(canonicalize))

    picked = tranco_rows(reserved_domains)
    synth = pd.DataFrame({"url": [synth_url(d, random.Random(f"{SEED}-{d}")) for d, _ in picked], "label": 0})
    synth["source"] = "tranco_synth"
    train = pd.concat([tr.assign(source="original"), pt[["url", "label"]].assign(source="phishtank"), synth],
                      ignore_index=True)
    train["canon"] = train["url"].map(canonicalize)
    before = train["source"].value_counts().to_dict()
    train = train[~train["canon"].isin(blocked)]
    # Same canonical URL with different labels => ambiguous, drop the group. Then dedupe.
    conflict = train.groupby("canon")["label"].transform("nunique") > 1
    train = train[~conflict].drop_duplicates("canon").drop(columns="canon")
    train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)

    te = te.assign(source="original")
    train.to_parquet(OUT / "train.parquet", index=False)
    te.to_parquet(OUT / "test.parquet", index=False)
    summary = lambda d: {"rows": len(d), "phishing_fraction": round(float(d["label"].mean()), 4),
                         "by_source": {k: int(v) for k, v in d["source"].value_counts().items()}}
    report = {"seed": SEED, "rows_before_filtering": {k: int(v) for k, v in before.items()},
              "dropped_conflicts": int(conflict.sum()), "reserved_domains": len(reserved_domains),
              "tranco_domains_sampled": len(picked), "train": summary(train), "test": summary(te),
              "synth_examples": synth["url"].sample(25, random_state=1).tolist()}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
