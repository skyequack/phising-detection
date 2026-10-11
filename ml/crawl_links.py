"""Harvest real same-site links from Tranco front pages (held-out legitimate test set).

Usage: python -m ml.crawl_links [--dry-run] [--domains-per-bucket 125]
Per domain: read robots.txt, then fetch ONLY the front page if allowed, and keep up to
LINKS_PER_DOMAIN random same-site links. Nothing is followed further. Output:
data/external/crawl/links.parquet (domain, rank, url) and crawl_report.json.
--dry-run prints the sampled domains and settings without any network request.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"
OUT = EXT / "crawl"
UA = "phishing-detection-research/0.1 (academic URL-dataset study; front page only; honours robots.txt)"
BUCKETS = [(1, 1_000), (1_001, 10_000), (10_001, 100_000), (100_001, 1_000_000)]
SEED, LINKS_PER_DOMAIN, TIMEOUT, MAX_BYTES, WORKERS, MAX_REDIRECTS = 7, 5, 10.0, 1_000_000, 8, 3


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def same_site_links(base: str, html_text: str, domain: str) -> list[str]:
    """Absolute http(s) links on the same host or a subdomain, fragment-only and self links removed."""
    p = _Links()
    p.feed(html_text)
    out: dict[str, None] = {}
    for h in p.hrefs:
        u = urljoin(base, h.strip())
        s = urlsplit(u)
        host = (s.hostname or "").lower()
        if s.scheme in ("http", "https") and (host == domain or host.endswith("." + domain)):
            if urldefrag(u)[0].rstrip("/") != base.rstrip("/") and len(u) <= 2048:
                out[u] = None
    return list(out)


def sample_domains(per_bucket: int) -> list[tuple[str, int]]:
    with zipfile.ZipFile(EXT / "tranco" / "top-1m.csv.zip") as z, z.open(z.namelist()[0]) as f:
        rows = list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))
    rng = random.Random(SEED)
    return [(rows[i][1], int(rows[i][0])) for lo, hi in BUCKETS
            for i in sorted(rng.sample(range(lo - 1, hi), per_bucket))]


def crawl_one(item: tuple[str, int], client: httpx.Client) -> tuple[list[dict], str]:
    domain, rank = item
    base = f"https://{domain}/"
    try:
        rp = RobotFileParser()
        r = client.get(base + "robots.txt")
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        if not rp.can_fetch(UA, base):
            return [], "robots_disallow"
        with client.stream("GET", base) as resp:
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                return [], f"status_{resp.status_code}_or_not_html"
            body = b"".join(resp.iter_bytes())[:MAX_BYTES].decode(resp.encoding or "utf-8", "replace")
            final = str(resp.url)
        links = same_site_links(final, body, domain)
        random.Random(f"{SEED}-{domain}").shuffle(links)
        return [{"domain": domain, "rank": rank, "url": u} for u in links[:LINKS_PER_DOMAIN]], "ok"
    except Exception as e:  # network errors are expected; record the type only
        return [], type(e).__name__


def run(per_bucket: int, dry_run: bool) -> dict:
    items = sample_domains(per_bucket)
    settings = {"seed": SEED, "domains": len(items), "links_per_domain": LINKS_PER_DOMAIN, "timeout_s": TIMEOUT,
                "max_bytes": MAX_BYTES, "workers": WORKERS, "max_redirects": MAX_REDIRECTS, "user_agent": UA,
                "pages_per_domain": 1, "respects_robots_txt": True}
    if dry_run:
        return {**settings, "first_domains": items[:10]}
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": UA}, timeout=TIMEOUT, follow_redirects=True,
                      max_redirects=MAX_REDIRECTS) as client, ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda it: crawl_one(it, client), items))
    rows = [r for rs, _ in results for r in rs]
    pd.DataFrame(rows).to_parquet(OUT / "links.parquet", index=False)
    outcomes: dict[str, int] = {}
    for _, o in results:
        outcomes[o] = outcomes.get(o, 0) + 1
    report = {**settings, "links": len(rows), "domains_with_links": len({r["domain"] for r in rows}),
              "outcomes": dict(sorted(outcomes.items(), key=lambda kv: -kv[1]))}
    (OUT / "crawl_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--domains-per-bucket", type=int, default=125)
    a = ap.parse_args()
    print(json.dumps(run(a.domains_per_bucket, a.dry_run), indent=2))
