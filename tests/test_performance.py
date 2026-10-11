"""Performance regression guards. The budgets are deliberately loose (5x or more headroom over the numbers
measured by `python -m ml.benchmark` on the development machine), so they catch real regressions, not noise.
The figures that go in the report come from reports/metrics/performance.json, not from these assertions."""

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend import db as D
from backend import main
from backend.service import MODELS, METRICS, Predictor
from ml import features as F
from ml.benchmark import pct, summary

ROOT = Path(__file__).resolve().parents[1]
SANITY = json.loads((ROOT / "tests" / "data" / "sanity_urls.json").read_text(encoding="utf-8"))
URLS = (SANITY["legit"] + SANITY["phishing"]) * 6
real_model = pytest.mark.skipif(not (MODELS / "phishing_model.joblib").exists(), reason="trained model not present")


def test_percentile_helpers():
    assert pct(list(range(1, 101)), 0.5) == 51 and pct([7], 0.99) == 7
    s = summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5 and s["max_ms"] == 100.0 and s["p50_ms"] == 3.0 and s["mean_ms"] == 22.0


def test_feature_extraction_throughput():
    urls = URLS * 20  # 1,440 URLs
    t = time.perf_counter()
    F.extract_matrix(urls)
    rate = len(urls) / (time.perf_counter() - t)
    assert rate > 3_000, f"{rate:.0f} URLs/s (measured about 20,000)"


@real_model
def test_single_prediction_latency_budget():
    p = Predictor.load()
    for u in URLS[:5]:
        p.predict(u)  # warm-up
    ms = []
    for u in URLS[:60]:
        t = time.perf_counter()
        p.predict(u)
        ms.append((time.perf_counter() - t) * 1000)
    s = summary(ms)
    assert s["p50_ms"] < 150 and s["p95_ms"] < 400, s  # measured about 26 ms and 30 ms


@real_model
def test_served_model_is_single_threaded():
    """With n_jobs=-1 each request fans out over every core: about twice the latency, and requests fight for the CPU."""
    clf = Predictor.load().model.base.named_steps["clf"]
    assert clf.n_jobs == 1


@real_model
def test_model_load_time_and_memory_budget():
    code = ("import time,json,sys; sys.path.insert(0,'.')\n"
            "from ml.benchmark import rss_mb\n"
            "from backend.service import Predictor\n"
            "t=time.perf_counter(); Predictor.load(); print(json.dumps({'load_s': time.perf_counter()-t, 'rss_mb': rss_mb()}))")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["load_s"] < 20, r  # measured about 2 s
    if r["rss_mb"] is not None:
        assert r["rss_mb"] < 2_500, r  # measured about 1,340 MB; the Random Forest is large


@real_model
def test_concurrent_requests_succeed_and_none_are_lost(tmp_path):
    engine = D.make_engine(f"sqlite:///{(tmp_path / 'perf.db').as_posix()}")
    factory = D.init_db(engine)
    predictor = Predictor.load()
    with factory() as db:
        D.register_model(db, predictor.model_info(), MODELS, METRICS)
    state = main.app.state
    state.predictor, state.db, state.hasher = predictor, factory, D.UrlHasher("k")
    client = TestClient(main.app)
    n_threads, per_thread = 8, 10

    def work(i):
        return [client.post("/predict", json={"url": u}).status_code for u in URLS[i * per_thread:(i + 1) * per_thread]]

    t = time.perf_counter()
    with ThreadPoolExecutor(n_threads) as ex:
        codes = [c for batch in ex.map(work, range(n_threads)) for c in batch]
    wall = time.perf_counter() - t
    assert codes == [200] * (n_threads * per_thread)
    assert wall < 60, f"{wall:.1f}s for 80 requests"  # measured a few seconds
    with factory() as db:
        assert db.scalar(select(func.count(D.Prediction.id))) == len(codes), "background writes were lost"
    engine.dispose()


def test_published_performance_report_is_consistent():
    path = METRICS / "performance.json"
    if not path.exists():
        pytest.skip("run python -m ml.benchmark first")
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["api"]["predict_sequential"]["errors"] == 0 and r["api"]["predict_concurrent"]["errors"] == 0
    seq = r["api"]["predict_sequential"]
    assert seq["p50_ms"] <= seq["p95_ms"] <= seq["p99_ms"] <= seq["max_ms"]
    assert r["model_version"] and r["model_file_mb"] > 0 and r["api"]["rss_mb_server_after_requests"] > 100
