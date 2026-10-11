"""Measure latency, throughput and memory of the served model. Writes reports/metrics/performance.json.

Usage: python -m ml.benchmark [--requests 300] [--extract 5000] [--threads 8]

The API numbers come from a real uvicorn process and real HTTP over loopback (a fresh temporary
SQLite database, admin disabled), so they include the server, JSON encoding, the explanation
and the background database write. They are single-machine numbers, not a capacity claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from backend.service import METRICS, MODELS, ROOT, Predictor
from ml import features as F


def rss_mb(pid: int | None = None) -> float | None:
    """Resident memory of a process in MB (Windows via psapi, Linux via /proc), None if unavailable."""
    pid = pid or os.getpid()
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        k32, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        handle = k32.OpenProcess(0x0410, False, pid)  # QUERY_INFORMATION | VM_READ
        if not handle:
            return None
        c = Counters()
        c.cb = ctypes.sizeof(c)
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
        k32.CloseHandle(handle)
        return round(c.WorkingSetSize / 2**20, 1) if ok else None
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    return None


def server_rss_mb(pid: int) -> float | None:
    """Largest resident size among the process and its children (on Windows the venv python.exe is a
    launcher that starts the real interpreter as a child, and the launcher itself uses only a few MB)."""
    pids = [pid]
    if sys.platform == "win32":
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={int(pid)}').ProcessId"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        pids += [int(x) for x in out if x.isdigit()]
    sizes = [m for m in (rss_mb(p) for p in pids) if m is not None]
    return max(sizes) if sizes else None


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def summary(ms: list[float]) -> dict:
    return {"n": len(ms), "mean_ms": round(statistics.fmean(ms), 2), "p50_ms": pct(ms, 0.50),
            "p95_ms": pct(ms, 0.95), "p99_ms": pct(ms, 0.99), "max_ms": round(max(ms), 2)}


def sample_urls(n: int) -> list[str]:
    """Real URLs from the held-out test split when present, else the sanity set."""
    test = ROOT / "data" / "processed_v2" / "test.parquet"
    if test.exists():
        return pd.read_parquet(test)["url"].sample(n, random_state=0).tolist()
    s = json.loads((ROOT / "tests" / "data" / "sanity_urls.json").read_text(encoding="utf-8"))
    pool = s["legit"] + s["phishing"]
    return (pool * (n // len(pool) + 1))[:n]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def local_measurements(n_extract: int) -> dict:
    urls = sample_urls(n_extract)
    t = time.perf_counter()
    X = F.extract_matrix(urls)
    extract_s = time.perf_counter() - t
    before = rss_mb()
    t = time.perf_counter()
    predictor = Predictor.load()
    load_s = time.perf_counter() - t
    after = rss_mb()
    t = time.perf_counter()
    predictor._proba(X.astype(float))
    score_s = time.perf_counter() - t
    single = []
    for u in urls[:200]:
        t = time.perf_counter()
        predictor.predict(u)
        single.append((time.perf_counter() - t) * 1000)
    model_file = MODELS / "phishing_model.joblib"
    return {
        "model_version": predictor.version, "model_file_mb": round(model_file.stat().st_size / 2**20, 1),
        "model_load_seconds": round(load_s, 2), "rss_mb_before_load": before, "rss_mb_after_load": after,
        "feature_extraction": {"urls": len(urls), "urls_per_second": round(len(urls) / extract_s), "us_per_url": round(extract_s / len(urls) * 1e6, 1)},
        "batch_scoring": {"rows": len(X), "rows_per_second": round(len(X) / score_s)},
        "predict_with_explanation_in_process": summary(single),
    }


def api_measurements(n_requests: int, threads: int) -> dict:
    urls = sample_urls(n_requests)
    port, tmp = free_port(), tempfile.mkdtemp(prefix="bench-")
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{Path(tmp).as_posix()}/bench.db", "URL_HASH_KEY": "bench-key-" + "x" * 24}
    env.pop("ADMIN_TOKEN", None)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=30) as c:
            t0 = time.time()
            while time.time() - t0 < 120:
                try:
                    if c.get("/health").json()["status"] == "ok":
                        break
                except httpx.HTTPError:
                    time.sleep(0.3)
            else:
                raise RuntimeError("server did not start")
            startup_s = round(time.time() - t0, 1)
            rss_idle = server_rss_mb(proc.pid)
            for u in urls[:10]:  # warm-up, not measured
                c.post("/predict", json={"url": u})
            seq, errors = [], 0
            for u in urls:
                t = time.perf_counter()
                r = c.post("/predict", json={"url": u})
                seq.append((time.perf_counter() - t) * 1000)
                errors += r.status_code != 200
            hist = []
            for _ in range(50):
                t = time.perf_counter()
                c.get("/predictions", params={"limit": 20})
                hist.append((time.perf_counter() - t) * 1000)
            rss_after = server_rss_mb(proc.pid)

        def one(u):
            with httpx.Client(base_url=base, timeout=60) as cc:
                t = time.perf_counter()
                r = cc.post("/predict", json={"url": u})
                return (time.perf_counter() - t) * 1000, r.status_code

        t = time.perf_counter()
        with ThreadPoolExecutor(threads) as ex:
            conc = list(ex.map(one, urls[: threads * 25]))
        wall = time.perf_counter() - t
        return {
            "server_startup_seconds": startup_s, "rss_mb_server_idle": rss_idle, "rss_mb_server_after_requests": rss_after,
            "predict_sequential": {**summary(seq), "errors": errors},
            "predictions_history_get": summary(hist),
            "predict_concurrent": {"threads": threads, **summary([m for m, _ in conc]),
                                   "requests_per_second": round(len(conc) / wall, 1),
                                   "errors": sum(code != 200 for _, code in conc)},
        }
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run(n_requests: int = 300, n_extract: int = 5000, threads: int = 8) -> dict:
    out = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "notes": "Loopback HTTP, one uvicorn worker, temporary SQLite database, URLs sampled from the original test split. "
                 "Single-machine numbers; they say nothing about capacity under production load.",
        **local_measurements(n_extract),
        "api": api_measurements(n_requests, threads),
    }
    (METRICS / "performance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--extract", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(run(a.requests, a.extract, a.threads), indent=2))
