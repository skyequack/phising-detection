"""Security sweep across every route: hostile input, type confusion, error leakage, CORS, path traversal,
auth bypass attempts, and static checks that the code has no raw SQL, no outbound HTTP and no shell."""

import json
import re
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from backend import db as D
from backend import main
from backend.service import Predictor

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "s" * 32
LEAK_MARKERS = ["Traceback", "sqlalchemy", "sqlite", 'File "', "Internal Server Error", "/Users/", "C:\\", "joblib",
                "pydantic", "sklearn"]

HOSTILE_URLS = [
    "http://a.com/?q=' OR 1=1;--", "http://a.com/'; DROP TABLE predictions;--", "' OR ''='",
    "http://a.com/<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>", "file:///etc/passwd", "ftp://a.com/x",
    "http://a.com/../../../../etc/passwd", "http://a.com/%2e%2e/%2e%2e/etc/passwd", "..\\..\\windows\\system32",
    "http://a.com/\x00null", "http://a.com/\r\nSet-Cookie: x=1", "http://[::1", "http://a.com:99999/",
    "http://" + "a." * 1500 + "com", "%s%s%n", "{{7*7}}${7*7}", "http://xn--/", "http://\u202eevil.com",
    "\uff28\uff34\uff34\uff30://\uff41.com", "http://a.com/" + "%" * 3000, "http://\u7528\u6237@\u4f8b\u3048.jp/\u30d1\u30b9",
    "http://a.com/\ud800", "http://" + "\u0000" * 5, " \t\n http://a.com \n", "http://a@b@c@d.com", "http://:@:/",
]


@pytest.fixture()
def api(tiny_model_dir, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    engine = D.make_engine("sqlite://")
    factory = D.init_db(engine)
    predictor = Predictor.load(tiny_model_dir, tiny_model_dir)
    with factory() as db:
        D.register_model(db, predictor.model_info(), tiny_model_dir, tiny_model_dir)
    state = main.app.state
    state.predictor, state.db, state.hasher = predictor, factory, D.UrlHasher("k")
    state.train_job = {"status": "idle", "lock": threading.Lock()}
    state.train_runner = lambda run_dir: None
    yield type("Api", (), {"client": TestClient(main.app, raise_server_exceptions=False), "factory": factory})
    engine.dispose()


def post_url(client, url):
    """Send the URL as raw JSON bytes so lone surrogates (which httpx refuses to encode) reach the server."""
    return client.post("/predict", content=json.dumps({"url": url}).encode(), headers={"Content-Type": "application/json"})


def assert_clean(resp):
    assert resp.status_code < 500, (resp.status_code, resp.text[:200])
    body = resp.text
    for marker in LEAK_MARKERS:
        assert marker not in body, f"{marker!r} leaked in {body[:200]!r}"


# ---------- hostile URLs through /predict ----------

@pytest.mark.parametrize("url", HOSTILE_URLS, ids=lambda u: repr(u[:30]))
def test_hostile_urls_never_crash_or_leak(api, url):
    r = post_url(api.client, url)
    assert r.status_code in (200, 422)
    assert r.headers["content-type"].startswith("application/json")
    assert_clean(r)
    if r.status_code == 422:
        assert r.json() == {"detail": "Invalid URL"} or "detail" in r.json()


def test_hostile_urls_leave_the_database_intact_and_unreflected(api):
    ok = 0
    for url in HOSTILE_URLS:
        ok += post_url(api.client, url).status_code == 200
    with api.factory() as db:
        assert db.scalar(select(D.Prediction.id).limit(1)) is not None
        assert len(db.scalars(select(D.Prediction)).all()) == ok
        names = {r[0] for r in db.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        assert {"models", "predictions", "feature_snapshots", "audit_logs"} <= names
        dump = " ".join(str(v) for t in names if not t.startswith("sqlite") for r in db.execute(text(f"SELECT * FROM {t}")) for v in r)
    assert "DROP TABLE" not in dump and "<script" not in dump and "etc/passwd" not in dump
    hist = api.client.get("/predictions").text
    assert "<" not in hist and "script" not in hist


def test_response_url_parts_are_json_strings_only(api):
    r = api.client.post("/predict", json={"url": "http://a.com/<script>alert(1)</script>?x=<b>"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert isinstance(json.loads(r.text)["url_parts"]["path"], str)


# ---------- malformed bodies and type confusion ----------

SENTINEL = "SENTINEL-9f3c1"


@pytest.mark.parametrize("body", [
    {"url": None}, {"url": 5}, {"url": True}, {"url": [SENTINEL]}, {"url": {"$ne": SENTINEL}}, {"u": SENTINEL},
    [], [SENTINEL], SENTINEL, 5, None, {"url": SENTINEL * 700},
])
def test_wrong_shapes_are_rejected_without_echoing_input(api, body):
    r = api.client.post("/predict", json=body)
    assert r.status_code == 422
    assert_clean(r)
    assert SENTINEL not in r.text, "validation errors must not echo the submitted value"


@pytest.mark.parametrize("kwargs", [
    {"content": b"{not json"}, {"content": b""}, {"content": b"\xff\xfe\x00"}, {"content": b'{"url": "a"'},
    {"content": "url=http://a.com", "headers": {"Content-Type": "application/x-www-form-urlencoded"}},
    {"content": "http://a.com", "headers": {"Content-Type": "text/plain"}},
    {"content": b"[" * 20000 + b"]" * 20000, "headers": {"Content-Type": "application/json"}},
])
def test_malformed_requests_get_4xx(api, kwargs):
    r = api.client.post("/predict", **kwargs)
    assert 400 <= r.status_code < 500
    assert_clean(r)


def test_huge_body_is_refused_quickly(api):
    r = api.client.post("/predict", json={"url": "http://a.com/" + "x" * 3_000_000})
    assert r.status_code == 422
    assert_clean(r)


def test_extra_fields_are_ignored_not_executed(api):
    r = api.client.post("/predict", json={"url": "http://a.com/", "model_version": "../../x", "threshold": -1})
    assert r.status_code == 200 and r.json()["threshold"] == 0.5


# ---------- routes, methods, paging ----------

@pytest.mark.parametrize("method,path", [
    ("get", "/predict"), ("put", "/predict"), ("delete", "/predictions"), ("post", "/models"), ("post", "/metrics"),
    ("get", "/admin"), ("get", "/nope"), ("get", "/.env"), ("get", "/../etc/passwd"), ("get", "/docs/../.env"),
    ("get", "/admin//models"), ("get", "/admin/models/"), ("post", "/admin/models//activate"),
])
def test_unknown_or_wrong_method_is_a_clean_4xx(api, method, path):
    r = getattr(api.client, method)(path)
    assert r.status_code in (401, 404, 405, 307, 308), r.status_code
    assert_clean(r)


@pytest.mark.parametrize("params", [
    {"limit": "1; DROP TABLE x"}, {"limit": 99999999999999999999}, {"offset": 99999999999999999999},
    {"offset": 10**30}, {"limit": -1}, {"limit": 1.5}, {"offset": "NaN"}, {"limit": ""},
])
def test_paging_parameters_cannot_break_queries(api, params):
    r = api.client.get("/predictions", params=params)
    assert r.status_code in (200, 422)
    assert_clean(r)


# ---------- path traversal in version parameters ----------

TRAVERSAL = ["..%2f..%2fetc%2fpasswd", "%2e%2e%2f%2e%2e%2fsecret", "..\\..\\x", "%252e%252e%252f", "a%00b",
             "C%3A%5CWindows%5Cwin.ini", "%2Fetc%2Fpasswd", "....//....//x", "a/b", "%00", "<script>", "a%20b", "a" * 65]


@pytest.mark.parametrize("version", TRAVERSAL)
def test_version_parameters_reject_traversal(api, version):
    r = api.client.get(f"/metrics/{version}")
    assert r.status_code in (404, 422)
    assert_clean(r)
    r = api.client.post(f"/admin/models/{version}/activate", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code in (404, 422)
    assert_clean(r)


# ---------- authentication bypass attempts ----------

BYPASS = [
    {"headers": {"X-Admin-Token": TOKEN}}, {"headers": {"X-Forwarded-For": "127.0.0.1"}}, {"headers": {"X-Real-IP": "127.0.0.1"}},
    {"params": {"token": TOKEN}}, {"params": {"access_token": TOKEN, "ADMIN_TOKEN": TOKEN}},
    {"cookies": {"ADMIN_TOKEN": TOKEN, "token": TOKEN}}, {"headers": {"Authorization": f"bearer {TOKEN} extra"}},
    {"headers": {"Authorization": f"Bearer {TOKEN}x"}}, {"headers": {"Authorization": f"Bearer {TOKEN[:-1]}"}},
    {"headers": {"Authorization": f"Bearer {'A' * 100000}"}}, {"headers": {"Authorization": "Bearer null"}},
    {"headers": {"Authorization": "Bearer \u00e9\u00e9\u00e9"}.copy()},
]


@pytest.mark.parametrize("kwargs", BYPASS, ids=range(len(BYPASS)))
@pytest.mark.parametrize("method,path", [("get", "/admin/models"), ("post", "/admin/train"), ("get", "/admin/audit-logs")])
def test_admin_cannot_be_bypassed(api, kwargs, method, path):
    try:
        r = getattr(api.client, method)(path, **kwargs)
    except UnicodeEncodeError:  # a non-latin-1 header cannot even be sent
        return
    assert r.status_code == 401 and r.json() == {"detail": "Unauthorized"}
    assert api.client.app.state.train_job["status"] == "idle"


def test_admin_error_bodies_do_not_say_why(api):
    wrong = api.client.get("/admin/models", headers={"Authorization": "Bearer nope"})
    missing = api.client.get("/admin/models")
    assert wrong.json() == missing.json() == {"detail": "Unauthorized"}


def test_admin_stays_closed_with_a_weak_token_even_if_guessed(api, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin")
    assert api.client.get("/admin/models", headers={"Authorization": "Bearer admin"}).status_code == 401


# ---------- CORS and headers ----------

def test_cors_does_not_reflect_unknown_origins(api):
    r = api.client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    ok = api.client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_preflight_cannot_authorize_the_authorization_header(api):
    r = api.client.options("/admin/models", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET",
                                                     "Access-Control-Request-Headers": "Authorization"})
    assert r.status_code == 400
    r = api.client.options("/predict", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "DELETE"})
    assert r.status_code == 400


def test_responses_carry_nosniff(api):
    for path in ("/health", "/models", "/predictions", "/nope"):
        assert api.client.get(path).headers.get("x-content-type-options") == "nosniff"


# ---------- static checks on the source ----------

def _sources(*names):
    return {n: (ROOT / n).read_text(encoding="utf-8") for n in names}


def test_no_raw_sql_in_the_backend():
    for name, src in _sources("backend/main.py", "backend/db.py", "backend/admin.py", "backend/service.py").items():
        assert not re.search(r"\btext\(", src), f"{name}: raw text() SQL"
        assert not re.search(r"\.execute\(\s*f?[\"']", src), f"{name}: string passed to execute()"
        assert not re.search(r"f[\"'][^\"']*\b(SELECT|INSERT|UPDATE|DELETE)\b[^\"']*\{", src, re.I), f"{name}: SQL in an f-string"


def test_prediction_path_has_no_network_or_shell_access():
    banned = re.compile(r"^\s*(import|from)\s+(requests|httpx|aiohttp|urllib\.request|http\.client|socket|ftplib|smtplib|"
                        r"subprocess|os\.system)\b", re.M)
    for name, src in _sources("backend/main.py", "backend/service.py", "backend/db.py", "ml/features.py").items():
        assert not banned.search(src), f"{name} imports a network or process module"
        assert not re.search(r"\b(eval|exec)\(|pickle\.loads|os\.system|shell\s*=\s*True", src), name


def test_the_only_process_spawn_uses_a_fixed_argument_list():
    src = (ROOT / "backend" / "admin.py").read_text(encoding="utf-8")
    assert src.count("subprocess.run(") == 1 and "shell" not in src.replace("shell-", "")
    signature = src[src.index("def default_train_runner"):].split(":", 1)[0]
    assert "request" not in signature  # the runner takes only a server-chosen directory


def test_secrets_are_not_in_the_repository_files():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ADMIN_TOKEN=change-me" in env_example and "URL_HASH_KEY=change-me" in env_example
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for name, src in _sources("backend/main.py", "backend/admin.py", "backend/db.py").items():
        assert not re.search(r"(token|key|secret)\s*=\s*[\"'][A-Za-z0-9]{12,}[\"']", src, re.I), name
