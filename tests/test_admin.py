import json
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend import admin
from backend import db as D
from backend import main
from backend.service import Predictor

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _variant(src, dst, trained_at):
    shutil.copytree(src, dst)
    meta = json.loads((dst / "metadata.json").read_text(encoding="utf-8"))
    meta["trained_at"] = trained_at
    (dst / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return dst


@pytest.fixture()
def env(tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(admin, "RUNS_DIR", tmp_path / "runs")
    engine = D.make_engine("sqlite://")
    factory = D.init_db(engine)
    predictor = Predictor.load(tiny_model_dir, tiny_model_dir)
    with factory() as db:
        D.register_model(db, predictor.model_info(), tiny_model_dir, tiny_model_dir)
    state = main.app.state
    state.predictor, state.db, state.hasher = predictor, factory, D.UrlHasher("k")
    state.train_job = {"status": "idle", "lock": threading.Lock()}
    state.train_runner = lambda run_dir: None
    yield type("Env", (), {"client": TestClient(main.app), "factory": factory, "model_dir": tiny_model_dir,
                           "tmp": tmp_path, "predictor": predictor})
    engine.dispose()


def _audit(env):
    with env.factory() as db:
        return [(r.actor, r.action, r.outcome) for r in db.scalars(select(D.AuditLog).order_by(D.AuditLog.id))]


def _wait_done(client, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get("/admin/train/status", headers=AUTH).json()
        if st["status"] != "running":
            return st
        time.sleep(0.05)
    raise AssertionError("training did not finish")


ADMIN_ROUTES = [("get", "/admin/models"), ("get", "/admin/audit-logs"), ("get", "/admin/train/status"),
                ("post", "/admin/train"), ("post", "/admin/models/x/activate")]


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN},
                                     {"Authorization": f"Basic {TOKEN}"}, {"Authorization": "Bearer "}])
def test_admin_requires_valid_token(env, method, path, headers):
    r = getattr(env.client, method)(path, headers=headers)
    assert r.status_code == 401 and r.json() == {"detail": "Unauthorized"}
    assert r.headers["www-authenticate"] == "Bearer"
    assert any(a[2] == "denied" for a in _audit(env))


@pytest.mark.parametrize("configured", ["", "change-me", "short-token", "x" * 15])
def test_admin_disabled_without_a_strong_token(env, monkeypatch, configured):
    monkeypatch.setenv("ADMIN_TOKEN", configured)
    r = env.client.get("/admin/models", headers={"Authorization": f"Bearer {configured}"})
    assert r.status_code == 401


def test_admin_disabled_when_token_unset(env, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN")
    assert env.client.get("/admin/models", headers=AUTH).status_code == 401


def test_denied_attempts_are_audited_with_route_template_not_raw_path(env):
    env.client.post("/admin/models/some-version/activate", headers={"Authorization": "Bearer bad"})
    assert ("anonymous", "POST /admin/models/{version}/activate", "denied") in _audit(env)


def test_public_endpoints_stay_public(env):
    assert env.client.get("/models").status_code == 200
    assert env.client.get("/predictions").status_code == 200


def test_list_models_and_audit_logs(env):
    models = env.client.get("/admin/models", headers=AUTH).json()
    assert models == [{"version": env.predictor.version, "algorithm": "Logistic Regression",
                       "trained_at": "2026-10-07T00:00:00+00:00", "active": True, "activatable": True}]
    env.client.post("/admin/models/nope/activate", headers=AUTH)
    logs = env.client.get("/admin/audit-logs", headers=AUTH, params={"limit": 1}).json()
    assert logs["items"][0]["action"] == "activate:nope" and logs["total"] == 1
    assert env.client.get("/admin/audit-logs", headers=AUTH, params={"limit": 0}).status_code == 422


def test_activate_switches_the_served_model(env):
    other = _variant(env.model_dir, env.tmp / "other", "2099-01-01T00:00:00+00:00")
    other_p = Predictor.load(other, other)
    with env.factory() as db:
        D.register_model(db, other_p.model_info(), other, other, activate=False)
    r = env.client.post(f"/admin/models/{other_p.version}/activate", headers=AUTH)
    assert r.status_code == 200 and r.json() == {"active": other_p.version}
    assert env.client.get("/models").json()["version"] == other_p.version
    active = [m["version"] for m in env.client.get("/admin/models", headers=AUTH).json() if m["active"]]
    assert active == [other_p.version]
    assert ("admin", f"activate:{other_p.version}", "ok") in _audit(env)


def test_activate_validation_and_failures_leak_nothing(env):
    before = env.client.get("/models").json()["version"]
    assert env.client.post("/admin/models/a%3Bb/activate", headers=AUTH).status_code == 422
    assert env.client.post("/admin/models/" + "a" * 65 + "/activate", headers=AUTH).status_code == 422
    assert env.client.post("/admin/models/unknown/activate", headers=AUTH).status_code == 404
    gone = _variant(env.model_dir, env.tmp / "gone", "2088-01-01T00:00:00+00:00")
    gp = Predictor.load(gone, gone)
    with env.factory() as db:
        D.register_model(db, gp.model_info(), gone, gone, activate=False)
    shutil.rmtree(gone)
    r = env.client.post(f"/admin/models/{gp.version}/activate", headers=AUTH)
    assert r.status_code == 409 and r.json() == {"detail": "Model artifacts are not available"}
    assert str(gone) not in r.text and "Traceback" not in r.text
    assert env.client.get("/models").json()["version"] == before
    with env.factory() as db:  # a registered row without artifacts cannot be activated either
        D.register_model(db, {"version": "no-files", "algorithm": "X", "trained_at": "2020"}, activate=False)
    assert env.client.post("/admin/models/no-files/activate", headers=AUTH).status_code == 409
    assert ("admin", "activate:unknown", "not_found") in _audit(env)


def _good_runner(date):
    def run(run_dir):
        models, metrics, _ = admin._paths(run_dir)
        shutil.rmtree(models)
        _variant(main.app.state.predictor_src, models, date)
    return run


def test_train_registers_an_inactive_model_and_does_not_change_the_served_one(env):
    main.app.state.predictor_src = env.model_dir
    main.app.state.train_runner = _good_runner("2077-05-05T00:00:00+00:00")
    served = env.client.get("/models").json()["version"]
    r = env.client.post("/admin/train", headers=AUTH)
    assert r.status_code == 202 and r.json()["status"] == "running"
    st = _wait_done(env.client)
    assert st["status"] == "succeeded" and st["message"] == "lr-2077-05-05"
    models = {m["version"]: m for m in env.client.get("/admin/models", headers=AUTH).json()}
    assert models["lr-2077-05-05"]["active"] is False and models["lr-2077-05-05"]["activatable"] is True
    assert env.client.get("/models").json()["version"] == served
    assert env.client.post("/admin/models/lr-2077-05-05/activate", headers=AUTH).status_code == 200
    assert [a for a in _audit(env) if a[1].startswith("train")] == [("admin", "train_start", "ok"),
                                                                    ("admin", "train_finished", "ok")]


def test_only_one_training_run_at_a_time(env):
    gate = threading.Event()
    main.app.state.train_runner = lambda run_dir: gate.wait(10)
    assert env.client.post("/admin/train", headers=AUTH).status_code == 202
    second = env.client.post("/admin/train", headers=AUTH)
    assert second.status_code == 409 and second.json() == {"detail": "A training run is already in progress"}
    gate.set()
    _wait_done(env.client)
    assert ("admin", "train_start", "busy") in _audit(env)


def test_failed_training_is_reported_generically_and_releases_the_lock(env):
    def boom(run_dir):
        raise RuntimeError("secret /internal/path exploded")
    main.app.state.train_runner = boom
    assert env.client.post("/admin/train", headers=AUTH).status_code == 202
    st = _wait_done(env.client)
    assert st["status"] == "failed" and st["message"] == "training failed"
    assert "secret" not in json.dumps(st)
    assert ("admin", "train_finished", "failed") in _audit(env)
    assert len(env.client.get("/admin/models", headers=AUTH).json()) == 1
    main.app.state.train_runner = lambda run_dir: None  # leaves empty folders: artifacts missing, still a clean failure
    assert env.client.post("/admin/train", headers=AUTH).status_code == 202
    assert _wait_done(env.client)["status"] == "failed"


def test_training_that_reproduces_an_existing_version_is_refused(env):
    main.app.state.predictor_src = env.model_dir
    main.app.state.train_runner = _good_runner("2026-10-07T00:00:00+00:00")  # same version as the served model
    env.client.post("/admin/train", headers=AUTH)
    st = _wait_done(env.client)
    assert st["status"] == "failed" and "already registered" in st["message"]


def test_startup_prefers_the_model_an_admin_activated(env):
    other = _variant(env.model_dir, env.tmp / "other2", "2066-06-06T00:00:00+00:00")
    op = Predictor.load(other, other)
    with env.factory() as db:
        D.register_model(db, op.model_info(), other, other, activate=True)
    assert main.load_predictor(env.factory).version == op.version
    shutil.rmtree(other)  # artifacts gone: fall back to the default folder instead of failing to start
    fallback = main.load_predictor(env.factory)
    assert fallback.version != op.version
