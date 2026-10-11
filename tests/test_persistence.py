import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from backend import db as D
from backend import main
from backend.service import Predictor
from conftest import LEGIT, PHISH


@pytest.fixture()
def client(tiny_model_dir):
    engine = D.make_engine("sqlite://")
    factory = D.init_db(engine)
    predictor = Predictor.load(tiny_model_dir, tiny_model_dir)
    with factory() as db:
        D.register_model(db, predictor.model_info())
    main.app.state.predictor, main.app.state.db = predictor, factory
    main.app.state.hasher = D.UrlHasher("test-key")
    yield TestClient(main.app)
    engine.dispose()


def _rows(client):
    with main.app.state.db() as db:
        return db.scalars(select(D.Prediction).order_by(D.Prediction.id)).all()


def test_predict_is_stored_with_snapshot_and_no_raw_url(client):
    url = PHISH[0]
    body = client.post("/predict", json={"url": url}).json()
    (row,) = _rows(client)
    assert row.label == body["label"] and row.probability == pytest.approx(body["probability"])
    assert len(row.url_hash) == 64 and url not in row.url_hash
    with main.app.state.db() as db:
        snap = db.scalar(select(D.FeatureSnapshot))
        everything = " ".join(str(v) for t in ("models", "predictions", "feature_snapshots", "audit_logs")
                              for r in db.execute(text(f"SELECT * FROM {t}")) for v in r)
    assert snap.features == body["features"]
    assert url not in everything and "192.168.4.7" not in everything


def test_hash_ignores_scheme_www_slash_and_case_but_depends_on_key():
    h = D.UrlHasher("k")
    assert h("https://www.Example.com/a") == h("http://example.com/a") == h("  HTTP://EXAMPLE.COM/a  ")
    assert h("https://www.example.com/") == h("http://example.com")  # a lone trailing slash is dropped
    assert h("http://example.com/a/") != h("http://example.com/a")  # a path slash is kept
    assert h("http://example.com/a") != h("http://example.com/b")
    assert h("http://example.com/a") != D.UrlHasher("other")("http://example.com/a")


def test_placeholder_key_gives_ephemeral_key():
    assert D.UrlHasher("change-me")("http://a.com") != D.UrlHasher("change-me")("http://a.com")


def test_predictions_newest_first_and_paged(client):
    for u in LEGIT[:3] + PHISH[:2]:
        client.post("/predict", json={"url": u})
    page = client.get("/predictions", params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 5 and [i["id"] for i in page["items"]] == [5, 4]
    assert page["items"][0]["created_at"].endswith("+00:00")  # explicit UTC, not a naive timestamp
    assert set(page["items"][0]) == {"id", "url_hash", "label", "probability", "model_version", "created_at"}
    assert [i["id"] for i in client.get("/predictions", params={"limit": 2, "offset": 4}).json()["items"]] == [1]


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "x"}])
def test_predictions_rejects_bad_paging(client, params):
    assert client.get("/predictions", params=params).status_code == 422


def test_failed_write_does_not_break_predict(client, caplog):
    def broken():
        raise RuntimeError("disk full")
    main.app.state.db = broken
    r = client.post("/predict", json={"url": LEGIT[0]})
    assert r.status_code == 200 and "disk full" not in r.text


def test_sql_injection_url_is_data_only(client):
    url = "http://a.com/?q=');DROP TABLE predictions;--"
    assert client.post("/predict", json={"url": url}).status_code == 200
    assert client.get("/predictions").json()["total"] == 1


def test_register_model_keeps_one_active(client):
    with main.app.state.db() as db:
        D.register_model(db, {"version": "lr-2099-01-01", "algorithm": "X", "trained_at": "2099", "test_metrics": {"f1": 0.9}})
        rows = db.scalars(select(D.ModelRow)).all()
        assert [r.version for r in rows if r.active] == ["lr-2099-01-01"]


def test_metrics_by_version(client):
    current = client.app.state.predictor.version
    assert "threshold_sweep" in client.get(f"/metrics/{current}").json()
    with main.app.state.db() as db:
        D.register_model(db, {"version": "old-1", "algorithm": "X", "trained_at": "2020", "test_metrics": {"f1": 0.7}})
    assert client.get("/metrics/old-1").json()["test"]["selected_at_chosen_threshold"] == {"f1": 0.7}
    assert client.get("/metrics/does-not-exist").status_code == 404


@pytest.mark.parametrize("version", ["a b", "x;DROP", "a" * 65, "..%2f..%2fetc", "a%00b", "<script>"])
def test_metrics_version_allow_list(client, version):
    r = client.get(f"/metrics/{version}")
    assert r.status_code in (404, 422) and "Traceback" not in r.text
    if r.status_code == 422:
        assert r.json() == {"detail": "Invalid model version"}
