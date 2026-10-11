import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.service import ModelNotReady, Predictor
from conftest import PHISH
from ml import features as F


@pytest.fixture(scope="module")
def client(tiny_model_dir):
    main.app.state.predictor = Predictor.load(tiny_model_dir, tiny_model_dir)
    # Skip the lifespan loader (it would look for the real, possibly absent, model).
    return TestClient(main.app)


def test_predict_shape_and_label(client):
    r = client.post("/predict", json={"url": PHISH[0]})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "phishing" and 0 <= body["probability"] <= 1
    assert len(body["features"]) == 30
    assert len(body["explanation"]["top_features"]) == 8
    assert set(body["explanation"]["group_contributions"]) == set(F.FEATURE_GROUPS)
    assert body["url_parts"]["host"] == "192.168.4.7"


def test_legit_url(client):
    assert client.post("/predict", json={"url": "https://github.com/user/repo"}).json()["label"] == "legitimate"


@pytest.mark.parametrize("url", ["", "x" * 5000])
def test_rejects_bad_input(client, url):
    assert client.post("/predict", json={"url": url}).status_code == 422


def test_whitespace_only_is_422_without_detail_leak(client):
    r = client.post("/predict", json={"url": "   "})
    assert r.status_code == 422 and r.json() == {"detail": "Invalid URL"}


def test_missing_body_is_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_injection_strings_are_just_text(client):
    r = client.post("/predict", json={"url": "http://a.com/?q=' OR 1=1;--<script>alert(1)</script>"})
    assert r.status_code == 200


def test_models_and_metrics(client):
    assert client.get("/models").json()["algorithm"] == "Logistic Regression"
    m = client.get("/metrics").json()
    assert set(m) == {"test", "threshold_sweep", "comparison", "ablation", "calibration", "curves", "external"}
    assert m["threshold_sweep"] == [] and m["curves"] is None  # the tiny test model has no report files


def test_model_missing_raises(tmp_path):
    with pytest.raises(ModelNotReady):
        Predictor.load(tmp_path, tmp_path)
