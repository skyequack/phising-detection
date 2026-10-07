import json
import socket

import pytest

from backend.service import MODELS, Predictor
from ml import features as F

pytestmark = pytest.mark.skipif(
    not (MODELS / "phishing_model.joblib").exists(), reason="run python -m ml.train first"
)
SANITY = json.loads((MODELS.parents[1] / "tests" / "data" / "sanity_urls.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictor():
    return Predictor.load()


def test_artifact_matches_feature_schema(predictor):
    m = predictor.meta
    assert m["feature_names"] == list(F.FEATURE_NAMES) and len(m["feature_names"]) == 30
    assert m["feature_schema_version"] == F.SCHEMA_VERSION
    assert m["frozen"] and 0.1 <= predictor.threshold <= 0.9


def test_prediction_shape(predictor):
    r = predictor.predict(SANITY["legit"][0])
    assert r["label"] in ("legitimate", "phishing") and 0 <= r["probability"] <= 1
    assert len(r["features"]) == 30


@pytest.mark.parametrize("a,b", [("https://www.github.com/x/y", "http://github.com/x/y")])
def test_scheme_and_www_do_not_affect_the_score(predictor, a, b):
    assert predictor.predict(a)["probability"] == pytest.approx(predictor.predict(b)["probability"])


def test_obvious_phishing_samples_are_flagged(predictor):
    flagged = sum(predictor.predict(u)["label"] == "phishing" for u in SANITY["phishing"])
    assert flagged / len(SANITY["phishing"]) >= 0.8


def test_prediction_makes_no_network_calls(predictor, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    predictor.predict("http://unroutable.invalid/login")
