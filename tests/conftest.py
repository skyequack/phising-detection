import json

import joblib
import pytest

from ml import features as F
from ml.preprocessing import build_pipeline

LEGIT = ["https://www.google.com/search?q=cats", "https://github.com/user/repo", "https://en.wikipedia.org/wiki/Cat",
         "https://www.bbc.co.uk/news", "https://docs.python.org/3/library/os.html", "https://stackoverflow.com/questions/1"]
PHISH = ["http://192.168.4.7/secure-login/verify.php?account=1", "http://paypa1-secure-update.com/signin/confirm",
         "http://bank.verify-account.example.xyz/login@evil.com", "http://203.0.113.9/a1b2c3d4e5f6/password",
         "http://secure-login-update-account.info/verify?id=9f8e7d6c5b4a", "http://x.y.z.w.example.ru/confirm/bank"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A small logistic-regression model with valid metadata, so API tests never need the real artifacts."""
    d = tmp_path_factory.mktemp("models")
    X = F.extract_matrix(LEGIT + PHISH)
    y = [0] * len(LEGIT) + [1] * len(PHISH)
    joblib.dump(build_pipeline("lr").fit(X, y), d / "phishing_model.joblib")
    meta = {
        "algorithm": "lr", "algorithm_name": "Logistic Regression", "trained_at": "2026-10-07T00:00:00+00:00",
        "feature_schema_version": F.SCHEMA_VERSION, "feature_names": list(F.FEATURE_NAMES),
        "threshold": {"threshold": 0.5}, "hyperparameters": {}, "train_rows": len(y), "candidates": [],
        "feature_importance": {n: 1 / 30 for n in F.FEATURE_NAMES}, "group_importance": {g: 0.0 for g in F.FEATURE_GROUPS},
        "legit_medians": {k: float(v) for k, v in X[[i == 0 for i in y]].median().items()}, "test_metrics": None,
    }
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d
