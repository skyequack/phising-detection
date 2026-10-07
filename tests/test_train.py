import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml.evaluate import compute_metrics
from ml.preprocessing import ALGORITHMS, SCALED, SEARCH_SPACES, build_pipeline, set_params
from ml.train import choose_threshold, select_model, sweep_thresholds


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(300, 8, weights=[0.8], random_state=0)
    return pd.DataFrame(X), y


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_every_pipeline_fits_and_predicts_probabilities(algo, data):
    X, y = data
    est = build_pipeline(algo, scale_pos_weight=4.0).fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert p.shape == (300,) and ((0 <= p) & (p <= 1)).all()
    assert ("scaler" in est.named_steps) == (algo in SCALED)
    assert all(k.startswith("clf__") for k in SEARCH_SPACES[algo])


def test_svm_params_remap_for_calibrated_model(data):
    X, y = data
    est = set_params(build_pipeline("svm"), {"clf__C": 2.0, "clf__gamma": 0.1}).fit(X, y)
    assert est.named_steps["clf"].estimator.C == 2.0


def test_scaler_is_fit_on_training_data_only(data):
    X, y = data
    est = build_pipeline("lr").fit(X.iloc[:100], y[:100])
    assert est.named_steps["scaler"].n_samples_seen_ == 100


def test_sweep_and_threshold_rule():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.6, 50), np.linspace(0.4, 1.0, 50)])
    sweep = sweep_thresholds(y, p)
    assert sweep["threshold"].min() == 0.10 and sweep["threshold"].max() == 0.90
    c = choose_threshold(sweep, target_fnr=0.10)
    assert c["fnr"] <= 0.10 and c["rule"] == "max_f1_subject_to_fnr"
    ok = sweep[sweep["fnr"] <= 0.10]
    assert c["f1"] == ok["f1"].max()
    # impossible target falls back to the lowest-FNR threshold
    assert choose_threshold(sweep, target_fnr=-1)["rule"] == "fallback_min_fnr"


def test_select_model_prefers_interpretable_within_tolerance():
    rows = [
        {"algo": "xgb", "cv_f1": 0.970, "cv_roc_auc": 0.995},
        {"algo": "dt", "cv_f1": 0.969, "cv_roc_auc": 0.994},
        {"algo": "rf", "cv_f1": 0.900, "cv_roc_auc": 0.990},
    ]
    assert select_model(rows) == "dt"
    rows[1]["cv_f1"] = 0.90
    assert select_model(rows) == "xgb"
    # F1 and ROC-AUC disagree on the best model: still returns one (F1 decides)
    split = [{"algo": "svm", "cv_f1": 0.974, "cv_roc_auc": 0.995}, {"algo": "xgb", "cv_f1": 0.971, "cv_roc_auc": 0.997}]
    assert select_model(split) == "svm"


def test_compute_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.2])
    m = compute_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == 0.75 and m["recall"] == 0.75 and m["fnr"] == 0.25
    assert m["specificity"] == 0.75 and m["precision"] == 0.75
