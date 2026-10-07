import numpy as np
import pytest

from ml.calibration import PrefitCalibrated, calibration_metrics, choose_method, fit_calibrator, reliability_table


class _Base:
    def __init__(self, p):
        self.p = np.asarray(p)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p[: len(X)], self.p[: len(X)]])


@pytest.fixture(scope="module")
def overconfident():
    rng = np.random.RandomState(0)
    true_p = rng.uniform(0, 1, 6000)
    y = (rng.uniform(size=6000) < true_p).astype(int)
    raw = np.clip(true_p ** 0.5, 0, 1)  # inflated scores: raw 0.5 really means ~0.25
    return raw, y


def test_perfectly_calibrated_scores_have_low_ece():
    rng = np.random.RandomState(1)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert calibration_metrics(y, p)["ece"] < 0.02


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_reduces_ece_and_is_monotone(overconfident, method):
    raw, y = overconfident
    cal = PrefitCalibrated(None, fit_calibrator(raw, y, method), method)
    out = cal.calibrate(raw)
    assert calibration_metrics(y, out)["ece"] < calibration_metrics(y, raw)["ece"]
    grid = np.linspace(0, 1, 101)
    assert (np.diff(cal.calibrate(grid)) >= -1e-12).all()
    assert ((0 <= out) & (out <= 1)).all()


def test_choose_method_returns_a_known_method(overconfident):
    method, scores = choose_method(*overconfident)
    assert method in ("isotonic", "platt") and set(scores) == {"isotonic", "platt"}


def test_wrapper_predict_proba_shape_and_rows_sum_to_one(overconfident):
    raw, y = overconfident
    w = PrefitCalibrated(_Base(raw), fit_calibrator(raw, y, "isotonic"), "isotonic")
    P = w.predict_proba(np.zeros((50, 3)))
    assert P.shape == (50, 2) and np.allclose(P.sum(axis=1), 1)


def test_reliability_table_counts_add_up(overconfident):
    raw, y = overconfident
    assert reliability_table(y, raw)["n"].sum() == len(y)
