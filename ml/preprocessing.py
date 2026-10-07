"""Phase 3: one sklearn Pipeline per algorithm plus the tuning search spaces (report Table 3.3).

A StandardScaler is part of the pipeline only for LR and SVM, so it is fit on
whatever training data the pipeline is fit on (never on the test set) and is
saved inside the model. All pipelines name the classifier step ``clf``.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED = 42

ALGORITHMS = ("lr", "dt", "rf", "svm", "xgb")
NAMES = {
    "lr": "Logistic Regression",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "svm": "SVM (RBF)",
    "xgb": "XGBoost",
}
SCALED = {"lr", "svm"}
# Tie-break order when models are within tolerance: most interpretable first.
INTERPRETABILITY = ("dt", "lr", "rf", "xgb", "svm")

SEARCH_SPACES: dict[str, dict] = {
    "lr": {"clf__C": loguniform(1e-3, 1e2)},
    "dt": {"clf__max_depth": [5, 10, 15, 20, 30, None], "clf__min_samples_leaf": [1, 2, 5, 10, 20, 50]},
    "rf": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [10, 20, 30, None],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2", 0.5],
    },
    "svm": {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-3, 1e0)},
    "xgb": {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(3, 10),
        "clf__learning_rate": loguniform(0.03, 0.3),
        "clf__subsample": uniform(0.6, 0.4),
    },
}


def build_pipeline(
    algo: str, *, scale_pos_weight: float = 1.0, n_jobs: int = 1, probability: bool = True
) -> Pipeline:
    """Unfitted pipeline. Class imbalance is handled by class weights, not resampling."""
    if algo == "lr":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    elif algo == "dt":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    elif algo == "rf":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=n_jobs, random_state=SEED)
    elif algo == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
        if probability:  # SVC(probability=True) is deprecated; calibrate the decision function instead
            clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3, ensemble=False)
    elif algo == "xgb":
        clf = XGBClassifier(
            tree_method="hist", scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=n_jobs, random_state=SEED,
        )
    else:
        raise ValueError(f"unknown algorithm {algo!r}")
    steps = [("scaler", StandardScaler())] if algo in SCALED else []
    return Pipeline(steps + [("clf", clf)])


def set_params(pipe: Pipeline, params: dict) -> Pipeline:
    """``pipe.set_params`` that also handles the calibrated SVM (``clf__estimator__*``)."""
    if isinstance(pipe.named_steps["clf"], CalibratedClassifierCV):
        params = {k.replace("clf__", "clf__estimator__", 1): v for k, v in params.items()}
    return pipe.set_params(**params)
