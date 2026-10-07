"""Phase 3a: tune the five algorithms, select one, choose the threshold, freeze artifacts.

Usage: python -m ml.train [--n-iter N] [--svm-cap N] [--quick]

Uses the TRAIN split only. The test split is read exclusively by ml.evaluate,
after this script has written the frozen model (its SHA-256 is stored in
metadata.json and re-checked there). No network access.

Outputs: ml/models/candidate_<algo>.joblib, phishing_model.joblib, metadata.json,
reports/metrics/cv_results.csv, threshold_sweep.csv, chosen_threshold.json,
reports/figures/threshold_sweep.png.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from ml import features as F
from ml.preprocessing import ALGORITHMS, INTERPRETABILITY, NAMES, SEARCH_SPACES, SEED, build_pipeline, set_params

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "ml" / "models"
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"

N_ITER = 8
RF_N_ITER = 4  # Random Forest on ~350k rows costs ~6 min per candidate; halved to keep a full run near an hour
SVM_CAP = 50_000
CV_FOLDS = 5
# Fixed BEFORE any test-set result exists: the threshold maximizes F1 subject to
# a false-negative rate (missed phishing) of at most this value on OOF predictions.
TARGET_FNR = 0.05
SELECTION_TOL = 0.002  # CV F1 within this of the best => prefer the more interpretable model


def _features_hash() -> str:
    return hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()[:8]


def load_features(split: str, processed: Path = PROCESSED) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix for a split, cached per version of ml/features.py."""
    cache = processed / f"features_{_features_hash()}_{split}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = pd.read_parquet(processed / f"{split}.parquet")
        df = F.extract_matrix(raw["url"])
        df["label"] = raw["label"].to_numpy()
        df.to_parquet(cache, index=False)
    return df[list(F.FEATURE_NAMES)], df["label"]


def sweep_thresholds(y: np.ndarray, proba: np.ndarray, grid: np.ndarray | None = None) -> pd.DataFrame:
    grid = np.round(np.arange(0.10, 0.9001, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        pred = proba >= t
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "fnr": 1 - recall_score(y, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, target_fnr: float = TARGET_FNR) -> dict:
    """Max-F1 threshold with FNR <= target; if none qualifies, the lowest-FNR one."""
    ok = sweep[sweep["fnr"] <= target_fnr]
    if len(ok):
        row, rule = ok.loc[ok["f1"].idxmax()], "max_f1_subject_to_fnr"
    else:
        row, rule = sweep.loc[sweep["fnr"].idxmin()], "fallback_min_fnr"
    return {"threshold": float(row["threshold"]), "rule": rule, "target_fnr": target_fnr,
            **{k: float(row[k]) for k in ("precision", "recall", "f1", "fnr")}}


def select_model(rows: list[dict], tol: float = SELECTION_TOL) -> str:
    """Best CV F1; among models within ``tol`` of that F1, the most interpretable.

    ROC-AUC is reported but is not part of the rule: F1 and ROC-AUC can rank
    models differently, which would leave no model within tolerance on both.
    """
    best_f1 = max(r["cv_f1"] for r in rows)
    close = [r["algo"] for r in rows if best_f1 - r["cv_f1"] <= tol]
    return min(close, key=INTERPRETABILITY.index)


def feature_importance(est, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Normalized global importance per feature (sums to 1)."""
    clf = est.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:  # SVM (RBF): permutation importance on a fixed sample
        idx = np.random.RandomState(SEED).choice(len(X), min(5000, len(X)), replace=False)
        imp = permutation_importance(est, X.iloc[idx], y.iloc[idx], scoring="average_precision",
                                     n_repeats=3, random_state=SEED).importances_mean.clip(min=0)
    imp = imp / imp.sum() if imp.sum() > 0 else np.full(len(imp), 1 / len(imp))
    return dict(zip(X.columns, map(float, imp)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(n_iter: int = N_ITER, svm_cap: int = SVM_CAP, quick: bool = False, resume: bool = False,
        processed: Path = PROCESSED, models: Path = MODELS, metrics: Path = METRICS,
        figures: Path = FIGURES) -> dict:
    for d in (models, metrics, figures):
        d.mkdir(parents=True, exist_ok=True)
    X, y = load_features("train", processed)
    if quick:  # smoke test only; never used for reported results
        X, _, y, _ = train_test_split(X, y, train_size=20_000, stratify=y, random_state=SEED)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    spw = float((y == 0).sum() / (y == 1).sum())

    cv_tables, rows, fitted, subsets = [], [], {}, {}
    for algo in ALGORITHMS:
        if resume:  # reuse a finished search (cv_results.csv + candidate_*.joblib), skip tuning
            res = pd.read_csv(metrics / "cv_results.csv").query("algo == @algo and best").iloc[0]
            fitted[algo] = joblib.load(models / f"candidate_{algo}.joblib")
            subsets[algo] = svm_cap if algo == "svm" and len(X) > svm_cap else len(X)
            pipe_params = fitted[algo].get_params()
            prefix = "clf__estimator__" if algo == "svm" else "clf__"
            hp = {k: pipe_params[prefix + k[5:]] for k in SEARCH_SPACES[algo]}
            rows.append({"algo": algo, "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in hp.items()},
                         "cv_ap": float(res["mean_test_ap"]), "cv_f1": float(res["mean_test_f1"]),
                         "cv_roc_auc": float(res["mean_test_roc_auc"]), "train_rows": subsets[algo],
                         "seconds": round(float(res["mean_fit_time"]), 1)})
            continue
        if algo == "svm" and len(X) > svm_cap:  # O(n^2-n^3): tune and fit on a stratified subsample
            Xa, _, ya, _ = train_test_split(X, y, train_size=svm_cap, stratify=y, random_state=SEED)
        else:
            Xa, ya = X, y
        subsets[algo] = len(Xa)
        heavy = algo in ("rf", "xgb")  # these parallelize internally
        base = build_pipeline(algo, scale_pos_weight=spw, n_jobs=-1 if heavy else 1, probability=False)
        t0 = time.time()
        search = RandomizedSearchCV(
            base, SEARCH_SPACES[algo], n_iter=(min(n_iter, RF_N_ITER) if algo == "rf" else n_iter) if not quick else 2, cv=cv,
            scoring={"ap": "average_precision", "f1": "f1", "roc_auc": "roc_auc"}, refit="ap",
            random_state=SEED, n_jobs=1 if heavy else -1, error_score="raise",
        ).fit(Xa, ya)
        best = search.best_index_
        res = pd.DataFrame(search.cv_results_)
        cv_tables.append(res[["params", "mean_test_ap", "std_test_ap", "mean_test_f1", "mean_test_roc_auc",
                              "mean_fit_time"]].assign(algo=algo, best=lambda d: d.index == best))
        final = build_pipeline(algo, scale_pos_weight=spw, n_jobs=-1 if heavy else 1, probability=True)
        set_params(final, search.best_params_).fit(Xa, ya)
        fitted[algo] = final
        joblib.dump(final, models / f"candidate_{algo}.joblib")
        rows.append({
            "algo": algo, "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
            "cv_ap": float(res.loc[best, "mean_test_ap"]), "cv_f1": float(res.loc[best, "mean_test_f1"]),
            "cv_roc_auc": float(res.loc[best, "mean_test_roc_auc"]), "train_rows": len(Xa),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{NAMES[algo]:20s} cv_f1={rows[-1]['cv_f1']:.4f} cv_ap={rows[-1]['cv_ap']:.4f} "
              f"cv_auc={rows[-1]['cv_roc_auc']:.4f} ({rows[-1]['seconds']}s)", flush=True)

    if not resume:
        pd.concat(cv_tables).to_csv(metrics / "cv_results.csv", index=False)
    chosen = select_model(rows)

    Xa, ya = (X, y) if subsets[chosen] == len(X) else train_test_split(
        X, y, train_size=svm_cap, stratify=y, random_state=SEED)[::2]
    oof = cross_val_predict(clone(fitted[chosen]), Xa, ya, cv=cv, method="predict_proba",
                            n_jobs=1 if chosen in ("rf", "xgb") else -1)[:, 1]
    sweep = sweep_thresholds(ya.to_numpy(), oof)
    thr = choose_threshold(sweep)
    sweep.to_csv(metrics / "threshold_sweep.csv", index=False)
    (metrics / "chosen_threshold.json").write_text(json.dumps({"algo": chosen, **thr}, indent=2), encoding="utf-8")
    ax = sweep.set_index("threshold")[["precision", "recall", "f1"]].plot(figsize=(6, 4))
    ax.axvline(thr["threshold"], color="k", ls="--", lw=1)
    ax.set_title(f"{NAMES[chosen]}: out-of-fold threshold sweep")
    ax.figure.savefig(figures / "threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)

    model_path = models / "phishing_model.joblib"
    shutil.copyfile(models / f"candidate_{chosen}.joblib", model_path)
    importance = feature_importance(fitted[chosen], Xa, ya)
    meta = {
        "algorithm": chosen, "algorithm_name": NAMES[chosen],
        "feature_schema_version": F.SCHEMA_VERSION, "features_source_hash": _features_hash(),
        "feature_names": list(F.FEATURE_NAMES), "feature_groups": {g: list(v) for g, v in F.FEATURE_GROUPS.items()},
        "threshold": thr, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "cv_folds": CV_FOLDS, "n_iter": n_iter, "quick": quick, "resumed_from_saved_search": resume,
        "hyperparameters": next(r["params"] for r in rows if r["algo"] == chosen),
        "candidates": rows, "train_rows": len(X), "model_train_rows": len(Xa),
        "feature_importance": importance,
        "group_importance": {g: sum(importance[n] for n in v) for g, v in F.FEATURE_GROUPS.items()},
        "legit_medians": {k: float(v) for k, v in X[y == 0].median().items()},
        "model_sha256": _sha256(model_path), "frozen": True, "test_metrics": None,
    }
    (models / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"selected {NAMES[chosen]}, threshold {thr['threshold']} ({thr['rule']}), model frozen")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-iter", type=int, default=N_ITER)
    p.add_argument("--svm-cap", type=int, default=SVM_CAP)
    p.add_argument("--resume", action="store_true", help="skip tuning; reuse saved cv_results.csv and candidates")
    p.add_argument("--quick", action="store_true", help="smoke test on 20k rows, 2 iterations")
    a = p.parse_args()
    if a.quick:  # keep smoke-test output away from the real artifacts
        tmp = ROOT / "data" / "processed" / "quick"
        run(a.n_iter, a.svm_cap, True, models=tmp / "models", metrics=tmp / "metrics", figures=tmp / "figures")
    else:
        run(a.n_iter, a.svm_cap, resume=a.resume)


if __name__ == "__main__":
    main()
