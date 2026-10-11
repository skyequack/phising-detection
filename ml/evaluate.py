"""Phase 3b: evaluate on the held-out test split, then run the feature-group ablation.

Usage: python -m ml.evaluate [--no-ablation]   (skip the slow ablation, keep the saved ablation.csv)

The test split is read only here, after ml.train froze the model: the stored
SHA-256 of phishing_model.joblib must match, or this script refuses to run.
Outputs: reports/metrics/test_metrics.json, curves.json, test_comparison.csv, ablation.csv, calibration.json, calibration_test.csv;
reports/figures/confusion_matrices.png, calibration.png, roc_curves.png, pr_curves.png, ablation.png.
The selected model's test metrics are also written back into metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from ml.calibration import calibration_metrics, reliability_table
from ml.preprocessing import ALGORITHMS, NAMES, build_pipeline
from ml.train import FIGURES, METRICS, MODELS, PROCESSED, SVM_CAP, SEED, _sha256, load_features


def compute_metrics(y, proba, threshold: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _score(est, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(X)


def ablation(meta: dict, X, y, Xt, yt) -> pd.DataFrame:
    """Retrain the selected model (fixed hyperparameters) with groups removed.

    Drop-one removes each group alone; cumulative removes groups from most to
    least important until one group is left. Scores use the test split.
    """
    algo, groups = meta["algorithm"], meta["feature_groups"]
    spw = float((y == 0).sum() / (y == 1).sum())
    if algo == "svm" and len(X) > SVM_CAP:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=SVM_CAP, stratify=y, random_state=SEED)

    def fit_eval(label, kind, dropped):
        cols = [c for g, v in groups.items() if g not in dropped for c in v]
        est = build_pipeline(algo, scale_pos_weight=spw, n_jobs=-1, probability=False)
        est.set_params(**meta["hyperparameters"]).fit(X[cols], y)
        s = _score(est, Xt[cols])
        pred = est.predict(Xt[cols])
        return {"setting": label, "kind": kind, "removed": ",".join(dropped), "n_features": len(cols),
                "roc_auc": roc_auc_score(yt, s), "pr_auc": average_precision_score(yt, s),
                "f1": f1_score(yt, pred), "recall": recall_score(yt, pred)}

    rows = [fit_eval("all features", "baseline", [])]
    for g in groups:
        rows.append(fit_eval(f"drop {g}", "drop_one", [g]))
    order = sorted(groups, key=lambda g: -meta["group_importance"][g])
    for i in range(1, len(order)):
        rows.append(fit_eval(f"drop top {i}", "cumulative", order[:i]))
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for m in ("roc_auc", "pr_auc", "f1", "recall"):
        df[f"d_{m}"] = df[m] - base[m]
    return df


def curve_data(results: dict, y, points: int = 150) -> dict:
    """ROC and precision-recall points per model (thinned evenly) for the dashboard; AUCs are exact."""
    def thin(xs, ys):
        idx = np.unique(np.linspace(0, len(xs) - 1, min(points, len(xs))).astype(int))
        return [[round(float(xs[i]), 5), round(float(ys[i]), 5)] for i in idx]

    out = {}
    for algo, p in results.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        out[algo] = {"name": NAMES[algo], "roc_auc": float(roc_auc_score(y, p)),
                     "pr_auc": float(average_precision_score(y, p)),
                     "roc": thin(fpr, tpr), "pr": thin(rec, prec)}
    return out


def _plot_curves(results: dict, y, figures: Path) -> None:
    for kind, fn, xl, yl in (("roc", roc_curve, "False positive rate", "True positive rate"),
                             ("pr", precision_recall_curve, "Recall", "Precision")):
        fig, ax = plt.subplots(figsize=(5, 4))
        for algo, p in results.items():
            a, b, _ = fn(y, p)
            ax.plot(a, b, label=NAMES[algo]) if kind == "roc" else ax.plot(b, a, label=NAMES[algo])
        ax.set(xlabel=xl, ylabel=yl, title=f"{kind.upper()} curves (test)")
        ax.legend(fontsize=7)
        fig.savefig(figures / f"{kind}_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run(processed: Path = PROCESSED, models: Path = MODELS, metrics: Path = METRICS, figures: Path = FIGURES,
        ablate: bool = True) -> dict:
    meta = json.loads((models / "metadata.json").read_text(encoding="utf-8"))
    assert meta["frozen"] and _sha256(models / "phishing_model.joblib") == meta["model_sha256"], \
        "model changed after freezing; re-run ml.train before evaluating"
    figures.mkdir(parents=True, exist_ok=True)
    X, y = load_features("train", processed)
    Xt, yt = load_features("test", processed)
    assert list(Xt.columns) == meta["feature_names"]

    chosen, thr = meta["algorithm"], meta["threshold"]["threshold"]
    probas, table = {}, {}
    for algo in ALGORITHMS:
        est = joblib.load(models / f"candidate_{algo}.joblib")
        probas[algo] = _score(est, Xt)
        table[algo] = compute_metrics(yt, probas[algo], 0.5)
    final = joblib.load(models / "phishing_model.joblib")  # calibrated wrapper (or plain pipeline)
    final_p = _score(final, Xt)
    tuned = compute_metrics(yt, final_p, thr)

    # Calibration of the selected model on the test split: raw candidate vs frozen final model.
    calib = {"raw": calibration_metrics(yt, probas[chosen]), "final": calibration_metrics(yt, final_p),
             "method": meta.get("calibration", {}).get("method")}
    rel = {"raw": reliability_table(yt, probas[chosen]), "final": reliability_table(yt, final_p)}
    pd.concat([t.assign(model=k) for k, t in rel.items()]).to_csv(metrics / "calibration_test.csv", index=False)
    (metrics / "calibration.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for k, t in rel.items():
        ax.plot(t["mean_pred"], t["actual_phishing"], marker="o", label=f"{k} (ECE {calib[k]['ece']:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Actual phishing share", title=f"Reliability ({NAMES[chosen]}, test)")
    ax.legend(fontsize=7)
    fig.savefig(figures / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    out = {"test_rows": len(yt), "selected": chosen, "all_models_at_0.5": table,
           "all_models_note": "uncalibrated candidates at 0.5", "selected_at_chosen_threshold": tuned,
           "calibration": calib}
    (metrics / "test_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    flat = [{"model": NAMES[a], **{k: v for k, v in m.items() if k != "confusion_matrix"}} for a, m in table.items()]
    flat.append({"model": f"{NAMES[chosen]} @ tuned threshold",
                 **{k: v for k, v in tuned.items() if k != "confusion_matrix"}})
    pd.DataFrame(flat).to_csv(metrics / "test_comparison.csv", index=False)

    fig, axes = plt.subplots(1, len(ALGORITHMS) + 1, figsize=(3 * (len(ALGORITHMS) + 1), 3))
    panels = [(NAMES[a], table[a]) for a in ALGORITHMS] + [(f"{NAMES[chosen]} @{thr}", tuned)]
    for ax, (title, m) in zip(axes, panels):
        c = m["confusion_matrix"]
        ax.imshow([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], cmap="Blues")
        for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [c["tn"], c["fp"], c["fn"], c["tp"]]):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        ax.set(title=title, xticks=[0, 1], yticks=[0, 1], xticklabels=["legit", "phish"], yticklabels=["legit", "phish"])
        ax.title.set_fontsize(8)
    fig.savefig(figures / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _plot_curves(probas, yt, figures)
    (metrics / "curves.json").write_text(json.dumps(curve_data(probas, yt)), encoding="utf-8")

    if ablate:
        abl = ablation(meta, X, y, Xt, yt)
        abl.to_csv(metrics / "ablation.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        sub = abl[abl["kind"] != "baseline"]
        ax.barh(sub["setting"], sub["d_pr_auc"])
        ax.set(xlabel="Change in PR-AUC vs all features", title=f"Feature-group ablation ({NAMES[chosen]})")
        fig.savefig(figures / "ablation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    meta["test_metrics"] = tuned
    (models / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(flat).round(4).to_string(index=False))
    print(json.dumps(calib, indent=1))
    if ablate:
        print(abl.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-ablation", action="store_true")
    for name, default in (("processed", PROCESSED), ("models", MODELS), ("metrics", METRICS), ("figures", FIGURES)):
        ap.add_argument(f"--{name}", type=Path, default=default)
    a = ap.parse_args()
    run(a.processed, a.models, a.metrics, a.figures, ablate=not a.no_ablation)
