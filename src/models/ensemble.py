"""
Final stacking ensemble – combines XGBoost-ES, LightGBM, and Ridge/Logistic
base learners with a Ridge (regression) / LogisticRegression (classification)
meta-learner.

Outputs
-------
- models/final_ensemble.joblib
- reports/final_metrics.json
- figures/final_metrics.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.feature_pipeline import FrequencyEncoder  # noqa: F401
from src.models.baseline import (
    CLF_TARGET,
    REG_TARGET,
    VIOLATION_THRESHOLD_US,
    build_feature_transformer,
    derive_targets,
    load_train_ready,
    regression_metrics,
    classification_metrics,
    time_split,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = ROOT / "figures"

SEED = 42


# ─────────────────────────────────────────────────────────────────
# Ensemble wrapper
# ─────────────────────────────────────────────────────────────────

class StackingEnsemble:
    """Simple 2-level stacking ensemble.

    Level-0: list of fitted base models (must expose ``.predict()`` and
             optionally ``.predict_proba()``).
    Level-1: meta-learner trained on stacked OOF predictions.
    """

    def __init__(self, base_models: list, meta_model, task: str = "regression"):
        self.base_models = base_models
        self.meta_model = meta_model
        self.task = task          # "regression" | "classification"

    # ── helpers ──────────────────────────────────────────────────
    def _base_preds(self, X: np.ndarray) -> np.ndarray:
        """Stack base-learner predictions as columns."""
        cols = []
        for m in self.base_models:
            if self.task == "classification" and hasattr(m, "predict_proba"):
                cols.append(m.predict_proba(X)[:, 1])
            else:
                cols.append(m.predict(X))
        return np.column_stack(cols)

    # ── public API ───────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self._base_preds(X)
        if self.task == "regression":
            return self.meta_model.predict(Z)
        else:
            return self.meta_model.predict(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) probability matrix for classification."""
        Z = self._base_preds(X)
        return self.meta_model.predict_proba(Z)


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

def _train_base_learners(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    y_train_clf: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_clf: np.ndarray,
) -> Dict[str, Any]:
    """Train 3 base learners for both regression and classification."""
    import lightgbm as lgb
    import xgboost as xgb

    models: Dict[str, Any] = {}

    # ── 1. XGBoost regression ────────────────────────
    print("  [1/6] XGBoost regressor …")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=0,
        early_stopping_rounds=30, eval_metric="mae",
    )
    xgb_reg.fit(X_train, y_train_reg,
                eval_set=[(X_val, y_val_reg)], verbose=False)
    models["xgb_reg"] = xgb_reg

    # ── 2. LightGBM regressor ────────────────────────
    print("  [2/6] LightGBM regressor …")
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb_reg.fit(X_train, y_train_reg,
                eval_set=[(X_val, y_val_reg)],
                callbacks=[lgb.early_stopping(30, verbose=False)])
    models["lgb_reg"] = lgb_reg

    # ── 3. Ridge regressor ───────────────────────────
    print("  [3/6] Ridge regressor …")
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_train, y_train_reg)
    models["ridge_reg"] = ridge

    # ── 4. XGBoost classifier ────────────────────────
    print("  [4/6] XGBoost classifier …")
    spw = (y_train_clf == 0).sum() / max((y_train_clf == 1).sum(), 1)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=0,
        early_stopping_rounds=30, eval_metric="logloss",
        use_label_encoder=False,
    )
    xgb_clf.fit(X_train, y_train_clf,
                eval_set=[(X_val, y_val_clf)], verbose=False)
    models["xgb_clf"] = xgb_clf

    # ── 5. LightGBM classifier ──────────────────────
    print("  [5/6] LightGBM classifier …")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb_clf.fit(X_train, y_train_clf,
                eval_set=[(X_val, y_val_clf)],
                callbacks=[lgb.early_stopping(30, verbose=False)])
    models["lgb_clf"] = lgb_clf

    # ── 6. Logistic regression ───────────────────────
    print("  [6/6] Logistic regression …")
    lr_clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=SEED,
    )
    lr_clf.fit(X_train, y_train_clf)
    models["lr_clf"] = lr_clf

    return models


def _build_stacking_ensemble(
    base_models: Dict[str, Any],
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    y_val_clf: np.ndarray,
) -> Dict[str, StackingEnsemble]:
    """Fit meta-learners on validation predictions (held-out from base training)."""

    # ── Regression meta-learner (Ridge) ─────────────────────
    reg_bases = [base_models["xgb_reg"], base_models["lgb_reg"], base_models["ridge_reg"]]
    Z_reg = np.column_stack([m.predict(X_val) for m in reg_bases])
    meta_reg = Ridge(alpha=1.0, random_state=SEED)
    meta_reg.fit(Z_reg, y_val_reg)
    ens_reg = StackingEnsemble(reg_bases, meta_reg, task="regression")

    # ── Classification meta-learner (LogisticRegression) ────
    clf_bases = [base_models["xgb_clf"], base_models["lgb_clf"], base_models["lr_clf"]]
    Z_clf = np.column_stack([
        m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_val)
        for m in clf_bases
    ])
    meta_clf = LogisticRegression(max_iter=1000, random_state=SEED)
    meta_clf.fit(Z_clf, y_val_clf)
    ens_clf = StackingEnsemble(clf_bases, meta_clf, task="classification")

    return {"reg": ens_reg, "clf": ens_clf}


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def _evaluate(
    ensembles: Dict[str, StackingEnsemble],
    base_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_clf: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate ensemble + individual base learners on test set."""
    metrics: Dict[str, Any] = {}

    # ── Ensemble regression ──────────────────────────
    pred_reg = ensembles["reg"].predict(X_test)
    metrics["ensemble_reg"] = regression_metrics(y_test_reg, pred_reg)

    # ── Ensemble classification ──────────────────────
    pred_clf = ensembles["clf"].predict(X_test)
    prob_clf = ensembles["clf"].predict_proba(X_test)[:, 1]
    metrics["ensemble_clf"] = classification_metrics(y_test_clf, pred_clf, prob_clf)

    # ── Per-base-learner metrics ─────────────────────
    for name in ["xgb_reg", "lgb_reg", "ridge_reg"]:
        p = base_models[name].predict(X_test)
        metrics[name] = regression_metrics(y_test_reg, p)

    for name in ["xgb_clf", "lgb_clf", "lr_clf"]:
        m = base_models[name]
        p = m.predict(X_test)
        prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None
        metrics[name] = classification_metrics(y_test_clf, p, prob)

    return metrics


# ─────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────

def _plot_metrics(metrics: Dict[str, Any], save_dir: Path) -> Path:
    """Bar charts comparing ensemble vs base learners."""
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Regression ───────────────────────────────────
    reg_names = ["ensemble_reg", "xgb_reg", "lgb_reg", "ridge_reg"]
    reg_labels = ["Ensemble", "XGBoost", "LightGBM", "Ridge"]
    mae_vals = [metrics[n]["mae"] for n in reg_names]
    rmse_vals = [metrics[n]["rmse"] for n in reg_names]

    x = np.arange(len(reg_names))
    w = 0.35
    axes[0].bar(x - w / 2, mae_vals, w, label="MAE", color="steelblue")
    axes[0].bar(x + w / 2, rmse_vals, w, label="RMSE", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg_labels, rotation=15)
    axes[0].set_ylabel("Error (μs)")
    axes[0].set_title("Regression: Ensemble vs Base Learners")
    axes[0].legend()

    # ── Classification ────────────────────────────────
    clf_names = ["ensemble_clf", "xgb_clf", "lgb_clf", "lr_clf"]
    clf_labels = ["Ensemble", "XGBoost", "LightGBM", "LogReg"]
    f1_vals = [metrics[n].get("f1", 0) for n in clf_names]
    acc_vals = [metrics[n].get("accuracy", 0) for n in clf_names]
    auc_vals = [metrics[n].get("roc_auc", 0) for n in clf_names]

    x = np.arange(len(clf_names))
    w = 0.25
    axes[1].bar(x - w, f1_vals, w, label="F1", color="steelblue")
    axes[1].bar(x, acc_vals, w, label="Accuracy", color="coral")
    axes[1].bar(x + w, auc_vals, w, label="AUC", color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(clf_labels, rotation=15)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Classification: Ensemble vs Base Learners")
    axes[1].legend()

    plt.tight_layout()
    path = save_dir / "final_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    # ── 1. Data ──────────────────────────────────────
    print("Loading data …")
    df = load_train_ready()
    df = derive_targets(df)
    train_df, val_df, test_df = time_split(df)

    ct = build_feature_transformer()
    X_train = ct.fit_transform(train_df)
    X_val = ct.transform(val_df)
    X_test = ct.transform(test_df)

    y_train_reg = train_df[REG_TARGET].values
    y_val_reg = val_df[REG_TARGET].values
    y_test_reg = test_df[REG_TARGET].values

    y_train_clf = train_df[CLF_TARGET].values
    y_val_clf = val_df[CLF_TARGET].values
    y_test_clf = test_df[CLF_TARGET].values

    # ── 2. Train base learners ───────────────────────
    print("\n▶ Training 6 base learners …")
    base_models = _train_base_learners(
        X_train, y_train_reg, y_train_clf,
        X_val, y_val_reg, y_val_clf,
    )

    # ── 3. Build stacking ensembles ──────────────────
    print("\n▶ Fitting meta-learners (stacking) …")
    ensembles = _build_stacking_ensemble(base_models, X_val, y_val_reg, y_val_clf)

    # ── 4. Evaluate ──────────────────────────────────
    print("\n▶ Evaluating on test set …")
    metrics = _evaluate(ensembles, base_models, X_test, y_test_reg, y_test_clf)

    # Add metadata
    metrics["_meta"] = {
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "n_features": X_train.shape[1],
        "base_learners_reg": ["XGBoost", "LightGBM", "Ridge"],
        "base_learners_clf": ["XGBoost", "LightGBM", "LogisticRegression"],
        "meta_reg": "Ridge",
        "meta_clf": "LogisticRegression",
        "violation_threshold_us": VIOLATION_THRESHOLD_US,
    }

    # ── 5. Save ensemble ─────────────────────────────
    ens_path = MODEL_DIR / "final_ensemble.joblib"
    joblib.dump({
        "reg": ensembles["reg"],
        "clf": ensembles["clf"],
        "transformer": ct,
    }, ens_path)
    print(f"  Saved {ens_path}")

    # ── 6. Save metrics ──────────────────────────────
    metrics_path = REPORT_DIR / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved {metrics_path}")

    # ── 7. Plot ──────────────────────────────────────
    _plot_metrics(metrics, FIGURE_DIR)

    # ── Summary ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL ENSEMBLE RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 50)
    for name, label in [("ensemble_reg", "Ensemble"),
                        ("xgb_reg", "XGBoost"),
                        ("lgb_reg", "LightGBM"),
                        ("ridge_reg", "Ridge")]:
        m = metrics[name]
        print(f"{label:<20} {m['mae']:8.3f} {m['rmse']:8.3f} {m['r2']:8.4f}")

    print(f"\n{'Model':<20} {'F1':>8} {'AUC':>8} {'Acc':>8}")
    print("-" * 50)
    for name, label in [("ensemble_clf", "Ensemble"),
                        ("xgb_clf", "XGBoost"),
                        ("lgb_clf", "LightGBM"),
                        ("lr_clf", "LogReg")]:
        m = metrics[name]
        print(f"{label:<20} {m.get('f1',0):8.4f} {m.get('roc_auc',0):8.4f} {m.get('accuracy',0):8.4f}")

    print("=" * 70)
    print("✓ Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
