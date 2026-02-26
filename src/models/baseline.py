"""
Baseline regression & classification models for CN_project.

Regression target  : latency_us  (microseconds, continuous)
Classification target: latency_violation  (derived: latency_us > 120 µs)

Models
------
Regression:
    - MeanPredictor  (sklearn DummyRegressor, strategy='mean')
    - Ridge          (sklearn Ridge, alpha=1.0)
    - XGBoost        (xgboost XGBRegressor, 200 rounds)

Classification:
    - LogisticRegression  (sklearn, max_iter=1000, balanced)
    - LightGBM            (lightgbm LGBMClassifier, 200 rounds, balanced)

Split strategy
--------------
Time-ordered 70 / 15 / 15 using ``timestamp_ns`` row-order boundaries.

Usage
-----
    python -m src.models.baseline          # full training + save
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# Re-use constants & FrequencyEncoder from the feature pipeline
from src.features.feature_pipeline import (
    CAT_FREQ_COLS,
    CAT_ONEHOT_COLS,
    CAT_ORDINAL_COLS,
    DEVICE_COLS,
    FrequencyEncoder,
    LAG_STEPS,
    ORDINAL_ORDERS,
    PACKET_FLOW_COLS,
    QUEUE_CONTROLLER_COLS,
    ROLLING_WINDOWS_SEC,
    SECURITY_COLS,
)

logger = logging.getLogger(__name__)

# ─── Feature / target constants ──────────────────────────────────────────
VIOLATION_THRESHOLD_US = 120.0          # ~P90 ⇒ ≈10 % positive class
REG_TARGET = "latency_us"
CLF_TARGET = "latency_violation"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Timing columns *without* latency_us (it is the regression target and the
# source of the classification target, so it must be excluded from features).
TIMING_COLS_NO_TARGET = [
    "jitter_us",
    "enforcement_latency_us",
    "control_action_delay_us",
]


# ─── Feature helpers ─────────────────────────────────────────────────────
def _numeric_feature_cols() -> List[str]:
    """All numeric feature columns (excludes ``latency_us``)."""
    base = (
        PACKET_FLOW_COLS
        + DEVICE_COLS
        + TIMING_COLS_NO_TARGET
        + QUEUE_CONTROLLER_COLS
        + SECURITY_COLS
    )
    rolling = []
    for w in ROLLING_WINDOWS_SEC:
        rolling.extend([
            f"latency_roll_mean_{w}s",
            f"latency_roll_std_{w}s",
            f"packet_rate_{w}s",
        ])
    lags = [f"latency_lag_{l}" for l in LAG_STEPS]
    return base + rolling + lags


# ─── Data loading / splitting ────────────────────────────────────────────
def load_train_ready(path: str | Path = "data/train_ready.parquet") -> pd.DataFrame:
    """Load the feature-enriched dataset."""
    df = pd.read_parquet(path)
    logger.info("Loaded %s  (%d rows, %d cols)", path, len(df), len(df.columns))
    return df


def derive_targets(df: pd.DataFrame, threshold: float = VIOLATION_THRESHOLD_US) -> pd.DataFrame:
    """Add ``latency_violation`` column (binary: latency_us > threshold)."""
    df = df.copy()
    df[CLF_TARGET] = (df[REG_TARGET] > threshold).astype(int)
    logger.info(
        "latency_violation distribution: %s",
        df[CLF_TARGET].value_counts().to_dict(),
    )
    return df


def time_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split by ``timestamp_ns`` ordering (70/15/15)."""
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ─── Column transformer (same layout as feature_pipeline, minus latency_us)
def build_feature_transformer() -> ColumnTransformer:
    """Build a ``ColumnTransformer`` identical to the feature pipeline but
    *excluding* ``latency_us`` from the numeric block."""
    numeric_cols = _numeric_feature_cols()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    onehot_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="if_binary",
        )),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Normal")),
        ("ordinal", OrdinalEncoder(
            categories=ORDINAL_ORDERS,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    freq_pipe = Pipeline([
        ("freq_enc", FrequencyEncoder()),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("onehot", onehot_pipe, CAT_ONEHOT_COLS),
            ("ordinal", ordinal_pipe, CAT_ORDINAL_COLS),
            ("freq", freq_pipe, CAT_FREQ_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return ct


def _get_feature_names(ct: ColumnTransformer) -> List[str]:
    """Extract feature names from a fitted ``ColumnTransformer``."""
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names: List[str] = []
        for tname, transformer, cols in ct.transformers_:
            if tname == "remainder":
                continue
            pipe = transformer
            if isinstance(pipe, Pipeline):
                last = pipe[-1]
                if hasattr(last, "get_feature_names_out"):
                    try:
                        names.extend(last.get_feature_names_out(cols))
                    except Exception:
                        names.extend(cols)
                else:
                    names.extend(cols)
            elif hasattr(pipe, "get_feature_names_out"):
                names.extend(pipe.get_feature_names_out())
            else:
                names.extend(cols)
        return names


# ─── Metric helpers ──────────────────────────────────────────────────────
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    m: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            m["roc_auc"] = float("nan")
        try:
            m["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            m["avg_precision"] = float("nan")
        try:
            m["log_loss"] = float(log_loss(y_true, y_prob))
        except ValueError:
            m["log_loss"] = float("nan")
    return m


# ─── Model training ─────────────────────────────────────────────────────
def train_regression_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Train regression baselines → ``{name: {model, train_metrics, val_metrics}}``."""
    import xgboost as xgb

    models = {
        "mean_predictor": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "xgboost_reg": xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        logger.info("Training regression model: %s", name)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        results[name] = {
            "model": model,
            "train_metrics": regression_metrics(y_train, train_pred),
            "val_metrics": regression_metrics(y_val, val_pred),
        }
        logger.info(
            "  val — MAE=%.4f  RMSE=%.4f  R²=%.4f",
            results[name]["val_metrics"]["mae"],
            results[name]["val_metrics"]["rmse"],
            results[name]["val_metrics"]["r2"],
        )
    return results


def train_classification_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Train classification baselines → ``{name: {model, train_metrics, val_metrics}}``."""
    import lightgbm as lgb

    models = {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm_clf": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        logger.info("Training classification model: %s", name)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]

        results[name] = {
            "model": model,
            "train_metrics": classification_metrics(y_train, train_pred, train_prob),
            "val_metrics": classification_metrics(y_val, val_pred, val_prob),
        }
        logger.info(
            "  val — F1=%.4f  AUC=%.4f  AP=%.4f",
            results[name]["val_metrics"]["f1"],
            results[name]["val_metrics"].get("roc_auc", 0),
            results[name]["val_metrics"].get("avg_precision", 0),
        )
    return results


# ─── Test-set evaluation ─────────────────────────────────────────────────
def evaluate_on_test(
    reg_results: Dict,
    clf_results: Dict,
    X_test: np.ndarray,
    y_test_reg: np.ndarray,
    y_test_clf: np.ndarray,
) -> Dict[str, Dict]:
    """Evaluate all models on the held-out test set."""
    metrics: Dict[str, Dict] = {}

    for name, res in reg_results.items():
        pred = res["model"].predict(X_test)
        metrics[name] = regression_metrics(y_test_reg, pred)

    for name, res in clf_results.items():
        pred = res["model"].predict(X_test)
        prob = res["model"].predict_proba(X_test)[:, 1]
        metrics[name] = classification_metrics(y_test_clf, pred, prob)

    return metrics


# ─── Per-device analysis ─────────────────────────────────────────────────
def per_device_metrics_top10(
    df_test: pd.DataFrame,
    clf_results: Dict,
    ct: ColumnTransformer,
    n_devices: int = 10,
) -> Dict[str, Dict[str, Dict]]:
    """Classification metrics for the top-*n* devices by violation count."""
    dev_counts = (
        df_test.groupby("src_device_id")[CLF_TARGET]
        .sum()
        .sort_values(ascending=False)
    )
    top_devices = dev_counts.head(n_devices).index.tolist()

    per_device: Dict[str, Dict[str, Dict]] = {}
    for name, res in clf_results.items():
        model = res["model"]
        device_metrics: Dict[str, Dict] = {}
        for dev_id in top_devices:
            mask = df_test["src_device_id"] == dev_id
            if mask.sum() < 2:
                continue
            sub = df_test.loc[mask]
            X_sub = ct.transform(sub)
            y_sub = sub[CLF_TARGET].values
            pred = model.predict(X_sub)

            # If only one class present, skip probabilistic metrics
            if len(np.unique(y_sub)) > 1:
                prob = model.predict_proba(X_sub)[:, 1]
            else:
                prob = None

            device_metrics[str(dev_id)] = classification_metrics(y_sub, pred, prob)
        per_device[name] = device_metrics
    return per_device


# ─── Save artifacts ──────────────────────────────────────────────────────
def save_artifacts(
    reg_results: Dict,
    clf_results: Dict,
    ct: ColumnTransformer,
    all_metrics: Dict,
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
) -> List[str]:
    """Persist models and metrics JSON; return list of created paths."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    saved: List[str] = []

    # Column transformer
    ct_path = model_dir / "baseline_transformer.joblib"
    joblib.dump(ct, ct_path)
    saved.append(str(ct_path))

    # Individual models
    for name, res in {**reg_results, **clf_results}.items():
        p = model_dir / f"baseline_{name}.joblib"
        joblib.dump(res["model"], p)
        saved.append(str(p))

    # Metrics JSON
    metrics_path = report_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    saved.append(str(metrics_path))

    logger.info("Saved %d artifacts", len(saved))
    return saved


# ─── Main entry-point ────────────────────────────────────────────────────
def main(
    data_path: str | Path = "data/train_ready.parquet",
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """Full baseline training pipeline. Returns collected metrics dict."""

    # 1 — Load & derive targets
    df = load_train_ready(data_path)
    df = derive_targets(df)

    # 2 — Time-based split
    train_df, val_df, test_df = time_split(df)

    # 3 — Build & fit column transformer (fit on TRAIN only)
    ct = build_feature_transformer()
    X_train = ct.fit_transform(train_df)
    X_val = ct.transform(val_df)
    X_test = ct.transform(test_df)

    feature_names = _get_feature_names(ct)
    logger.info("Feature matrix: %d features, X_train=%s", len(feature_names), X_train.shape)

    # 4 — Extract targets
    y_train_reg = train_df[REG_TARGET].values
    y_val_reg = val_df[REG_TARGET].values
    y_test_reg = test_df[REG_TARGET].values

    y_train_clf = train_df[CLF_TARGET].values
    y_val_clf = val_df[CLF_TARGET].values
    y_test_clf = test_df[CLF_TARGET].values

    # 5 — Train regression baselines
    reg_results = train_regression_models(X_train, y_train_reg, X_val, y_val_reg)

    # 6 — Train classification baselines
    clf_results = train_classification_models(X_train, y_train_clf, X_val, y_val_clf)

    # 7 — Test-set evaluation
    test_metrics = evaluate_on_test(
        reg_results, clf_results, X_test, y_test_reg, y_test_clf,
    )

    # 8 — Per-device metrics (top 10 by violation count)
    per_device = per_device_metrics_top10(test_df, clf_results, ct)

    # 9 — Assemble all metrics
    all_metrics: Dict[str, Any] = {
        "split": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "violation_threshold_us": VIOLATION_THRESHOLD_US,
        "violation_rate": {
            "train": float(y_train_clf.mean()),
            "val": float(y_val_clf.mean()),
            "test": float(y_test_clf.mean()),
        },
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "regression": {
            name: {
                "train": res["train_metrics"],
                "val": res["val_metrics"],
                "test": test_metrics[name],
            }
            for name, res in reg_results.items()
        },
        "classification": {
            name: {
                "train": res["train_metrics"],
                "val": res["val_metrics"],
                "test": test_metrics[name],
            }
            for name, res in clf_results.items()
        },
        "per_device_top10": per_device,
    }

    # 10 — Save everything
    saved = save_artifacts(reg_results, clf_results, ct, all_metrics, model_dir, report_dir)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE MODEL RESULTS")
    print("=" * 70)
    print(f"\nData split:  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Features:    {len(feature_names)}")
    print(f"Violation threshold: >{VIOLATION_THRESHOLD_US} µs")
    vr = all_metrics["violation_rate"]
    print(f"Violation rate:  train={vr['train']:.3f}  val={vr['val']:.3f}  test={vr['test']:.3f}")

    print(f"\n{'─' * 70}")
    print("REGRESSION  (target: latency_us)")
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"{'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8}")
    for name in reg_results:
        m = test_metrics[name]
        print(f"{name:<20} {m['mae']:8.3f} {m['rmse']:8.3f} {m['r2']:8.4f}")

    print(f"\n{'─' * 70}")
    print("CLASSIFICATION  (target: latency_violation)")
    print(f"{'Model':<20} {'F1':>8} {'AUC':>8} {'AP':>8} {'Acc':>8}")
    print(f"{'─' * 20} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for name in clf_results:
        m = test_metrics[name]
        print(
            f"{name:<20} {m['f1']:8.4f} "
            f"{m.get('roc_auc', 0):8.4f} "
            f"{m.get('avg_precision', 0):8.4f} "
            f"{m['accuracy']:8.4f}"
        )

    print(f"\n{'─' * 70}")
    print("Saved artifacts:")
    for p in saved:
        print(f"  • {p}")
    print("=" * 70)

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
