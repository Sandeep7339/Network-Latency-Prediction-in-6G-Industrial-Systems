"""
Hyperparameter optimisation for XGBoost and LSTM with time-series CV.

Time-series CV
--------------
``time_series_cv(df, n_splits)`` produces expanding-window folds:

    Fold 1:  train=[0, b1)   val=[b1, b2)
    Fold 2:  train=[0, b2)   val=[b2, b3)
    ...
    Fold K:  train=[0, bK)   val=[bK, bK+1)

where boundaries are equi-spaced breakpoints in the sorted data (by
``timestamp_ns``), so validation windows are non-overlapping and the
training window always *expands*.

XGBoost HPO
-----------
Randomised search over 12 sample configurations. Each is scored with
time-series CV (n_splits=4) using MAE (regression) or log-loss
(classification). Best params saved.

LSTM HPO
--------
10 random trials over learning_rate × hidden_dim × dropout.  Each
trained for 15 epochs max with early stopping (patience=5).

Usage
-----
    python -m src.models.hpo
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, log_loss
from torch.utils.data import DataLoader, Dataset

from src.features.feature_pipeline import FrequencyEncoder  # noqa: F401
from src.models.baseline import (
    CLF_TARGET,
    REG_TARGET,
    build_feature_transformer,
    derive_targets,
    load_train_ready,
    regression_metrics,
    classification_metrics,
    time_split,
)
from src.models.advanced import (
    LatencyLSTM,
    SequenceDataset,
    predict_lstm,
    SEQ_LEN,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Time-series cross-validation
# ═══════════════════════════════════════════════════════════════════════════
def time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    timestamp_col: str = "timestamp_ns",
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Expanding-window time-series CV with non-overlapping validation folds.

    Parameters
    ----------
    df : DataFrame sorted by *timestamp_col*.
    n_splits : number of train/val folds to produce.

    Yields
    ------
    (train_df, val_df) for each fold.

    Example with n_splits=4 on 100 rows:
        boundaries → [0, 20, 40, 60, 80, 100]
        fold 1: train=[0:20]   val=[20:40]
        fold 2: train=[0:40]   val=[40:60]
        fold 3: train=[0:60]   val=[60:80]
        fold 4: train=[0:80]   val=[80:100]
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)

    # n_splits folds ⇒ n_splits+1 equi-spaced boundaries
    boundaries = [int(round(i * n / (n_splits + 1))) for i in range(n_splits + 2)]

    for fold_idx in range(n_splits):
        train_end = boundaries[fold_idx + 1]
        val_end = boundaries[fold_idx + 2]
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        logger.info(
            "  Fold %d/%d — train: %d rows  val: %d rows",
            fold_idx + 1, n_splits, len(train_df), len(val_df),
        )
        yield train_df, val_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. XGBoost HPO
# ═══════════════════════════════════════════════════════════════════════════
XGB_SEARCH_SPACE: List[Dict[str, Any]] = []

_xgb_grid = {
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [1.0, 5.0],
}


def _sample_xgb_configs(n: int = 12, seed: int = 42) -> List[Dict[str, Any]]:
    """Draw *n* random configurations from ``_xgb_grid``."""
    rng = np.random.RandomState(seed)
    configs = []
    keys = list(_xgb_grid.keys())
    for _ in range(n):
        cfg = {k: rng.choice(_xgb_grid[k]) for k in keys}
        # Cast numpy types to Python types for JSON serialisation
        cfg = {k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
               for k, v in cfg.items()}
        configs.append(cfg)
    return configs


def hpo_xgboost(
    df: pd.DataFrame,
    target: str = REG_TARGET,
    task: str = "regression",
    n_splits: int = 4,
    n_configs: int = 12,
    n_estimators: int = 500,
    early_stopping_rounds: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Randomised HPO for XGBoost with time-series CV."""
    import xgboost as xgb

    configs = _sample_xgb_configs(n=n_configs, seed=seed)
    results: List[Dict[str, Any]] = []

    for cfg_idx, params in enumerate(configs, 1):
        fold_scores = []
        logger.info("Config %d/%d: %s", cfg_idx, n_configs, params)

        for train_df, val_df in time_series_cv(df, n_splits=n_splits):
            ct = build_feature_transformer()
            X_tr = ct.fit_transform(train_df)
            X_va = ct.transform(val_df)
            y_tr = train_df[target].values
            y_va = val_df[target].values

            if task == "regression":
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_metric="mae",
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=0,
                    **params,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                y_pred = model.predict(X_va)
                score = mean_absolute_error(y_va, y_pred)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_metric="logloss",
                    scale_pos_weight=(
                        (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
                    ),
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=0,
                    use_label_encoder=False,
                    **params,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                y_prob = model.predict_proba(X_va)[:, 1]
                score = log_loss(y_va, y_prob)

            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        results.append({
            "params": params,
            "fold_scores": [round(s, 6) for s in fold_scores],
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
        })
        logger.info(
            "  → mean_score=%.4f ± %.4f", mean_score, std_score,
        )

    # Lower is better for both MAE and log-loss
    results.sort(key=lambda r: r["mean_score"])
    best = results[0]
    logger.info("Best XGBoost config: %s  (score=%.4f)", best["params"], best["mean_score"])

    return {
        "task": task,
        "target": target,
        "metric": "mae" if task == "regression" else "logloss",
        "n_splits": n_splits,
        "n_configs": n_configs,
        "best_params": best["params"],
        "best_score": best["mean_score"],
        "all_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. LSTM HPO
# ═══════════════════════════════════════════════════════════════════════════
_lstm_grid = {
    "hidden_dim": [32, 64, 128],
    "learning_rate": [5e-4, 1e-3, 3e-3, 5e-3],
    "dropout": [0.1, 0.2, 0.3],
}


def _sample_lstm_configs(n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    configs = []
    keys = list(_lstm_grid.keys())
    for _ in range(n):
        cfg = {k: rng.choice(_lstm_grid[k]) for k in keys}
        cfg = {k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
               for k, v in cfg.items()}
        configs.append(cfg)
    return configs


def _train_lstm_quick(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    seq_len: int = SEQ_LEN,
    epochs: int = 10,
    batch_size: int = 256,
    patience: int = 4,
    task: str = "regression",
) -> Dict[str, Any]:
    """Train a small LSTM and return best val loss + best epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LatencyLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task=task,
    ).to(device)

    criterion = nn.MSELoss() if task == "regression" else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Build simple tensor datasets
    train_t = torch.from_numpy(X_train).float()
    val_t = torch.from_numpy(X_val).float()
    y_train_t = torch.from_numpy(y_train).float()
    y_val_t = torch.from_numpy(y_val).float()

    train_ds = torch.utils.data.TensorDataset(train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl = 0.0
        nb = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item()
                nb += 1
        vl /= max(nb, 1)

        if vl < best_val:
            best_val = vl
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    best_epoch = epoch - wait
    return {"best_val_loss": float(best_val), "best_epoch": best_epoch}


def _build_sequence_arrays(
    df: pd.DataFrame,
    ct,
    target: str,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform df → (X_seq, y_seq) arrays of shape (N, seq_len, features)."""
    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    X_all = ct.transform(df)
    y_all = df[target].values

    seqs, tgts = [], []
    for i in range(len(df) - seq_len + 1):
        seqs.append(X_all[i: i + seq_len])
        tgts.append(y_all[i + seq_len - 1])

    return (
        np.array(seqs, dtype=np.float32),
        np.array(tgts, dtype=np.float32),
    )


def hpo_lstm(
    df: pd.DataFrame,
    target: str = REG_TARGET,
    task: str = "regression",
    n_splits: int = 4,
    n_configs: int = 10,
    seq_len: int = SEQ_LEN,
    epochs_per_trial: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Randomised HPO for LSTM with time-series CV."""

    configs = _sample_lstm_configs(n=n_configs, seed=seed)
    results: List[Dict[str, Any]] = []

    for cfg_idx, params in enumerate(configs, 1):
        fold_scores = []
        logger.info("LSTM config %d/%d: %s", cfg_idx, n_configs, params)

        for train_df, val_df in time_series_cv(df, n_splits=n_splits):
            ct = build_feature_transformer()
            ct.fit(train_df)
            n_features = ct.transform(train_df[:1]).shape[1]

            X_tr_seq, y_tr_seq = _build_sequence_arrays(train_df, ct, target, seq_len)
            X_va_seq, y_va_seq = _build_sequence_arrays(val_df, ct, target, seq_len)

            if len(X_tr_seq) == 0 or len(X_va_seq) == 0:
                logger.warning("  Fold skipped — not enough rows for seq_len=%d", seq_len)
                continue

            res = _train_lstm_quick(
                X_tr_seq, y_tr_seq,
                X_va_seq, y_va_seq,
                input_dim=n_features,
                hidden_dim=params["hidden_dim"],
                learning_rate=params["learning_rate"],
                dropout=params["dropout"],
                seq_len=seq_len,
                epochs=epochs_per_trial,
                task=task,
            )
            fold_scores.append(res["best_val_loss"])

        if not fold_scores:
            continue

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        results.append({
            "params": params,
            "fold_scores": [round(s, 6) for s in fold_scores],
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
        })
        logger.info("  → mean_loss=%.4f ± %.4f", mean_score, std_score)

    results.sort(key=lambda r: r["mean_score"])
    best = results[0] if results else {"params": {}, "mean_score": float("nan")}
    logger.info("Best LSTM config: %s  (loss=%.4f)", best["params"], best["mean_score"])

    return {
        "task": task,
        "target": target,
        "metric": "mse_loss" if task == "regression" else "bce_loss",
        "n_splits": n_splits,
        "n_configs": n_configs,
        "seq_len": seq_len,
        "epochs_per_trial": epochs_per_trial,
        "best_params": best["params"],
        "best_score": best["mean_score"],
        "all_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════
def main(
    data_path: str | Path = "data/train_ready.parquet",
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
    n_splits: int = 4,
    xgb_n_configs: int = 12,
    lstm_n_configs: int = 6,
) -> Dict[str, Any]:
    """Run full HPO pipeline and save results."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    # Load data — use train+val portion only (first 85 %, hold out test)
    df = load_train_ready(data_path)
    df = derive_targets(df)
    train_df, val_df, test_df = time_split(df)
    # Combine train & val for CV (test stays untouched)
    cv_df = pd.concat([train_df, val_df], ignore_index=True)
    cv_df = cv_df.sort_values("timestamp_ns").reset_index(drop=True)
    logger.info("CV pool: %d rows  (test held out: %d rows)", len(cv_df), len(test_df))

    t0 = time.time()
    all_results: Dict[str, Any] = {}

    # ── XGBoost regression ───────────────────────────────────────────
    print("\n▶ XGBoost regression HPO…")
    xgb_reg = hpo_xgboost(
        cv_df, target=REG_TARGET, task="regression",
        n_splits=n_splits, n_configs=xgb_n_configs,
    )
    all_results["xgb_regression"] = xgb_reg

    # ── XGBoost classification ───────────────────────────────────────
    print("\n▶ XGBoost classification HPO…")
    xgb_clf = hpo_xgboost(
        cv_df, target=CLF_TARGET, task="classification",
        n_splits=n_splits, n_configs=xgb_n_configs,
    )
    all_results["xgb_classification"] = xgb_clf

    # ── LSTM regression ──────────────────────────────────────────────
    print("\n▶ LSTM regression HPO…")
    lstm_reg = hpo_lstm(
        cv_df, target=REG_TARGET, task="regression",
        n_splits=min(n_splits, 3), n_configs=lstm_n_configs,
        epochs_per_trial=10,
    )
    all_results["lstm_regression"] = lstm_reg

    # ── LSTM classification ──────────────────────────────────────────
    print("\n▶ LSTM classification HPO…")
    lstm_clf = hpo_lstm(
        cv_df, target=CLF_TARGET, task="classification",
        n_splits=min(n_splits, 3), n_configs=lstm_n_configs,
        epochs_per_trial=10,
    )
    all_results["lstm_classification"] = lstm_clf

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    # ── Collect best_params ──────────────────────────────────────────
    best_params: Dict[str, Any] = {
        "xgb_regression": xgb_reg["best_params"],
        "xgb_classification": xgb_clf["best_params"],
        "lstm_regression": lstm_reg["best_params"],
        "lstm_classification": lstm_clf["best_params"],
    }

    # ── Save ─────────────────────────────────────────────────────────
    results_path = report_dir / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    params_path = model_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HPO RESULTS")
    print("=" * 70)
    print(f"CV pool: {len(cv_df)} rows  |  n_splits: {n_splits}")
    print(f"Elapsed: {elapsed:.0f}s\n")

    print(f"{'Task':<25} {'Model':<8} {'Metric':<10} {'Best Score':>10}")
    print(f"{'─'*25} {'─'*8} {'─'*10} {'─'*10}")
    for key in ["xgb_regression", "xgb_classification",
                 "lstm_regression", "lstm_classification"]:
        r = all_results[key]
        model_name = "XGBoost" if "xgb" in key else "LSTM"
        print(f"{key:<25} {model_name:<8} {r['metric']:<10} {r['best_score']:10.4f}")

    print(f"\nBest XGBoost regression params: {xgb_reg['best_params']}")
    print(f"Best XGBoost classification params: {xgb_clf['best_params']}")
    print(f"Best LSTM regression params: {lstm_reg['best_params']}")
    print(f"Best LSTM classification params: {lstm_clf['best_params']}")

    print(f"\nSaved:")
    print(f"  • {results_path}")
    print(f"  • {params_path}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
