"""
Advanced models for CN_project — XGBoost (early-stopping) + LSTM (PyTorch).

Regression target  : latency_us
Classification target: latency_violation  (>120 µs)

Models
------
1. XGBoost with early stopping (500 rounds, patience 30)
2. LSTM sequence model (PyTorch) — consumes last N=30 time-steps per
   ``src_device_id`` group.  Supports both regression and classification.

Usage
-----
    python -m src.models.advanced           # full training + save
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.baseline import (
    CLF_TARGET,
    REG_TARGET,
    VIOLATION_THRESHOLD_US,
    build_feature_transformer,
    classification_metrics,
    derive_targets,
    load_train_ready,
    regression_metrics,
    time_split,
    _get_feature_names,
)
from src.features.feature_pipeline import FrequencyEncoder  # noqa: F401

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────
SEQ_LEN: int = 30          # look-back window for LSTM
LSTM_HIDDEN: int = 64
LSTM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2
LSTM_LR: float = 1e-3
LSTM_EPOCHS: int = 30
LSTM_BATCH: int = 256
LSTM_PATIENCE: int = 8

XGB_ROUNDS: int = 500
XGB_PATIENCE: int = 30


# ═══════════════════════════════════════════════════════════════════════════
# 1. XGBoost with early stopping
# ═══════════════════════════════════════════════════════════════════════════
def train_xgb_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = XGB_ROUNDS,
    early_stopping_rounds: int = XGB_PATIENCE,
    task: str = "regression",
) -> Dict[str, Any]:
    """Train XGBoost with early stopping.

    Parameters
    ----------
    task : 'regression' or 'classification'

    Returns
    -------
    dict with keys: model, train_metrics, val_metrics, best_iteration, history
    """
    import xgboost as xgb

    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="mae",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_m = regression_metrics(y_train, train_pred)
        val_m = regression_metrics(y_val, val_pred)
    else:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        train_m = classification_metrics(y_train, train_pred, train_prob)
        val_m = classification_metrics(y_val, val_pred, val_prob)

    best_iter = getattr(model, "best_iteration", n_estimators)
    evals = model.evals_result()

    logger.info(
        "XGBoost (%s) — best_iteration=%d  val_metric=%s",
        task, best_iter,
        {k: round(v, 4) for k, v in val_m.items()},
    )

    return {
        "model": model,
        "train_metrics": train_m,
        "val_metrics": val_m,
        "best_iteration": best_iter,
        "evals_result": evals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sequence dataset & LSTM
# ═══════════════════════════════════════════════════════════════════════════
class SequenceDataset(Dataset):
    """Sliding-window dataset for LSTM.

    Sorts the full split by ``timestamp_ns``, transforms features via the
    fitted ``ColumnTransformer``, and creates overlapping windows of length
    ``seq_len``.  The target is the value at the *last* time-step.

    If ``group_by_device=True`` (default), windows are built per
    ``src_device_id`` so that sequences don't cross device boundaries.
    When a split has too few rows per device (common in val / test), set
    ``group_by_device=False`` to use a global sliding window.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ct,
        target_col: str,
        seq_len: int = SEQ_LEN,
        device_col: str = "src_device_id",
        group_by_device: bool = True,
    ):
        self.sequences: List[np.ndarray] = []
        self.targets: List[float] = []

        df = df.sort_values("timestamp_ns").reset_index(drop=True)

        if group_by_device:
            # Per-device windows
            for dev_id, grp in df.groupby(device_col):
                if len(grp) < seq_len:
                    continue
                X_dev = ct.transform(grp)
                y_dev = grp[target_col].values
                for i in range(len(grp) - seq_len + 1):
                    self.sequences.append(X_dev[i: i + seq_len])
                    self.targets.append(y_dev[i + seq_len - 1])
        else:
            # Global sliding window (ignores device boundaries)
            X_all = ct.transform(df)
            y_all = df[target_col].values
            for i in range(len(df) - seq_len + 1):
                self.sequences.append(X_all[i: i + seq_len])
                self.targets.append(y_all[i + seq_len - 1])

        self.sequences = np.array(self.sequences, dtype=np.float32) if self.sequences else np.empty((0, seq_len, 0), dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32) if self.targets else np.empty(0, dtype=np.float32)
        logger.info(
            "SequenceDataset: %d sequences  (seq_len=%d, features=%s)",
            len(self.sequences), seq_len,
            self.sequences.shape[2] if len(self.sequences) > 0 else 0,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.tensor(self.targets[idx]),
        )


class LatencyLSTM(nn.Module):
    """LSTM model for latency regression or classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = LSTM_HIDDEN,
        n_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = 1
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden)
        out = self.fc(self.dropout(last_hidden))  # (batch, 1)
        if self.task == "classification":
            out = torch.sigmoid(out)
        return out.squeeze(-1)


def train_lstm(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_dim: int,
    task: str = "regression",
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH,
    lr: float = LSTM_LR,
    patience: int = LSTM_PATIENCE,
) -> Dict[str, Any]:
    """Train the LSTM model; return dict with model, metrics, history."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("LSTM training on device: %s", device)

    model = LatencyLSTM(input_dim=input_dim, task=task).to(device)

    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d/%d — train_loss=%.4f  val_loss=%.4f",
                epoch, epochs, train_loss, val_loss,
            )

        # ── Early stopping ───────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": len(history["train_loss"]) - wait,
        "device": device,
    }


def predict_lstm(
    model: LatencyLSTM,
    ds: SequenceDataset,
    device: torch.device,
    batch_size: int = LSTM_BATCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays from the LSTM on a SequenceDataset."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y_batch.numpy())
    return np.concatenate(all_true), np.concatenate(all_preds)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Main entry-point
# ═══════════════════════════════════════════════════════════════════════════
def main(
    data_path: str | Path = "data/train_ready.parquet",
    model_dir: str | Path = "models",
    report_dir: str | Path = "reports",
    seq_len: int = SEQ_LEN,
) -> Dict[str, Any]:
    """Full advanced training pipeline."""

    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    # 1 — Load & split (same as baseline for fair comparison)
    df = load_train_ready(data_path)
    df = derive_targets(df)
    train_df, val_df, test_df = time_split(df)

    # 2 — Build & fit transformer (train only)
    ct = build_feature_transformer()
    X_train = ct.fit_transform(train_df)
    X_val = ct.transform(val_df)
    X_test = ct.transform(test_df)
    feature_names = _get_feature_names(ct)
    n_features = X_train.shape[1]

    y_train_reg = train_df[REG_TARGET].values
    y_val_reg = val_df[REG_TARGET].values
    y_test_reg = test_df[REG_TARGET].values

    y_train_clf = train_df[CLF_TARGET].values
    y_val_clf = val_df[CLF_TARGET].values
    y_test_clf = test_df[CLF_TARGET].values

    all_metrics: Dict[str, Any] = {
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "n_features": n_features,
        "seq_len": seq_len,
    }

    # ═══ 3A — XGBoost regression (early stopping) ════════════════════
    print("\n▶ Training XGBoost regression (early stopping)…")
    xgb_reg = train_xgb_early_stopping(
        X_train, y_train_reg, X_val, y_val_reg, task="regression",
    )
    xgb_reg_test_pred = xgb_reg["model"].predict(X_test)
    xgb_reg_test_m = regression_metrics(y_test_reg, xgb_reg_test_pred)
    all_metrics["xgb_reg"] = {
        "train": xgb_reg["train_metrics"],
        "val": xgb_reg["val_metrics"],
        "test": xgb_reg_test_m,
        "best_iteration": xgb_reg["best_iteration"],
    }

    # ═══ 3B — XGBoost classification (early stopping) ════════════════
    print("\n▶ Training XGBoost classification (early stopping)…")
    xgb_clf = train_xgb_early_stopping(
        X_train, y_train_clf, X_val, y_val_clf, task="classification",
    )
    xgb_clf_test_pred = xgb_clf["model"].predict(X_test)
    xgb_clf_test_prob = xgb_clf["model"].predict_proba(X_test)[:, 1]
    xgb_clf_test_m = classification_metrics(y_test_clf, xgb_clf_test_pred, xgb_clf_test_prob)
    all_metrics["xgb_clf"] = {
        "train": xgb_clf["train_metrics"],
        "val": xgb_clf["val_metrics"],
        "test": xgb_clf_test_m,
        "best_iteration": xgb_clf["best_iteration"],
    }

    # Save XGBoost models
    xgb_path = model_dir / "advanced_xgb.joblib"
    joblib.dump({
        "reg": xgb_reg["model"],
        "clf": xgb_clf["model"],
        "evals_reg": xgb_reg["evals_result"],
        "evals_clf": xgb_clf["evals_result"],
    }, xgb_path)
    logger.info("Saved XGBoost models → %s", xgb_path)

    # ═══ 4A — LSTM regression ════════════════════════════════════════
    # Decide grouping strategy: use per-device if devices have enough rows,
    # otherwise fall back to global sliding window.
    dev_counts_val = val_df.groupby("src_device_id").size()
    use_device_group = (dev_counts_val >= seq_len).sum() > 10
    logger.info(
        "Sequence grouping: %s  (devices≥%d in val: %d)",
        "per-device" if use_device_group else "global",
        seq_len, (dev_counts_val >= seq_len).sum(),
    )

    print(f"\n▶ Building sequence datasets (seq_len={seq_len}, group_by_device={use_device_group})…")
    train_reg_ds = SequenceDataset(train_df, ct, REG_TARGET, seq_len=seq_len, group_by_device=use_device_group)
    val_reg_ds = SequenceDataset(val_df, ct, REG_TARGET, seq_len=seq_len, group_by_device=use_device_group)
    test_reg_ds = SequenceDataset(test_df, ct, REG_TARGET, seq_len=seq_len, group_by_device=use_device_group)

    print(f"\n▶ Training LSTM regression (epochs={LSTM_EPOCHS})…")
    lstm_reg_res = train_lstm(
        train_reg_ds, val_reg_ds,
        input_dim=n_features,
        task="regression",
    )
    lstm_reg_model = lstm_reg_res["model"]
    lstm_reg_device = lstm_reg_res["device"]

    y_true_reg_seq, y_pred_reg_seq = predict_lstm(lstm_reg_model, test_reg_ds, lstm_reg_device)
    lstm_reg_test_m = regression_metrics(y_true_reg_seq, y_pred_reg_seq)
    all_metrics["lstm_reg"] = {
        "test": lstm_reg_test_m,
        "best_epoch": lstm_reg_res["best_epoch"],
        "best_val_loss": float(lstm_reg_res["best_val_loss"]),
        "history": {k: [round(v, 6) for v in vals] for k, vals in lstm_reg_res["history"].items()},
    }

    # ═══ 4B — LSTM classification ════════════════════════════════════
    train_clf_ds = SequenceDataset(train_df, ct, CLF_TARGET, seq_len=seq_len, group_by_device=use_device_group)
    val_clf_ds = SequenceDataset(val_df, ct, CLF_TARGET, seq_len=seq_len, group_by_device=use_device_group)
    test_clf_ds = SequenceDataset(test_df, ct, CLF_TARGET, seq_len=seq_len, group_by_device=use_device_group)

    print(f"\n▶ Training LSTM classification (epochs={LSTM_EPOCHS})…")
    lstm_clf_res = train_lstm(
        train_clf_ds, val_clf_ds,
        input_dim=n_features,
        task="classification",
    )
    lstm_clf_model = lstm_clf_res["model"]
    lstm_clf_device = lstm_clf_res["device"]

    y_true_clf_seq, y_prob_clf_seq = predict_lstm(lstm_clf_model, test_clf_ds, lstm_clf_device)
    y_pred_clf_seq = (y_prob_clf_seq > 0.5).astype(int)
    lstm_clf_test_m = classification_metrics(y_true_clf_seq, y_pred_clf_seq, y_prob_clf_seq)
    all_metrics["lstm_clf"] = {
        "test": lstm_clf_test_m,
        "best_epoch": lstm_clf_res["best_epoch"],
        "best_val_loss": float(lstm_clf_res["best_val_loss"]),
        "history": {k: [round(v, 6) for v in vals] for k, vals in lstm_clf_res["history"].items()},
    }

    # Save LSTM models
    lstm_path = model_dir / "advanced_lstm.pt"
    torch.save({
        "reg_state_dict": lstm_reg_model.state_dict(),
        "clf_state_dict": lstm_clf_model.state_dict(),
        "input_dim": n_features,
        "hidden_dim": LSTM_HIDDEN,
        "n_layers": LSTM_LAYERS,
        "seq_len": seq_len,
    }, lstm_path)
    logger.info("Saved LSTM models → %s", lstm_path)

    # Save transformer used
    ct_path = model_dir / "advanced_transformer.joblib"
    joblib.dump(ct, ct_path)

    # Save metrics
    metrics_path = report_dir / "advanced_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ADVANCED MODEL RESULTS")
    print("=" * 70)
    print(f"Split:  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Features: {n_features}   Seq len: {seq_len}")

    print(f"\n{'─' * 70}")
    print("REGRESSION  (target: latency_us)")
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Note':>20}")
    print(f"{'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 20}")
    m = xgb_reg_test_m
    print(f"{'XGBoost-ES':<25} {m['mae']:8.3f} {m['rmse']:8.3f} {m['r2']:8.4f} {'iter=' + str(xgb_reg['best_iteration']):>20}")
    m = lstm_reg_test_m
    print(f"{'LSTM':<25} {m['mae']:8.3f} {m['rmse']:8.3f} {m['r2']:8.4f} {'ep=' + str(lstm_reg_res['best_epoch']):>20}")

    print(f"\n{'─' * 70}")
    print("CLASSIFICATION  (target: latency_violation)")
    print(f"{'Model':<25} {'F1':>8} {'AUC':>8} {'AP':>8} {'Acc':>8}")
    print(f"{'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    m = xgb_clf_test_m
    print(f"{'XGBoost-ES':<25} {m['f1']:8.4f} {m.get('roc_auc',0):8.4f} {m.get('avg_precision',0):8.4f} {m['accuracy']:8.4f}")
    m = lstm_clf_test_m
    print(f"{'LSTM':<25} {m['f1']:8.4f} {m.get('roc_auc',0):8.4f} {m.get('avg_precision',0):8.4f} {m['accuracy']:8.4f}")

    print(f"\n{'─' * 70}")
    saved_files = [
        str(xgb_path), str(lstm_path), str(ct_path), str(metrics_path),
    ]
    print("Saved artifacts:")
    for p in saved_files:
        print(f"  • {p}")
    print("=" * 70)

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
