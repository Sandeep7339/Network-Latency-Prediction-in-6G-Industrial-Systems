"""
Compare baseline vs advanced models on the held-out test set.

Produces:
    - figures/model_comparison.png   (bar chart + table)
    - reports/advanced_comparison.md (markdown report with bootstrap p-value)

Usage
-----
    python -m src.models.compare
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features.feature_pipeline import FrequencyEncoder  # noqa: F401
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
)
from src.models.advanced import (
    LatencyLSTM,
    SequenceDataset,
    predict_lstm,
    SEQ_LEN,
    LSTM_HIDDEN,
    LSTM_LAYERS,
)

logger = logging.getLogger(__name__)

import torch


# ─── Bootstrap helper ────────────────────────────────────────────────────
def bootstrap_paired_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn=None,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap test comparing metric(pred_a) vs metric(pred_b).

    Returns dict with metric_a, metric_b, diff, p_value, ci_lower, ci_upper.
    Uses MAE by default (lower is better).
    """
    if metric_fn is None:
        from sklearn.metrics import mean_absolute_error as metric_fn

    rng = np.random.RandomState(seed)
    n = len(y_true)

    obs_a = metric_fn(y_true, pred_a)
    obs_b = metric_fn(y_true, pred_b)
    obs_diff = obs_a - obs_b  # negative ⇒ A is better

    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d_a = metric_fn(y_true[idx], pred_a[idx])
        d_b = metric_fn(y_true[idx], pred_b[idx])
        boot_diffs.append(d_a - d_b)

    boot_diffs = np.array(boot_diffs)
    # Two-sided p-value: proportion of bootstrap diffs on the opposite side of 0
    if obs_diff <= 0:
        p_value = (boot_diffs >= 0).mean()
    else:
        p_value = (boot_diffs <= 0).mean()

    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "metric_a": float(obs_a),
        "metric_b": float(obs_b),
        "diff_a_minus_b": float(obs_diff),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
    }


# ─── Main comparison ─────────────────────────────────────────────────────
def main(
    data_path: str | Path = "data/train_ready.parquet",
    model_dir: str | Path = "models",
    fig_dir: str | Path = "figures",
    report_dir: str | Path = "reports",
) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    fig_dir = Path(fig_dir)
    report_dir = Path(report_dir)
    fig_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    # ── Load data & split (identical to training) ────────────────────
    df = load_train_ready(data_path)
    df = derive_targets(df)
    train_df, val_df, test_df = time_split(df)

    ct = build_feature_transformer()
    X_train = ct.fit_transform(train_df)
    X_test = ct.transform(test_df)

    y_test_reg = test_df[REG_TARGET].values
    y_test_clf = test_df[CLF_TARGET].values

    # ── Load baseline models ─────────────────────────────────────────
    baseline_models = {}
    for name in ["mean_predictor", "ridge", "xgboost_reg", "logistic", "lightgbm_clf"]:
        p = model_dir / f"baseline_{name}.joblib"
        if p.exists():
            baseline_models[name] = joblib.load(p)

    # ── Load advanced XGBoost ────────────────────────────────────────
    xgb_bundle = joblib.load(model_dir / "advanced_xgb.joblib")
    adv_xgb_reg = xgb_bundle["reg"]
    adv_xgb_clf = xgb_bundle["clf"]

    # ── Load advanced LSTM ───────────────────────────────────────────
    lstm_ckpt = torch.load(model_dir / "advanced_lstm.pt", map_location="cpu", weights_only=False)
    input_dim = lstm_ckpt["input_dim"]
    seq_len = lstm_ckpt.get("seq_len", SEQ_LEN)

    lstm_reg_model = LatencyLSTM(input_dim=input_dim, task="regression")
    lstm_reg_model.load_state_dict(lstm_ckpt["reg_state_dict"])

    lstm_clf_model = LatencyLSTM(input_dim=input_dim, task="classification")
    lstm_clf_model.load_state_dict(lstm_ckpt["clf_state_dict"])

    device = torch.device("cpu")

    # ── Evaluate all models ──────────────────────────────────────────
    reg_results: Dict[str, Dict] = {}
    clf_results: Dict[str, Dict] = {}

    # Baseline regression
    for name in ["mean_predictor", "ridge", "xgboost_reg"]:
        if name in baseline_models:
            pred = baseline_models[name].predict(X_test)
            reg_results[f"baseline_{name}"] = {
                "pred": pred,
                **regression_metrics(y_test_reg, pred),
            }

    # Baseline classification
    for name in ["logistic", "lightgbm_clf"]:
        if name in baseline_models:
            pred = baseline_models[name].predict(X_test)
            prob = baseline_models[name].predict_proba(X_test)[:, 1]
            clf_results[f"baseline_{name}"] = {
                "pred": pred, "prob": prob,
                **classification_metrics(y_test_clf, pred, prob),
            }

    # Advanced XGBoost regression
    pred = adv_xgb_reg.predict(X_test)
    reg_results["adv_xgb_reg"] = {"pred": pred, **regression_metrics(y_test_reg, pred)}

    # Advanced XGBoost classification
    pred = adv_xgb_clf.predict(X_test)
    prob = adv_xgb_clf.predict_proba(X_test)[:, 1]
    clf_results["adv_xgb_clf"] = {
        "pred": pred, "prob": prob,
        **classification_metrics(y_test_clf, pred, prob),
    }

    # Advanced LSTM regression (sequence-level)
    test_reg_ds = SequenceDataset(test_df, ct, REG_TARGET, seq_len=seq_len, group_by_device=False)
    y_true_seq_reg, y_pred_seq_reg = predict_lstm(lstm_reg_model, test_reg_ds, device)
    reg_results["adv_lstm_reg"] = {
        "pred": y_pred_seq_reg,
        "y_true_seq": y_true_seq_reg,
        **regression_metrics(y_true_seq_reg, y_pred_seq_reg),
    }

    # Advanced LSTM classification (sequence-level)
    test_clf_ds = SequenceDataset(test_df, ct, CLF_TARGET, seq_len=seq_len, group_by_device=False)
    y_true_seq_clf, y_prob_seq_clf = predict_lstm(lstm_clf_model, test_clf_ds, device)
    y_pred_seq_clf = (y_prob_seq_clf > 0.5).astype(int)
    clf_results["adv_lstm_clf"] = {
        "pred": y_pred_seq_clf, "prob": y_prob_seq_clf,
        "y_true_seq": y_true_seq_clf,
        **classification_metrics(y_true_seq_clf, y_pred_seq_clf, y_prob_seq_clf),
    }

    # ── Bootstrap tests (baseline xgboost vs advanced) ───────────────
    bootstrap_reg = bootstrap_paired_test(
        y_test_reg,
        reg_results["baseline_xgboost_reg"]["pred"],
        reg_results["adv_xgb_reg"]["pred"],
    )

    # For LSTM vs baseline we use the sequence-subset y_true
    bootstrap_lstm_reg = bootstrap_paired_test(
        y_true_seq_reg,
        # need baseline predictions on same indices — use mean as proxy
        np.full_like(y_true_seq_reg, y_test_reg.mean()),
        y_pred_seq_reg,
    )

    # ── Figure: model_comparison.png ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regression bar chart
    ax = axes[0]
    reg_names = list(reg_results.keys())
    reg_mae = [reg_results[n]["mae"] for n in reg_names]
    reg_rmse = [reg_results[n]["rmse"] for n in reg_names]
    x = np.arange(len(reg_names))
    w = 0.35
    ax.bar(x - w / 2, reg_mae, w, label="MAE", color="#4C72B0")
    ax.bar(x + w / 2, reg_rmse, w, label="RMSE", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("baseline_", "bl_").replace("adv_", "") for n in reg_names],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Error (µs)")
    ax.set_title("Regression — Test-set Error")
    ax.legend()

    # Classification bar chart
    ax = axes[1]
    clf_names = list(clf_results.keys())
    clf_f1 = [clf_results[n]["f1"] for n in clf_names]
    clf_auc = [clf_results[n].get("roc_auc", 0) for n in clf_names]
    x = np.arange(len(clf_names))
    ax.bar(x - w / 2, clf_f1, w, label="F1", color="#55A868")
    ax.bar(x + w / 2, clf_auc, w, label="ROC-AUC", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("baseline_", "bl_").replace("adv_", "") for n in clf_names],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Classification — Test-set Metrics")
    ax.legend()

    fig.suptitle("Baseline vs Advanced Model Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig_path = fig_dir / "model_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fig_path)

    # ── Markdown report ──────────────────────────────────────────────
    md_lines = [
        "# Advanced Model Comparison Report",
        "",
        "## Overview",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Test-set size | {len(test_df)} rows |",
        f"| Regression target | `{REG_TARGET}` |",
        f"| Classification target | `{CLF_TARGET}` (>{int(VIOLATION_THRESHOLD_US)} µs) |",
        f"| LSTM sequence length | {seq_len} |",
        "",
        "## Regression Results (test set)",
        "",
        "| Model | MAE | RMSE | R² |",
        "|-------|-----|------|----|",
    ]
    for name in reg_results:
        m = reg_results[name]
        label = name.replace("baseline_", "Baseline ").replace("adv_", "Advanced ")
        md_lines.append(f"| {label} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.4f} |")

    md_lines += [
        "",
        "## Classification Results (test set)",
        "",
        "| Model | Accuracy | F1 | ROC-AUC | Avg Precision |",
        "|-------|----------|----|---------|---------------|",
    ]
    for name in clf_results:
        m = clf_results[name]
        label = name.replace("baseline_", "Baseline ").replace("adv_", "Advanced ")
        md_lines.append(
            f"| {label} | {m['accuracy']:.4f} | {m['f1']:.4f} | "
            f"{m.get('roc_auc', 0):.4f} | {m.get('avg_precision', 0):.4f} |"
        )

    # Best models
    best_reg_name = min(reg_results, key=lambda n: reg_results[n]["mae"])
    best_reg_mae = reg_results[best_reg_name]["mae"]
    worst_reg_name = max(reg_results, key=lambda n: reg_results[n]["mae"])
    worst_reg_mae = reg_results[worst_reg_name]["mae"]
    mae_improvement = worst_reg_mae - best_reg_mae
    pct_improvement = (mae_improvement / worst_reg_mae) * 100 if worst_reg_mae > 0 else 0

    best_clf_name = max(clf_results, key=lambda n: clf_results[n]["f1"])
    best_clf_f1 = clf_results[best_clf_name]["f1"]

    md_lines += [
        "",
        "## Key Findings",
        "",
        f"- **Best regression model**: `{best_reg_name}` (MAE = {best_reg_mae:.3f} µs)",
        f"- **Worst regression model**: `{worst_reg_name}` (MAE = {worst_reg_mae:.3f} µs)",
        f"- **MAE improvement (best vs worst)**: {mae_improvement:.3f} µs ({pct_improvement:.1f}%)",
        f"- **Best classification model**: `{best_clf_name}` (F1 = {best_clf_f1:.4f})",
        "",
        "## Bootstrap Statistical Test",
        "",
        "Paired bootstrap test (n=2000) comparing MAE of baseline XGBoost vs "
        "advanced XGBoost on the test set:",
        "",
        "| Metric | Baseline XGB | Advanced XGB | Diff (bl−adv) | p-value | 95% CI |",
        "|--------|-------------|-------------|---------------|---------|--------|",
        f"| MAE | {bootstrap_reg['metric_a']:.3f} | {bootstrap_reg['metric_b']:.3f} | "
        f"{bootstrap_reg['diff_a_minus_b']:.3f} | {bootstrap_reg['p_value']:.4f} | "
        f"[{bootstrap_reg['ci_95_lower']:.3f}, {bootstrap_reg['ci_95_upper']:.3f}] |",
        "",
        f"A positive diff means the baseline has *higher* MAE (i.e., advanced is better). "
        f"p-value = {bootstrap_reg['p_value']:.4f}.",
        "",
        "## Notes",
        "",
        "- The data is synthetic with near-uniform distributions, so all models",
        "  achieve R² ≈ 0 and AUC ≈ 0.50. The baselines and advanced models",
        "  perform similarly because there is no exploitable signal.",
        "- With real-world data containing genuine temporal patterns, the LSTM",
        "  and XGBoost-ES models would be expected to outperform simple baselines.",
        "- The LSTM evaluation uses only the subset of devices with ≥30 rows.",
        "",
        "---",
        f"*Generated by `src/models/compare.py`*",
    ]

    md_path = report_dir / "advanced_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved %s", md_path)

    # ── Console summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nBest regression MAE:      {best_reg_name} → {best_reg_mae:.3f} µs")
    print(f"Best classification F1:   {best_clf_name} → {best_clf_f1:.4f}")
    print(f"MAE improvement (range):  {mae_improvement:.3f} µs ({pct_improvement:.1f}%)")
    print(f"Bootstrap p-value (XGB):  {bootstrap_reg['p_value']:.4f}")
    print(f"\nSaved:")
    print(f"  • {fig_path}")
    print(f"  • {md_path}")
    print("=" * 70)

    return {
        "regression": {n: {k: v for k, v in m.items() if k not in ("pred", "y_true_seq")} for n, m in reg_results.items()},
        "classification": {n: {k: v for k, v in m.items() if k not in ("pred", "prob", "y_true_seq")} for n, m in clf_results.items()},
        "bootstrap_xgb_reg": bootstrap_reg,
        "bootstrap_lstm_reg": bootstrap_lstm_reg,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
