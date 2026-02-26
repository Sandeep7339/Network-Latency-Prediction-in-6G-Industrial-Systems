"""
SHAP-based explainability for the best tree model (XGBoost).

Produces
--------
- ``figures/shap_summary.png``           – global SHAP beeswarm / bar plot
- ``figures/shap_dependence_<feat>.png`` – dependence plots for top-5 features
- ``figures/shap_device_<id>.png``       – per-device SHAP bar summaries (top 5 devices)

Usage
-----
    python -m src.eval.explain
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")                    # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.feature_pipeline import FrequencyEncoder  # noqa: F401 (needed for joblib)
from src.models.baseline import (
    CLF_TARGET,
    REG_TARGET,
    _get_feature_names,
    build_feature_transformer,
    derive_targets,
    load_train_ready,
    time_split,
)

logger = logging.getLogger(__name__)

FIG_DIR = Path("figures")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _load_best_xgb(model_dir: Path = Path("models")) -> Any:
    """Load the best saved XGBoost model (prefer advanced, fall back to baseline)."""
    for name in ("advanced_xgb.joblib", "baseline_xgboost_reg.joblib"):
        p = model_dir / name
        if p.exists():
            logger.info("Loading tree model: %s", p)
            obj = joblib.load(p)
            # advanced saves a dict: {reg: XGBRegressor, clf: XGBClassifier, ...}
            if isinstance(obj, dict) and "reg" in obj:
                return obj["reg"]
            return obj
    raise FileNotFoundError("No XGBoost model found in %s" % model_dir)


def _prepare_test_data(
    data_path: str | Path = "data/train_ready.parquet",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Return (test_df, X_test, y_test, feature_names)."""
    df = load_train_ready(data_path)
    df = derive_targets(df)
    train_df, _, test_df = time_split(df)

    ct = build_feature_transformer()
    ct.fit(train_df)
    X_test = ct.transform(test_df)
    y_test = test_df[REG_TARGET].values
    feature_names = _get_feature_names(ct)

    return test_df, X_test, y_test, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# SHAP computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 2000,
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer."""
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X, columns=feature_names))
    return shap_values


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════
def save_shap_summary(sv: shap.Explanation, fig_dir: Path = FIG_DIR) -> Path:
    """Global SHAP bar summary → shap_summary.png."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "shap_summary.png"

    plt.figure(figsize=(10, 8))
    shap.plots.bar(sv, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved %s", out)
    return out


def save_shap_dependence(
    sv: shap.Explanation,
    top_n: int = 5,
    fig_dir: Path = FIG_DIR,
) -> List[Path]:
    """Dependence plots for top-N features → shap_dependence_<feat>.png."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Rank features by mean absolute SHAP value
    mean_abs = np.abs(sv.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    feature_names = sv.feature_names

    paths: List[Path] = []
    for i in top_idx:
        feat = feature_names[i]
        safe_name = feat.replace("/", "_").replace("\\", "_").replace(" ", "_")
        out = fig_dir / f"shap_dependence_{safe_name}.png"

        plt.figure(figsize=(8, 5))
        shap.plots.scatter(sv[:, i], show=False)
        plt.title(f"SHAP dependence — {feat}")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        paths.append(out)
        logger.info("Saved %s", out)

    return paths


def save_shap_per_device(
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    feature_names: list[str],
    model,
    top_n_devices: int = 5,
    fig_dir: Path = FIG_DIR,
) -> List[Path]:
    """Per-device SHAP bar summaries for top-N devices (by row count)."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    device_col = "src_device_id"
    if device_col not in test_df.columns:
        logger.warning("Column %s not found — skipping per-device SHAP.", device_col)
        return []

    top_devices = (
        test_df[device_col]
        .value_counts()
        .head(top_n_devices)
        .index.tolist()
    )

    explainer = shap.TreeExplainer(model)
    paths: List[Path] = []

    for dev_id in top_devices:
        mask = test_df[device_col].values == dev_id
        if mask.sum() == 0:
            continue
        X_dev = X_test[mask]
        sv_dev = explainer(pd.DataFrame(X_dev, columns=feature_names))

        safe_id = str(dev_id).replace("/", "_").replace(" ", "_")
        out = fig_dir / f"shap_device_{safe_id}.png"

        plt.figure(figsize=(10, 6))
        shap.plots.bar(sv_dev, max_display=15, show=False)
        plt.title(f"SHAP summary — device {dev_id}  ({mask.sum()} samples)")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        paths.append(out)
        logger.info("Saved %s", out)

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main(
    data_path: str | Path = "data/train_ready.parquet",
    model_dir: str | Path = "models",
    fig_dir: str | Path = "figures",
) -> Dict[str, Any]:
    """Run full SHAP explainability pipeline."""
    model_dir = Path(model_dir)
    fig_dir = Path(fig_dir)

    print("Loading model & data…")
    model = _load_best_xgb(model_dir)
    test_df, X_test, y_test, feature_names = _prepare_test_data(data_path)

    print(f"Computing SHAP values on {min(X_test.shape[0], 2000)} test samples…")
    sv = compute_shap_values(model, X_test, feature_names, max_samples=2000)

    print("Saving global SHAP summary…")
    summary_path = save_shap_summary(sv, fig_dir)

    print("Saving SHAP dependence plots (top 5)…")
    dep_paths = save_shap_dependence(sv, top_n=5, fig_dir=fig_dir)

    print("Saving per-device SHAP summaries (top 5 devices)…")
    device_paths = save_shap_per_device(
        test_df, X_test, feature_names, model,
        top_n_devices=5, fig_dir=fig_dir,
    )

    # Top feature importances by mean |SHAP|
    mean_abs = np.abs(sv.values).mean(axis=0)
    importance = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True,
    )[:20]

    result = {
        "summary_figure": str(summary_path),
        "dependence_figures": [str(p) for p in dep_paths],
        "device_figures": [str(p) for p in device_paths],
        "top_features": [{"feature": f, "mean_abs_shap": round(float(v), 6)} for f, v in importance],
    }

    print("\n✓ SHAP explainability complete.")
    print(f"  Global summary : {summary_path}")
    print(f"  Dependence     : {len(dep_paths)} figures")
    print(f"  Per-device     : {len(device_paths)} figures")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
