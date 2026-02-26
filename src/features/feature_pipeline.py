"""
Feature engineering pipeline for the CN_project.

Builds grouped features (packet/flow, device, timing, queue/controller,
security), rolling & lag features, categorical encodings, and wraps
everything in a scikit-learn ``ColumnTransformer`` that can be persisted
with joblib.

Usage
-----
    from src.features.feature_pipeline import build_pipeline, prepare_dataset

    X, y, feature_names, pipeline = build_pipeline("data/Combined_Dataset_40k.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Feature groups — each list names the raw columns that belong to a group
# ──────────────────────────────────────────────────────────────────────────
PACKET_FLOW_COLS = [
    "packet_size_bytes",
    "flow_priority",
    "scheduled_slot",
    "packet_loss",
]

DEVICE_COLS = [
    "cpu_usage",
    "memory_usage",
    "battery_level",
    "trust_score",
]

TIMING_COLS = [
    "latency_us",
    "jitter_us",
    "enforcement_latency_us",
    "control_action_delay_us",
]

QUEUE_CONTROLLER_COLS = [
    "queue_occupancy",
    "rerouted_flows",
    "scheduling_adjustment",
    "latency_restored",
]

SECURITY_COLS = [
    "traffic_deviation",
    "behavior_anomaly_score",
    "affected_flows",
    "execution_slot",
]

# Categorical groupings
CAT_ONEHOT_COLS = [
    "traffic_type",        # 4
    "protocol",            # 3
    "device_type",         # 4
    "vendor",              # 4
    "mobility_state",      # 2
    "attack_type",         # 5
    "action_type",         # 3
]

CAT_ORDINAL_COLS = [
    "operational_state",   # Normal < Degraded < Fault
    "severity_level",      # Low < Medium < High < Critical
    "controller_state",    # Normal < Congested < Under Attack
]

ORDINAL_ORDERS = [
    ["Normal", "Degraded", "Fault"],
    ["Low", "Medium", "High", "Critical"],
    ["Normal", "Congested", "Under Attack"],
]

# High-cardinality columns → frequency encoding
CAT_FREQ_COLS = [
    "src_device_id",
    "dst_device_id",
    "firmware_version",
]

# Target column
TARGET_COL = "success_flag"

# Columns explicitly excluded
DROP_COLS = [
    "timestamp_ns",
    "event_id",
    "action_id",
    "anomaly_label",       # constant (all True)
]

# Rolling / lag configuration
ROLLING_WINDOWS_SEC = [1, 10, 60]
LAG_STEPS = list(range(1, 11))  # lag_1 … lag_10


# ──────────────────────────────────────────────────────────────────────────
# Custom transformers
# ──────────────────────────────────────────────────────────────────────────
def _reconstruct_frequency_encoder(freq_maps):
    """Reconstruct helper for pickling FrequencyEncoder via canonical path."""
    enc = FrequencyEncoder.__new__(FrequencyEncoder)
    enc.freq_maps_ = freq_maps
    return enc

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set frequency (0-1)."""

    def __init__(self):
        self.freq_maps_: Dict[str, Dict] = {}

    def __reduce__(self):
        # Always pickle with the canonical module path so joblib.load works
        # regardless of whether the class was defined in __main__.
        return (
            _reconstruct_frequency_encoder,
            (self.freq_maps_,),
        )

    def fit(self, X: pd.DataFrame, y=None):
        self.freq_maps_ = {}
        for col in X.columns:
            counts = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = counts.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.freq_maps_.get(col, {})
            out[col] = X[col].map(mapping).fillna(0.0)
        return out.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{c}_freq" for c in input_features])
        return np.array([f"{c}_freq" for c in self.freq_maps_])


class RollingLagFeatures(BaseEstimator, TransformerMixin):
    """Compute rolling & lag features from a *pre-sorted* DataFrame.

    This transformer expects ``timestamp_ns`` and ``latency_us`` (and
    optionally ``packet_size_bytes``) in the input.  It is applied
    **before** the ColumnTransformer, directly on the DataFrame.
    """

    def __init__(
        self,
        rolling_windows_sec: List[int] | None = None,
        lag_steps: List[int] | None = None,
    ):
        self.rolling_windows_sec = rolling_windows_sec or ROLLING_WINDOWS_SEC
        self.lag_steps = lag_steps or LAG_STEPS
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        # stateless — just record feature names
        names = []
        for w in self.rolling_windows_sec:
            names.extend([
                f"latency_roll_mean_{w}s",
                f"latency_roll_std_{w}s",
                f"packet_rate_{w}s",
            ])
        for lag in self.lag_steps:
            names.append(f"latency_lag_{lag}")
        self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Ensure sorted by timestamp
        if "timestamp_ns" in df.columns:
            df = df.sort_values("timestamp_ns").reset_index(drop=True)

        # Convert timestamp_ns → datetime index for rolling
        if "timestamp_ns" in df.columns:
            df["_ts"] = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
        else:
            df["_ts"] = pd.RangeIndex(len(df))

        df = df.set_index("_ts", drop=True)

        # ── Rolling features ─────────────────────────────────────────
        for w in self.rolling_windows_sec:
            win = f"{w}s"
            roll = df["latency_us"].rolling(win, min_periods=1)
            df[f"latency_roll_mean_{w}s"] = roll.mean()
            df[f"latency_roll_std_{w}s"] = roll.std().fillna(0.0)

            if "packet_size_bytes" in df.columns:
                df[f"packet_rate_{w}s"] = (
                    df["packet_size_bytes"]
                    .rolling(win, min_periods=1)
                    .count()
                )
            else:
                df[f"packet_rate_{w}s"] = (
                    df["latency_us"]
                    .rolling(win, min_periods=1)
                    .count()
                )

        # ── Lag features ─────────────────────────────────────────────
        for lag in self.lag_steps:
            df[f"latency_lag_{lag}"] = df["latency_us"].shift(lag)

        df = df.reset_index(drop=True)

        # Fill lag NaNs at the top with forward value (or 0)
        lag_cols = [f"latency_lag_{l}" for l in self.lag_steps]
        df[lag_cols] = df[lag_cols].bfill().fillna(0.0)

        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline assembly helpers
# ──────────────────────────────────────────────────────────────────────────
def _all_numeric_cols() -> List[str]:
    """Union of all numeric feature groups + rolling/lag names."""
    base = (
        PACKET_FLOW_COLS
        + DEVICE_COLS
        + TIMING_COLS
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


def build_column_transformer() -> ColumnTransformer:
    """Build the scikit-learn ``ColumnTransformer``."""
    numeric_cols = _all_numeric_cols()

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
            ("num",     numeric_pipe,  numeric_cols),
            ("onehot",  onehot_pipe,   CAT_ONEHOT_COLS),
            ("ordinal", ordinal_pipe,  CAT_ORDINAL_COLS),
            ("freq",    freq_pipe,     CAT_FREQ_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return ct


def get_feature_names(ct: ColumnTransformer) -> List[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    names: List[str] = []
    for tname, transformer, cols in ct.transformers_:
        if tname == "remainder":
            continue
        pipe = transformer
        if hasattr(pipe, "get_feature_names_out"):
            names.extend(pipe.get_feature_names_out())
        elif isinstance(pipe, Pipeline):
            last = pipe[-1]
            if hasattr(last, "get_feature_names_out"):
                try:
                    names.extend(last.get_feature_names_out(cols))
                except Exception:
                    names.extend(cols)
            else:
                names.extend(cols)
        else:
            names.extend(cols)
    return names


# ──────────────────────────────────────────────────────────────────────────
# Top-level API
# ──────────────────────────────────────────────────────────────────────────
def prepare_dataset(
    path: str | Path = "data/Combined_Dataset_40k.parquet",
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load parquet -> add rolling/lag features -> return (df, y).

    The returned ``df`` still has all columns (including target);
    ``y`` is extracted separately as a Series.
    """
    df = pd.read_parquet(path)
    logger.info("Loaded %s  (%d rows, %d cols)", path, len(df), len(df.columns))

    # Cast boolean cols to int
    for c in df.select_dtypes("boolean").columns:
        df[c] = df[c].astype("Int64").astype("float64")

    # Add rolling & lag features
    rl = RollingLagFeatures()
    rl.fit(df)
    df = rl.transform(df)

    # Extract target
    y = df[target].copy()

    return df, y


def build_pipeline(
    parquet_path: str | Path = "data/Combined_Dataset_40k.parquet",
    target: str = TARGET_COL,
    save_dir: str | Path = "models",
    save_train_ready: bool = True,
) -> Tuple[np.ndarray, pd.Series, List[str], ColumnTransformer]:
    """End-to-end: load → engineer → fit ColumnTransformer → return X, y.

    Parameters
    ----------
    parquet_path : path to the combined dataset.
    target : target column name.
    save_dir : directory for saved artifacts.
    save_train_ready : if True, also save ``data/train_ready.parquet``.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : Series
    feature_names : list[str]
    ct : fitted ColumnTransformer
    """
    df, y = prepare_dataset(parquet_path, target)

    # Optionally save the enriched DataFrame before transformation
    if save_train_ready:
        tr_path = Path(parquet_path).parent / "train_ready.parquet"
        df.to_parquet(tr_path, index=False)
        logger.info("Saved train-ready frame → %s  (%d rows)", tr_path, len(df))

    # Build & fit ColumnTransformer
    ct = build_column_transformer()
    X = ct.fit_transform(df)
    feature_names = get_feature_names(ct)

    logger.info("X shape: %s   features: %d", X.shape, len(feature_names))

    # Persist pipeline
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    pipeline_path = save_dir / "preprocess_pipeline.joblib"
    joblib.dump(ct, pipeline_path)
    logger.info("Pipeline saved → %s", pipeline_path)

    return X, y, feature_names, ct


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    X, y, feat_names, ct = build_pipeline()

    print(f"\n{'='*60}")
    print(f"X shape        : {X.shape}")
    print(f"y shape        : {y.shape}")
    print(f"NaN count in X : {np.isnan(X).sum()}")
    print(f"Features ({len(feat_names)}):")
    for i, fn in enumerate(feat_names, 1):
        print(f"  {i:3d}. {fn}")
    print(f"{'='*60}")
