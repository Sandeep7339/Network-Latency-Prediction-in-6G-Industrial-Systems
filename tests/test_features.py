"""Unit tests for the feature engineering pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_pipeline import (  # noqa: E402
    RollingLagFeatures,
    build_column_transformer,
    build_pipeline,
    prepare_dataset,
    get_feature_names,
    _all_numeric_cols,
    CAT_ONEHOT_COLS,
    CAT_ORDINAL_COLS,
    CAT_FREQ_COLS,
)

DATA_PATH = ROOT / "data" / "Combined_Dataset_40k.parquet"
PIPELINE_PATH = ROOT / "models" / "preprocess_pipeline.joblib"
TRAIN_READY_PATH = ROOT / "data" / "train_ready.parquet"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def prepared():
    """Load and prepare the full dataset (cached per test module)."""
    df, y = prepare_dataset(DATA_PATH)
    return df, y


@pytest.fixture(scope="module")
def pipeline_artifacts():
    """Build pipeline if needed, return (X, y, feature_names, ct)."""
    # Import FrequencyEncoder so pickle can resolve it during joblib.load
    from src.features.feature_pipeline import FrequencyEncoder as _FE  # noqa: F811,F401
    try:
        if PIPELINE_PATH.exists() and TRAIN_READY_PATH.exists():
            ct = joblib.load(PIPELINE_PATH)
            df = pd.read_parquet(TRAIN_READY_PATH)
            y = df["success_flag"]
            X = ct.transform(df)
            feature_names = get_feature_names(ct)
            return X, y, feature_names, ct
    except Exception:
        pass
    return build_pipeline(DATA_PATH, save_dir=ROOT / "models")


# --------------------------------------------------------------------------
# Tests: prepare_dataset
# --------------------------------------------------------------------------
class TestPrepareDataset:
    def test_returns_dataframe_and_series(self, prepared):
        df, y = prepared
        assert isinstance(df, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_rolling_columns_present(self, prepared):
        df, _ = prepared
        for w in [1, 10, 60]:
            assert f"latency_roll_mean_{w}s" in df.columns, f"Missing roll mean {w}s"
            assert f"latency_roll_std_{w}s" in df.columns, f"Missing roll std {w}s"
            assert f"packet_rate_{w}s" in df.columns, f"Missing packet_rate {w}s"

    def test_lag_columns_present(self, prepared):
        df, _ = prepared
        for lag in range(1, 11):
            assert f"latency_lag_{lag}" in df.columns, f"Missing lag {lag}"

    def test_no_nans_in_rolling_lag(self, prepared):
        df, _ = prepared
        eng_cols = [c for c in df.columns if "roll_" in c or "lag_" in c or "packet_rate_" in c]
        total_nan = df[eng_cols].isna().sum().sum()
        assert total_nan == 0, f"Found {total_nan} NaNs in rolling/lag columns"


# --------------------------------------------------------------------------
# Tests: ColumnTransformer pipeline
# --------------------------------------------------------------------------
class TestPipeline:
    def test_X_has_no_nans(self, pipeline_artifacts):
        X, _, _, _ = pipeline_artifacts
        assert np.isnan(X).sum() == 0, f"X has {np.isnan(X).sum()} NaNs"

    def test_X_shape_rows(self, pipeline_artifacts):
        X, y, _, _ = pipeline_artifacts
        assert X.shape[0] == len(y), "Row mismatch between X and y"

    def test_feature_count_matches(self, pipeline_artifacts):
        X, _, feature_names, _ = pipeline_artifacts
        assert X.shape[1] == len(feature_names), (
            f"X has {X.shape[1]} cols but {len(feature_names)} feature names"
        )

    def test_feature_count_range(self, pipeline_artifacts):
        _, _, feature_names, _ = pipeline_artifacts
        # 20 numeric-base + 9 rolling + 10 lag + ~25 one-hot + 3 ordinal + 3 freq ≈ 69
        assert 50 <= len(feature_names) <= 100, (
            f"Expected 50–100 features, got {len(feature_names)}"
        )


# --------------------------------------------------------------------------
# Tests: saved artifacts
# --------------------------------------------------------------------------
class TestSavedArtifacts:
    def test_pipeline_joblib_exists(self):
        assert PIPELINE_PATH.exists(), f"Missing {PIPELINE_PATH}"

    def test_train_ready_parquet_exists(self):
        assert TRAIN_READY_PATH.exists(), f"Missing {TRAIN_READY_PATH}"

    def test_pipeline_loads_and_transforms(self):
        ct = joblib.load(PIPELINE_PATH)
        df = pd.read_parquet(TRAIN_READY_PATH).head(50)
        X = ct.transform(df)
        assert X.shape[0] == 50
        assert np.isnan(X).sum() == 0, "NaNs after transforming sample batch"

    def test_pipeline_transform_on_single_row(self):
        ct = joblib.load(PIPELINE_PATH)
        df = pd.read_parquet(TRAIN_READY_PATH).iloc[[0]]
        X = ct.transform(df)
        assert X.shape[0] == 1
        assert np.isnan(X).sum() == 0


# --------------------------------------------------------------------------
# Tests: RollingLagFeatures transformer
# --------------------------------------------------------------------------
class TestRollingLagFeatures:
    def test_produces_expected_columns(self):
        df = pd.DataFrame({
            "timestamp_ns": np.arange(0, 100_000_000_000, 1_000_000_000),  # 100 pts, 1s apart
            "latency_us": np.random.RandomState(42).uniform(50, 150, 100),
            "packet_size_bytes": np.random.RandomState(42).randint(64, 1500, 100),
        })
        rl = RollingLagFeatures()
        rl.fit(df)
        out = rl.transform(df)
        for w in [1, 10, 60]:
            assert f"latency_roll_mean_{w}s" in out.columns
        for lag in range(1, 11):
            assert f"latency_lag_{lag}" in out.columns

    def test_no_nans_after_transform(self):
        df = pd.DataFrame({
            "timestamp_ns": np.arange(0, 20_000_000_000, 1_000_000_000),
            "latency_us": np.random.RandomState(7).uniform(50, 150, 20),
            "packet_size_bytes": np.random.RandomState(7).randint(64, 1500, 20),
        })
        rl = RollingLagFeatures()
        rl.fit(df)
        out = rl.transform(df)
        eng_cols = [c for c in out.columns if "roll_" in c or "lag_" in c or "packet_rate_" in c]
        assert out[eng_cols].isna().sum().sum() == 0
