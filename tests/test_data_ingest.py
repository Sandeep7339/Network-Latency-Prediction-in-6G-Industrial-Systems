"""Unit tests for data ingestion pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.load_data import (  # noqa: E402
    EXPECTED_FILES,
    load_all_data,
    to_datetime_ns,
)


# ---------------------------------------------------------------------------
# Tests for load_all_data
# ---------------------------------------------------------------------------
class TestLoadAllData:
    """Verify load_all_data returns the expected structure."""

    @pytest.fixture(scope="class")
    def datasets(self):
        return load_all_data(ROOT / "data")

    def test_returns_dict(self, datasets):
        assert isinstance(datasets, dict)

    def test_all_expected_keys_present(self, datasets):
        for key in EXPECTED_FILES:
            assert key in datasets, f"Missing dataset key: {key}"

    def test_values_are_dataframes(self, datasets):
        for key, df in datasets.items():
            assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"

    def test_non_empty(self, datasets):
        for key, df in datasets.items():
            assert len(df) > 0, f"{key} is empty"


# ---------------------------------------------------------------------------
# Tests for to_datetime_ns
# ---------------------------------------------------------------------------
class TestToDatetimeNs:
    """Verify timestamp conversion to datetime."""

    def test_converts_int_column(self):
        df = pd.DataFrame({"timestamp_ns": [1_000_000_000, 2_000_000_000]})
        result = to_datetime_ns(df, "timestamp_ns")
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_ns"])

    def test_converts_string_column(self):
        df = pd.DataFrame({"timestamp_ns": ["2025-01-01", "2025-06-15"]})
        result = to_datetime_ns(df, "timestamp_ns")
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp_ns"])

    def test_missing_column_is_noop(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = to_datetime_ns(df, "timestamp_ns")
        assert "timestamp_ns" not in result.columns

    def test_utc_localized(self):
        df = pd.DataFrame({"timestamp_ns": [1_000_000_000]})
        result = to_datetime_ns(df, "timestamp_ns", utc=True)
        assert result["timestamp_ns"].dt.tz is not None


# ---------------------------------------------------------------------------
# Test data profile
# ---------------------------------------------------------------------------
class TestDataProfile:
    """Verify that running load_data.py produces a data profile."""

    def test_profile_json_exists(self):
        profile_path = ROOT / "data" / "data_profile.json"
        if not profile_path.exists():
            # Generate it
            subprocess.run(
                [sys.executable, str(ROOT / "src" / "data" / "load_data.py")],
                cwd=str(ROOT),
                check=True,
            )
        assert profile_path.exists(), "data_profile.json was not created"

        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        for key in EXPECTED_FILES:
            assert key in profile, f"Profile missing key: {key}"
            assert "rows" in profile[key]
            assert "column_details" in profile[key]


# ---------------------------------------------------------------------------
# Test combined dataset (runs the script if needed)
# ---------------------------------------------------------------------------
class TestCombinedDataset:
    """Verify that the combined 40k dataset can be built."""

    PARQUET_PATH = ROOT / "data" / "Combined_Dataset_40k.parquet"
    CSV_PATH = ROOT / "data" / "Combined_Dataset_40k.csv"

    @pytest.fixture(scope="class", autouse=True)
    def ensure_combined_exists(self):
        """Run the build script if the parquet output doesn't already exist."""
        if not self.PARQUET_PATH.exists():
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "make_combined_40k.py")],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.fail(
                    f"make_combined_40k.py failed:\n{result.stdout}\n{result.stderr}"
                )

    def test_parquet_exists(self):
        assert self.PARQUET_PATH.exists(), "Combined parquet file not found"

    def test_csv_exists(self):
        assert self.CSV_PATH.exists(), "Combined CSV file not found"

    def test_row_count_approximately_40k(self):
        df = pd.read_parquet(self.PARQUET_PATH)
        assert 35_000 <= len(df) <= 45_000, (
            f"Expected ~40k rows, got {len(df)}"
        )

    def test_has_key_columns(self):
        df = pd.read_parquet(self.PARQUET_PATH)
        for col in ["timestamp_ns", "src_device_id", "traffic_type", "latency_us"]:
            assert col in df.columns, f"Missing expected column: {col}"
