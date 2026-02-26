"""
Data ingestion, type coercion, and profiling utilities.

Usage:
    from src.data.load_data import load_all_data, to_datetime_ns
    datasets = load_all_data("data/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-type mappings
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "latency_us",
    "jitter_us",
    "queue_delay_us",
    "enforcement_latency_us",
    "control_action_delay_us",
    "cycle_time_us",
    "deadline_us",
]

BOOLEAN_COLS = [
    "latency_violation",
    "jitter_violation",
    "success_flag",
    "anomaly_label",
]

# The six "core" datasets we expect in data/
EXPECTED_FILES = {
    "Network_Traffic":          "Network_Traffic.csv",
    "Device_Profile":           "Device_Profile.csv",
    "Time_Deterministic_Stats": "Time_Deterministic_Stats.csv",
    "Security_Events":          "Security_Events.csv",
    "Enforcement_Actions":      "Enforcement_Actions.csv",
    "Stabilization_Controller": "Stabilization_Controller.csv",
}


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------
def load_csv_or_excel(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Uses the file extension to choose the reader.  CSV files are read with
    ``encoding='utf-8-sig'`` so that a BOM (if present) is stripped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif ext in (".xls", ".xlsx", ".xlsm"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logger.info("Loaded %s  (%d rows, %d cols)", path.name, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------
def to_datetime_ns(
    df: pd.DataFrame,
    col: str = "timestamp_ns",
    *,
    utc: bool = True,
) -> pd.DataFrame:
    """Convert a nanosecond-integer or string timestamp column to pandas
    ``datetime64[ns, UTC]``.

    If the column is already datetime, it is left unchanged (but localised to
    UTC if *utc* is True and the column is tz-naive).

    Parameters
    ----------
    df : DataFrame
        Input (modified **in place** and also returned for chaining).
    col : str
        Column name to convert.
    utc : bool
        If True, localise / convert to UTC.

    Returns
    -------
    DataFrame
        The same DataFrame with the column converted.
    """
    if col not in df.columns:
        return df

    series = df[col]

    if pd.api.types.is_datetime64_any_dtype(series):
        if utc and series.dt.tz is None:
            df[col] = series.dt.tz_localize("UTC")
        return df

    # Try numeric (nanosecond epoch) first
    if pd.api.types.is_numeric_dtype(series):
        df[col] = pd.to_datetime(series, unit="ns", utc=utc)
    else:
        # Fall back to string parsing
        df[col] = pd.to_datetime(series, utc=utc)

    return df


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------
def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns to float64 (errors → NaN)."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _coerce_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known boolean columns to nullable boolean dtype."""
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .apply(lambda x: None if pd.isna(x) else bool(int(x)))
            )
            df[col] = df[col].astype("boolean")
    return df


# ---------------------------------------------------------------------------
# Load-all orchestrator
# ---------------------------------------------------------------------------
def load_all_data(
    data_dir: Union[str, Path] = "data/",
    *,
    coerce_types: bool = True,
    convert_timestamps: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load every expected CSV from *data_dir* and return a dict keyed by
    logical name (e.g. ``"Network_Traffic"``).

    Parameters
    ----------
    data_dir : str | Path
        Directory containing the raw CSV files.
    coerce_types : bool
        If True, apply numeric & boolean coercion.
    convert_timestamps : bool
        If True, convert ``timestamp_ns`` columns to datetime.

    Returns
    -------
    dict[str, DataFrame]
    """
    data_dir = Path(data_dir)
    datasets: Dict[str, pd.DataFrame] = {}

    for name, filename in EXPECTED_FILES.items():
        fpath = data_dir / filename
        if not fpath.exists():
            logger.warning("Expected file missing – skipping: %s", fpath)
            continue

        df = load_csv_or_excel(fpath)

        if coerce_types:
            _coerce_numeric(df)
            _coerce_boolean(df)

        if convert_timestamps and "timestamp_ns" in df.columns:
            to_datetime_ns(df, "timestamp_ns")

        datasets[name] = df

    return datasets


# ---------------------------------------------------------------------------
# Data profiling
# ---------------------------------------------------------------------------
def build_profile(datasets: Dict[str, pd.DataFrame]) -> dict:
    """Build a JSON-serialisable profile dict summarising each DataFrame."""
    profile: dict = {}
    for name, df in datasets.items():
        col_info = {}
        for col in df.columns:
            col_info[col] = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "non_null": int(df[col].notna().sum()),
            }
        profile[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_details": col_info,
        }
    return profile


def save_profile(
    datasets: Dict[str, pd.DataFrame],
    out_path: Union[str, Path] = "data/data_profile.json",
) -> Path:
    """Build a data profile and write it to *out_path*."""
    out_path = Path(out_path)
    profile = build_profile(datasets)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Data profile saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry-point (optional convenience)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    datasets = load_all_data("data/")
    save_profile(datasets, "data/data_profile.json")
    for name, df in datasets.items():
        print(f"  {name:30s}  {len(df):>8,} rows   {len(df.columns):>3} cols")
