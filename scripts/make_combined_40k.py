#!/usr/bin/env python
"""
make_combined_40k.py
====================
Build a ~40 000-row combined/sampled dataset from the six core CSVs.

Strategy
--------
1. Load & type-coerce all six tables via ``src.data.load_data``.
2. Start from **Network_Traffic** (the "spine"—has ``timestamp_ns``).
3. Left-join Device_Profile on ``src_device_id == device_id``.
4. Left-join Time_Deterministic_Stats on ``flow_id`` (if present in spine).
5. Merge-asof Security_Events within ±MERGE_TOLERANCE on ``timestamp_ns``,
   keyed by ``src_device_id == device_id``.
6. Left-join Enforcement_Actions on ``event_id``.
7. Concatenate Stabilization_Controller row-aligned (same length as spine).
8. **Stratified sample** to ~40 000 rows on (traffic_type, latency_violation)
   when those columns are present; otherwise pure random sample.
9. Save to ``data/Combined_Dataset_40k.parquet`` and ``.csv``.

Usage
-----
    python scripts/make_combined_40k.py          # defaults
    python scripts/make_combined_40k.py --rows 40000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path so ``src`` is importable ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_all_data, save_profile  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ROWS = 40_000
DEFAULT_SEED = 42
MERGE_TOLERANCE = pd.Timedelta("200ms")

# Suffix to avoid collisions on duplicate column names during merges
_SUFFIXES = ("", "_dup")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``*_dup`` columns that were produced by merge suffixes."""
    dup_cols = [c for c in df.columns if c.endswith("_dup")]
    if dup_cols:
        df = df.drop(columns=dup_cols)
    return df


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    strat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a reproducible stratified (or random) sample of *n* rows."""
    rng = np.random.RandomState(seed)

    if strat_cols:
        # Keep only strat cols that actually exist
        strat_cols = [c for c in strat_cols if c in df.columns]

    if not strat_cols:
        logger.info("No stratification columns available – using random sample.")
        return df.sample(n=min(n, len(df)), random_state=rng)

    # Build a group key
    group_key = df[strat_cols].astype(str).agg("|".join, axis=1)
    group_counts = group_key.value_counts()

    # Proportional allocation
    fractions = group_counts / len(df)
    target_per_group = (fractions * n).round().astype(int)
    # Adjust total to exactly n
    diff = n - target_per_group.sum()
    if diff != 0:
        # add/subtract from the largest group
        largest = target_per_group.idxmax()
        target_per_group[largest] += diff

    sampled_parts = []
    for gval, gcount in target_per_group.items():
        mask = group_key == gval
        pool = df.loc[mask]
        take = min(gcount, len(pool))
        sampled_parts.append(pool.sample(n=take, random_state=rng))

    result = pd.concat(sampled_parts, ignore_index=True)
    return result.sample(frac=1, random_state=rng).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_combined(
    data_dir: str | Path = "data/",
    target_rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
    merge_tolerance: pd.Timedelta = MERGE_TOLERANCE,
) -> pd.DataFrame:
    """Build the combined dataset and return it as a DataFrame."""

    logger.info("Loading all datasets from %s …", data_dir)
    ds = load_all_data(data_dir)

    # Save data profile as a side-effect
    save_profile(ds, Path(data_dir) / "data_profile.json")

    # ── 1. Spine = Network_Traffic ────────────────────────────────────────
    spine = ds["Network_Traffic"].copy()
    logger.info("Spine (Network_Traffic): %d rows", len(spine))

    # ── 2. Device_Profile (left join on src_device_id) ────────────────────
    if "Device_Profile" in ds:
        dev = ds["Device_Profile"].copy()
        # Rename device_id → src_device_id for merge
        if "device_id" in dev.columns:
            dev = dev.rename(columns={"device_id": "src_device_id"})
        spine = spine.merge(dev, on="src_device_id", how="left", suffixes=_SUFFIXES)
        spine = _deduplicate_columns(spine)
        logger.info("After Device_Profile join: %d cols", len(spine.columns))

    # ── 3. Time_Deterministic_Stats (left join on flow_id) ────────────────
    if "Time_Deterministic_Stats" in ds and "flow_id" in spine.columns:
        tds = ds["Time_Deterministic_Stats"].copy()
        # flow_id may not be unique in tds; keep first
        tds = tds.drop_duplicates(subset="flow_id", keep="first")
        spine = spine.merge(tds, on="flow_id", how="left", suffixes=_SUFFIXES)
        spine = _deduplicate_columns(spine)
        logger.info("After TDS join: %d cols", len(spine.columns))

    # ── 4. Security_Events (merge_asof on timestamp_ns) ───────────────────
    if "Security_Events" in ds and "timestamp_ns" in spine.columns:
        sec = ds["Security_Events"].copy()
        # Ensure both are sorted by timestamp_ns  
        spine = spine.sort_values("timestamp_ns").reset_index(drop=True)
        sec = sec.sort_values("timestamp_ns").reset_index(drop=True)

        # rename device_id → src_device_id for by-key merge
        if "device_id" in sec.columns:
            sec = sec.rename(columns={"device_id": "src_device_id"})

        # merge_asof requires the by-column types to match
        if "src_device_id" in sec.columns and "src_device_id" in spine.columns:
            spine["src_device_id"] = spine["src_device_id"].astype("int64")
            sec["src_device_id"] = sec["src_device_id"].astype("int64")

            spine = pd.merge_asof(
                spine,
                sec,
                on="timestamp_ns",
                by="src_device_id",
                tolerance=merge_tolerance,
                direction="nearest",
                suffixes=_SUFFIXES,
            )
            spine = _deduplicate_columns(spine)
        else:
            spine = pd.merge_asof(
                spine,
                sec,
                on="timestamp_ns",
                tolerance=merge_tolerance,
                direction="nearest",
                suffixes=_SUFFIXES,
            )
            spine = _deduplicate_columns(spine)
        logger.info("After Security_Events asof-merge: %d cols", len(spine.columns))

    # ── 5. Enforcement_Actions (left join on event_id) ────────────────────
    if "Enforcement_Actions" in ds and "event_id" in spine.columns:
        enf = ds["Enforcement_Actions"].copy()
        enf = enf.drop_duplicates(subset="event_id", keep="first")
        spine = spine.merge(enf, on="event_id", how="left", suffixes=_SUFFIXES)
        spine = _deduplicate_columns(spine)
        logger.info("After Enforcement_Actions join: %d cols", len(spine.columns))

    # ── 6. Stabilization_Controller (row-aligned concat) ──────────────────
    if "Stabilization_Controller" in ds:
        stab = ds["Stabilization_Controller"].copy()
        # Align lengths: trim or pad
        if len(stab) >= len(spine):
            stab = stab.iloc[: len(spine)].reset_index(drop=True)
        else:
            pad = pd.DataFrame(
                np.nan,
                index=range(len(spine) - len(stab)),
                columns=stab.columns,
            )
            stab = pd.concat([stab, pad], ignore_index=True)

        # Avoid column name collisions
        existing = set(spine.columns)
        rename_map = {c: c for c in stab.columns}
        for c in stab.columns:
            if c in existing:
                rename_map[c] = c + "_stab"
        stab = stab.rename(columns=rename_map)

        spine = pd.concat([spine.reset_index(drop=True), stab], axis=1)
        logger.info("After Stabilization concat: %d cols", len(spine.columns))

    logger.info("Combined shape before sampling: %s", spine.shape)

    # ── 7. Stratified sample ─────────────────────────────────────────────
    combined = _stratified_sample(
        spine,
        n=target_rows,
        seed=seed,
        strat_cols=["traffic_type", "latency_violation"],
    )
    logger.info("Sampled shape: %s", combined.shape)

    return combined


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_combined(
    df: pd.DataFrame,
    data_dir: str | Path = "data/",
    stem: str = "Combined_Dataset_40k",
) -> tuple[Path, Path]:
    """Write the combined DataFrame as parquet + CSV and return the paths."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / f"{stem}.parquet"
    csv_path = data_dir / f"{stem}.csv"

    # Convert datetime cols to int64 ns for parquet/csv compatibility
    df_out = df.copy()
    for col in df_out.select_dtypes(include=["datetimetz", "datetime64[ns]"]).columns:
        df_out[col] = df_out[col].astype("int64")

    df_out.to_parquet(parquet_path, index=False, engine="pyarrow")
    df_out.to_csv(csv_path, index=False)

    logger.info("Saved %s  (%d rows)", parquet_path, len(df))
    logger.info("Saved %s  (%d rows)", csv_path, len(df))
    return parquet_path, csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build Combined_Dataset_40k")
    parser.add_argument("--data-dir", default="data/", help="Path to data folder")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Target row count")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    combined = build_combined(
        data_dir=args.data_dir,
        target_rows=args.rows,
        seed=args.seed,
    )
    pq, csv = save_combined(combined, data_dir=args.data_dir)
    print(f"\n✓  Parquet → {pq}  ({len(combined):,} rows)")
    print(f"✓  CSV     → {csv}  ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
