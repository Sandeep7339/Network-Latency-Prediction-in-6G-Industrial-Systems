#!/usr/bin/env python
"""
run_eda.py — Execute the EDA notebook logic headlessly, generate all figures
and print correlation / summary stats needed for eda_summary.md.
"""
import sys, warnings, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

def savefig(name):
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path.relative_to(PROJECT_ROOT)}")

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_parquet(PROJECT_ROOT / "data" / "Combined_Dataset_40k.parquet")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

if df["timestamp_ns"].dtype != "datetime64[ns, UTC]":
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)

bool_cols = df.select_dtypes(include="boolean").columns.tolist()
for c in bool_cols:
    df[c + "_int"] = df[c].astype("Int64")

# ── 1. Missingness ───────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df = pd.DataFrame({"missing": missing, "pct": missing_pct}).sort_values("pct", ascending=False)
print("=== Missing values ===")
print(miss_df[miss_df["missing"] > 0].to_string() if (miss_df["missing"] > 0).any() else "No missing values!")

fig, ax = plt.subplots(figsize=(14, 5))
sample_idx = np.linspace(0, len(df)-1, 500, dtype=int)
sns.heatmap(df.iloc[sample_idx].isnull().T, cbar=False, yticklabels=True, cmap="Reds", ax=ax)
ax.set_title("Missingness Heatmap (500-row subsample)")
ax.set_xlabel("Row index (subsampled)")
savefig("01_missingness_heatmap")

# ── 2. Distinct counts ──────────────────────────────────────────────────
print("\n=== Distinct Counts ===")
for col in ["src_device_id", "dst_device_id", "device_type", "vendor",
            "firmware_version", "traffic_type", "protocol", "attack_type",
            "severity_level", "action_type", "controller_state"]:
    if col in df.columns:
        print(f"  {col:30s}  {df[col].nunique():>6}")

# ── 3. Numeric describe ─────────────────────────────────────────────────
print("\n=== Numeric Summary ===")
for col in ["latency_us", "jitter_us", "enforcement_latency_us", "control_action_delay_us"]:
    if col in df.columns:
        s = df[col].dropna()
        print(f"  {col:35s}  min={s.min():10.2f}  max={s.max():10.2f}  "
              f"median={s.median():10.2f}  mean={s.mean():10.2f}")

df["latency_ms"]  = df["latency_us"] / 1_000
df["jitter_ms"]   = df["jitter_us"]  / 1_000

# ── 4. Latency histograms ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(df["latency_us"], bins=80, edgecolor="white", color="steelblue", alpha=0.85)
axes[0].set_title("Latency (µs) — Linear"); axes[0].set_xlabel("latency_us"); axes[0].set_ylabel("Count")
axes[1].hist(df["latency_us"], bins=80, edgecolor="white", color="darkorange", alpha=0.85)
axes[1].set_yscale("log"); axes[1].set_title("Latency (µs) — Log Scale")
axes[1].set_xlabel("latency_us"); axes[1].set_ylabel("Count (log)")
plt.tight_layout(); savefig("02_latency_us_hist")

# ── 5. Jitter histogram ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(df["jitter_us"], bins=60, edgecolor="white", color="seagreen", alpha=0.85)
ax.set_title("Jitter (µs) Distribution"); ax.set_xlabel("jitter_us"); ax.set_ylabel("Count")
savefig("03_jitter_us_hist")

# ── 6. Packet size ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["packet_size_bytes"], bins=60, edgecolor="white", color="mediumpurple", alpha=0.85)
axes[0].set_title("Packet Size (bytes) Distribution"); axes[0].set_xlabel("packet_size_bytes")
order = df["traffic_type"].value_counts().index.tolist()
sns.boxplot(data=df, x="traffic_type", y="packet_size_bytes", order=order, ax=axes[1], palette="Set2")
axes[1].set_title("Packet Size by Traffic Type"); axes[1].set_xlabel("")
plt.tight_layout(); savefig("04_packet_size_dist")

# ── 7. Scatter latency vs enforcement ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter_df = df.sample(n=5000, random_state=42)
colors = scatter_df["success_flag_int"].map({1: "tab:green", 0: "tab:red"}).fillna("grey")
ax.scatter(scatter_df["latency_us"], scatter_df["enforcement_latency_us"], c=colors, alpha=0.35, s=8)
ax.set_xlabel("latency_us"); ax.set_ylabel("enforcement_latency_us")
ax.set_title("Latency vs Enforcement Latency (green=success, red=fail)")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="tab:green", markersize=7, label="success=True"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="tab:red",   markersize=7, label="success=False"),
]
ax.legend(handles=legend_elements, loc="upper right")
savefig("05_latency_vs_enforcement")

# ── 8. Correlation heatmap ──────────────────────────────────────────────
num_df = df.select_dtypes(include=[np.number])
drop_ids = [c for c in num_df.columns if c.endswith("_id") or c in ("timestamp_ns", "event_id", "action_id")]
num_df = num_df.drop(columns=drop_ids, errors="ignore")
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.4, ax=ax, annot_kws={"size": 7}, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap (numeric features)")
savefig("06_correlation_heatmap")

# Top-15 correlations
print("\n=== Top-15 features correlated with latency_us ===")
if "latency_us" in corr.columns:
    top15 = corr["latency_us"].drop("latency_us", errors="ignore").abs().sort_values(ascending=False).head(15)
    for feat, val in top15.items():
        raw = corr.loc[feat, "latency_us"]
        print(f"  {feat:35s}  r = {raw:+.4f}  (|r| = {val:.4f})")

# ── 9. Device timeline ──────────────────────────────────────────────────
top_device = df["src_device_id"].value_counts().idxmax()
dev_df = df[df["src_device_id"] == top_device].copy()
dev_df = dev_df.sort_values("timestamp_utc").set_index("timestamp_utc")
print(f"\nDevice {top_device}: {len(dev_df)} rows")

rs = dev_df.resample("1s").agg({
    "latency_us":       "mean",
    "queue_occupancy":  "mean",
    "packet_size_bytes": "count",
}).rename(columns={"packet_size_bytes": "packet_rate"})
rs = rs.dropna(how="all")

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(rs.index, rs["latency_us"], color="steelblue", lw=0.8)
axes[0].set_ylabel("latency_us (mean)")
axes[0].set_title(f"Device {top_device} — 1-s Resampled Timeline")
axes[1].plot(rs.index, rs["queue_occupancy"], color="darkorange", lw=0.8)
axes[1].set_ylabel("queue_occupancy (mean)")
axes[2].bar(rs.index, rs["packet_rate"], width=pd.Timedelta("0.8s"), color="seagreen", alpha=0.7)
axes[2].set_ylabel("packet_rate (count/s)"); axes[2].set_xlabel("Time (UTC)")
plt.tight_layout(); savefig("07_device_timeline")

# ── 10. Attack timeline ─────────────────────────────────────────────────
attack_df = df[["timestamp_utc", "attack_type"]].copy()
attack_df = attack_df.sort_values("timestamp_utc").set_index("timestamp_utc")
attack_ts = attack_df.groupby("attack_type").resample("1s").size().unstack(level=0, fill_value=0)

fig, ax = plt.subplots(figsize=(14, 5))
palette = sns.color_palette("Set1", n_colors=attack_ts.shape[1])
for i, col in enumerate(attack_ts.columns):
    ax.plot(attack_ts.index, attack_ts[col], label=col, lw=0.6, alpha=0.8, color=palette[i])
ax.set_title("Attack Event Count by Type (1-s bins)")
ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Events / second")
ax.legend(fontsize=8, loc="upper right")
savefig("08_attack_timeline")

# ── 11. Latency by traffic_type violin ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=df, x="traffic_type", y="latency_us", palette="muted", ax=ax, inner="quart", cut=0)
ax.set_title("Latency Distribution by Traffic Type"); ax.set_xlabel("")
savefig("09_latency_by_traffic_type")

# ── 12. Queue occupancy by controller state ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="controller_state", y="queue_occupancy", palette="Pastel1", ax=ax)
ax.set_title("Queue Occupancy by Controller State"); ax.set_xlabel("")
savefig("10_queue_by_controller_state")

# ── 13. Success rate by severity ─────────────────────────────────────────
success_by_severity = df.groupby("severity_level")["success_flag_int"].mean().sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
success_by_severity.plot.barh(ax=ax, color="teal", edgecolor="white")
ax.set_xlabel("Success Rate"); ax.set_title("Enforcement Success Rate by Severity Level")
ax.set_xlim(0, 1)
for i, v in enumerate(success_by_severity):
    ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=9)
savefig("11_success_rate_by_severity")

# ── 14. Pair plot ────────────────────────────────────────────────────────
pair_cols = ["latency_us", "jitter_us", "queue_occupancy",
             "enforcement_latency_us", "control_action_delay_us"]
pair_df = df[pair_cols + ["traffic_type"]].sample(2000, random_state=42)
g = sns.pairplot(pair_df, hue="traffic_type", palette="Set2",
                 plot_kws={"s": 10, "alpha": 0.5}, diag_kws={"alpha": 0.6}, height=2.2)
g.figure.suptitle("Pair Plot — Key Metrics (2k subsample)", y=1.02)
g.savefig(FIG_DIR / "12_pair_plot.png", dpi=120)
plt.close()
print("  Saved → figures/12_pair_plot.png")

# ── Summary stats for eda_summary.md ─────────────────────────────────────
print("\n=== Additional Stats for Summary ===")
print(f"  traffic_type distribution: {dict(df['traffic_type'].value_counts())}")
print(f"  protocol distribution: {dict(df['protocol'].value_counts())}")
print(f"  controller_state distribution: {dict(df['controller_state'].value_counts())}")
print(f"  success_flag rate: {df['success_flag_int'].mean():.4f}")
print(f"  anomaly_label unique: {df['anomaly_label'].nunique()}")
print(f"  operational_state distribution: {dict(df['operational_state'].value_counts())}")
print(f"  severity_level distribution: {dict(df['severity_level'].value_counts())}")
print(f"  mobility_state distribution: {dict(df['mobility_state'].value_counts())}")
print(f"  queue_occupancy mean by controller_state:")
for state, grp in df.groupby("controller_state"):
    print(f"    {state}: {grp['queue_occupancy'].mean():.2f}")

print("\n=== All figures ===")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  {f.relative_to(PROJECT_ROOT)}  ({f.stat().st_size/1024:.0f} KB)")

print("\n✓ EDA script completed successfully.")
