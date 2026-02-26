"""
Causal-effect estimation of enforcement actions on network latency.

Methods
-------
1. Pre / Post window analysis  – mean & p95 latency changes, restoration time.
2. Difference-in-Differences (DiD) – treatment vs. control groups.
3. Propensity-Score Matching (PSM) – matched causal ATE estimates.

Outputs
-------
- reports/enforcement_analysis.md
- reports/enforcement_effects_summary.csv
- figures/enforcement_pre_post.png
- figures/enforcement_ate_forest.png
- figures/enforcement_effectiveness_heatmap.png
"""

from __future__ import annotations

import pathlib
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

# ── analysis parameters ──────────────────────────────────────────
WINDOW_NS: int = 100_000          # ±100 μs (~100 rows each side)
BOOTSTRAP_N: int = 2_000
ALPHA: float = 0.05
RNG_SEED: int = 42

# ── PSM covariates ───────────────────────────────────────────────
PSM_COVARIATES = [
    "cpu_usage",
    "memory_usage",
    "queue_occupancy",
    "trust_score",
    "packet_size_bytes",
    "flow_priority",
    "traffic_deviation",
    "behavior_anomaly_score",
    "queue_delay_us",
]

# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load full Combined_Dataset.csv sorted by timestamp."""
    df = pd.read_csv(DATA / "Combined_Dataset.csv")
    df.sort_values("timestamp_ns", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────
# 1.  Pre / Post Window Analysis
# ─────────────────────────────────────────────────────────────────

def _window_indices(sorted_ts: np.ndarray, centre: int, half_w: int):
    """Return (pre_lo, pre_hi, post_lo, post_hi) index ranges."""
    lo = np.searchsorted(sorted_ts, centre - half_w, side="left")
    mid = np.searchsorted(sorted_ts, centre, side="left")
    hi = np.searchsorted(sorted_ts, centre + half_w, side="right")
    return lo, mid, mid + 1, hi          # pre=[lo:mid), post=[mid+1:hi)


def _restoration_time(lat: np.ndarray, ts: np.ndarray,
                      pre_mean: float, post_lo: int, post_hi: int,
                      t_event: int, smooth: int = 5) -> float:
    """Time (ns) until a 5‐point rolling mean drops to pre‐event level."""
    seg_lat = lat[post_lo:post_hi]
    seg_ts = ts[post_lo:post_hi]
    if len(seg_lat) < smooth:
        return np.nan
    for i in range(smooth - 1, len(seg_lat)):
        if np.mean(seg_lat[i - smooth + 1 : i + 1]) <= pre_mean:
            return float(seg_ts[i] - t_event)
    return float(seg_ts[-1] - t_event) if len(seg_ts) else np.nan


def pre_post_analysis(df: pd.DataFrame,
                      window_ns: int = WINDOW_NS,
                      min_pts: int = 5) -> pd.DataFrame:
    """Per-enforcement-event pre/post latency statistics."""
    enforced = df[df["action_type"].notna()]
    ts = df["timestamp_ns"].values
    lat = df["latency_us"].values

    rows: list[dict] = []
    for _, r in enforced.iterrows():
        t = int(r["timestamp_ns"])
        lo, mid, post_lo, hi = _window_indices(ts, t, window_ns)

        pre_lat = lat[lo:mid]
        post_lat = lat[post_lo:hi]
        if len(pre_lat) < min_pts or len(post_lat) < min_pts:
            continue

        pre_m = float(np.mean(pre_lat))
        post_m = float(np.mean(post_lat))
        pre_p95 = float(np.percentile(pre_lat, 95))
        post_p95 = float(np.percentile(post_lat, 95))

        rows.append(
            {
                "event_id": r["event_id"],
                "action_type": r["action_type"],
                "attack_type": r["attack_type"],
                "severity_level": r.get("severity_level", np.nan),
                "timestamp_ns": t,
                "pre_mean_latency": pre_m,
                "post_mean_latency": post_m,
                "pre_p95_latency": pre_p95,
                "post_p95_latency": post_p95,
                "pre_n": len(pre_lat),
                "post_n": len(post_lat),
                "delta_mean": post_m - pre_m,
                "delta_p95": post_p95 - pre_p95,
                "restoration_time_ns": _restoration_time(
                    lat, ts, pre_m, post_lo, hi, t
                ),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# 2.  Difference-in-Differences
# ─────────────────────────────────────────────────────────────────

def _bootstrap_ci(a: np.ndarray, b: np.ndarray,
                  n_boot: int = BOOTSTRAP_N,
                  alpha: float = ALPHA) -> tuple[float, float]:
    rng = np.random.RandomState(RNG_SEED)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (
            np.mean(rng.choice(a, len(a), replace=True))
            - np.mean(rng.choice(b, len(b), replace=True))
        )
    return (
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )


def did_analysis(df: pd.DataFrame,
                 window_ns: int = WINDOW_NS) -> pd.DataFrame:
    """DiD estimate of ATE per action_type with bootstrap 95 % CI."""
    enforced = df[df["action_type"].notna()]
    ts = df["timestamp_ns"].values
    lat = df["latency_us"].values
    ts_min, ts_max = int(ts.min()), int(ts.max())

    results: list[dict] = []
    for atype in sorted(enforced["action_type"].unique()):
        treat_deltas: list[float] = []
        ctrl_deltas: list[float] = []

        for _, r in enforced[enforced["action_type"] == atype].iterrows():
            t = int(r["timestamp_ns"])
            lo, mid, plo, hi = _window_indices(ts, t, window_ns)
            pre = lat[lo:mid]
            post = lat[plo:hi]
            if len(pre) < 5 or len(post) < 5:
                continue
            treat_deltas.append(float(np.mean(post) - np.mean(pre)))

            # Control: shift 5× window away (prefer left, fallback right)
            t_ctrl = t - 5 * window_ns
            if t_ctrl < ts_min:
                t_ctrl = t + 5 * window_ns
            if t_ctrl > ts_max:
                continue
            c_lo, c_mid, c_plo, c_hi = _window_indices(ts, t_ctrl, window_ns)
            c_pre = lat[c_lo:c_mid]
            c_post = lat[c_plo:c_hi]
            if len(c_pre) >= 5 and len(c_post) >= 5:
                ctrl_deltas.append(float(np.mean(c_post) - np.mean(c_pre)))

        t_arr = np.asarray(treat_deltas)
        c_arr = np.asarray(ctrl_deltas)
        if len(t_arr) < 5 or len(c_arr) < 5:
            continue

        ate = float(np.mean(t_arr) - np.mean(c_arr))
        ci_lo, ci_hi = _bootstrap_ci(t_arr, c_arr)
        _, p_val = stats.ttest_ind(t_arr, c_arr, equal_var=False)

        results.append(
            {
                "method": "DiD",
                "action_type": atype,
                "ate_estimate": ate,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(p_val),
                "n_treated": len(t_arr),
                "n_control": len(c_arr),
            }
        )
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────
# 3.  Propensity-Score Matching
# ─────────────────────────────────────────────────────────────────

def psm_analysis(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """PSM ATE per action_type.  Returns (ate_df, treated_df, matched_ctrl_df)."""
    df2 = df.copy()
    df2["treated"] = df2["action_type"].notna().astype(int)
    df2 = df2.dropna(subset=PSM_COVARIATES)

    X = df2[PSM_COVARIATES].values
    y = df2["treated"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=RNG_SEED)
    lr.fit(X_sc, y)
    df2["propensity"] = lr.predict_proba(X_sc)[:, 1]

    treated = df2[df2["treated"] == 1].reset_index(drop=True)
    control = df2[df2["treated"] == 0].reset_index(drop=True)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(control[["propensity"]].values)
    _, idx = nn.kneighbors(treated[["propensity"]].values)
    matched_ctrl = control.iloc[idx.flatten()].reset_index(drop=True)

    results: list[dict] = []
    for atype in sorted(treated["action_type"].unique()):
        mask = treated["action_type"] == atype
        t_lat = treated.loc[mask, "latency_us"].values
        c_lat = matched_ctrl.loc[mask, "latency_us"].values

        ate = float(np.mean(t_lat) - np.mean(c_lat))
        ci = _bootstrap_ci_paired(t_lat, c_lat)
        _, p_val = stats.ttest_ind(t_lat, c_lat, equal_var=False)

        results.append(
            {
                "method": "PSM",
                "action_type": atype,
                "ate_estimate": ate,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                "p_value": float(p_val),
                "n_treated": int(mask.sum()),
                "n_control": int(mask.sum()),
            }
        )
    return pd.DataFrame(results), treated, matched_ctrl


def _bootstrap_ci_paired(a: np.ndarray, b: np.ndarray,
                          n_boot: int = BOOTSTRAP_N,
                          alpha: float = ALPHA) -> tuple[float, float]:
    rng = np.random.RandomState(RNG_SEED)
    n = min(len(a), len(b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = np.mean(a[idx]) - np.mean(b[idx])
    return (
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )


# ─────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────

def plot_pre_post(pp_df: pd.DataFrame,
                  save_dir: pathlib.Path = FIGURES) -> pathlib.Path:
    """Bar chart: pre vs post mean & p95 latency + restoration time."""
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    agg = (
        pp_df.groupby("action_type")
        .agg(
            pre_mean=("pre_mean_latency", "mean"),
            post_mean=("post_mean_latency", "mean"),
            pre_p95=("pre_p95_latency", "mean"),
            post_p95=("post_p95_latency", "mean"),
            restoration=("restoration_time_ns", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(agg))
    w = 0.35

    # Mean latency
    axes[0].bar(x - w / 2, agg["pre_mean"], w, label="Pre", color="steelblue")
    axes[0].bar(x + w / 2, agg["post_mean"], w, label="Post", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agg["action_type"], rotation=15)
    axes[0].set_ylabel("Mean Latency (μs)")
    axes[0].set_title("Pre vs Post – Mean Latency")
    axes[0].legend()

    # P95
    axes[1].bar(x - w / 2, agg["pre_p95"], w, label="Pre", color="steelblue")
    axes[1].bar(x + w / 2, agg["post_p95"], w, label="Post", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agg["action_type"], rotation=15)
    axes[1].set_ylabel("P95 Latency (μs)")
    axes[1].set_title("Pre vs Post – P95 Latency")
    axes[1].legend()

    # Restoration
    axes[2].bar(x, agg["restoration"] / 1000, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(agg["action_type"], rotation=15)
    axes[2].set_ylabel("Restoration Time (μs)")
    axes[2].set_title("Mean Restoration Time")

    plt.tight_layout()
    path = save_dir / "enforcement_pre_post.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_did_psm(did_df: pd.DataFrame,
                 psm_df: pd.DataFrame,
                 save_dir: pathlib.Path = FIGURES) -> pathlib.Path:
    """Forest plot of ATE estimates with 95 % CI."""
    save_dir = pathlib.Path(save_dir)
    combined = pd.concat([did_df, psm_df], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"DiD": "steelblue", "PSM": "coral"}
    y_pos = 0
    y_labels, y_ticks = [], []
    drawn_labels: set[str] = set()

    for atype in sorted(combined["action_type"].unique()):
        for method in ["DiD", "PSM"]:
            row = combined[
                (combined["action_type"] == atype) & (combined["method"] == method)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            lbl = method if method not in drawn_labels else None
            drawn_labels.add(method)
            ax.errorbar(
                r["ate_estimate"],
                y_pos,
                xerr=[
                    [r["ate_estimate"] - r["ci_lower"]],
                    [r["ci_upper"] - r["ate_estimate"]],
                ],
                fmt="o",
                capsize=4,
                color=colors[method],
                label=lbl,
            )
            y_labels.append(f"{atype}\n({method})")
            y_ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.5

    ax.axvline(0, color="gray", ls="--", alpha=0.7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("ATE  (Δ Latency μs)")
    ax.set_title("Causal Effect Estimates – DiD & PSM per Action Type")
    ax.legend()
    plt.tight_layout()

    path = save_dir / "enforcement_ate_forest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_effectiveness_heatmap(pp_df: pd.DataFrame,
                               save_dir: pathlib.Path = FIGURES) -> pathlib.Path:
    """Heatmap of Δ mean latency by action_type × attack_type."""
    save_dir = pathlib.Path(save_dir)
    pivot = pp_df.pivot_table(
        values="delta_mean", index="action_type", columns="attack_type", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Δ Mean Latency (μs)")
    ax.set_title("Effectiveness:  Δ Latency by Action × Attack Type")
    plt.tight_layout()

    path = save_dir / "enforcement_effectiveness_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────

def _df_to_md(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body: list[str] = []
    for _, r in df.iterrows():
        vals = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([hdr, sep, *body])


def save_results(
    pp_df: pd.DataFrame,
    did_df: pd.DataFrame,
    psm_df: pd.DataFrame,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write CSV + Markdown report.  Returns (csv_path, md_path)."""
    REPORTS.mkdir(exist_ok=True)

    # ── CSV ──────────────────────────────────────────────────────
    combined = pd.concat([did_df, psm_df], ignore_index=True)
    csv_path = REPORTS / "enforcement_effects_summary.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    # ── aggregated tables ────────────────────────────────────────
    pp_agg = (
        pp_df.groupby("action_type")
        .agg(
            n_events=("event_id", "count"),
            pre_mean=("pre_mean_latency", "mean"),
            post_mean=("post_mean_latency", "mean"),
            delta_mean=("delta_mean", "mean"),
            delta_mean_std=("delta_mean", "std"),
            delta_p95=("delta_p95", "mean"),
            mean_restoration_us=("restoration_time_ns", lambda x: x.mean() / 1000),
        )
        .reset_index()
    )

    pp_attack = (
        pp_df.groupby(["action_type", "attack_type"])
        .agg(n=("event_id", "count"), delta_mean=("delta_mean", "mean"),
             delta_p95=("delta_p95", "mean"))
        .reset_index()
    )

    pp_sev = (
        pp_df.dropna(subset=["severity_level"])
        .groupby(["action_type", "severity_level"])
        .agg(n=("event_id", "count"), delta_mean=("delta_mean", "mean"))
        .reset_index()
    )

    # ── recommendations ──────────────────────────────────────────
    recs_atk: list[dict] = []
    for atk in pp_attack["attack_type"].unique():
        sub = pp_attack[pp_attack["attack_type"] == atk]
        best = sub.loc[sub["delta_mean"].idxmin()]
        recs_atk.append(
            {"attack_type": atk, "recommended_action": best["action_type"],
             "delta_mean": best["delta_mean"], "delta_p95": best["delta_p95"]}
        )
    recs_atk_df = pd.DataFrame(recs_atk)

    recs_sev: list[dict] = []
    for sev in pp_sev["severity_level"].unique():
        sub = pp_sev[pp_sev["severity_level"] == sev]
        best = sub.loc[sub["delta_mean"].idxmin()]
        recs_sev.append(
            {"severity_level": sev, "recommended_action": best["action_type"],
             "delta_mean": best["delta_mean"]}
        )
    recs_sev_df = pd.DataFrame(recs_sev)

    # ── markdown ─────────────────────────────────────────────────
    md = textwrap.dedent(f"""\
    # Enforcement Action Effectiveness Analysis

    ## 1. Pre / Post Window Analysis  (±{WINDOW_NS / 1000:.0f} μs)

    ### 1.1  Per Action Type
    {_df_to_md(pp_agg.round(4))}

    ### 1.2  Per Action Type × Attack Type
    {_df_to_md(pp_attack.round(4))}

    ## 2. Difference-in-Differences (DiD)

    {_df_to_md(did_df.round(4))}

    **Interpretation:** A negative ATE means enforcement *reduced* latency
    relative to the control window; a positive value means transient overhead.

    ## 3. Propensity-Score Matching (PSM)

    {_df_to_md(psm_df.round(4))}

    ## 4. Combined ATE Summary

    {_df_to_md(combined.round(4))}

    ## 5. Recommendations

    ### Best Action per Attack Type  (lowest Δ mean latency)
    {_df_to_md(recs_atk_df.round(4))}

    ### Best Action per Severity Level
    {_df_to_md(recs_sev_df.round(4))}

    ## 6. Methodology Notes

    | Step | Detail |
    | --- | --- |
    | Window | ±{WINDOW_NS / 1000:.0f} μs around each enforcement timestamp |
    | DiD control | Same-size window shifted 5× away from treatment event |
    | PSM covariates | {', '.join(PSM_COVARIATES)} |
    | CI method | Percentile bootstrap ({BOOTSTRAP_N} resamples, α={ALPHA}) |
    | p-value | Welch's two-sample t-test |

    > **Caveat:** The dataset is *synthetic* with near-uniform latency, so large
    > causal effects are not expected.  The methodology provides a sound framework
    > that transfers directly to production data.
    """)

    md_path = REPORTS / "enforcement_analysis.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  Saved {md_path}")
    return csv_path, md_path


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    df = load_data()

    print("\n── 1. Pre / Post Window Analysis ───────────────")
    pp_df = pre_post_analysis(df)
    print(f"   {len(pp_df)} events with sufficient window data")
    print(
        pp_df.groupby("action_type")[["delta_mean", "delta_p95"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n── 2. Difference-in-Differences ─────────────────")
    did_df = did_analysis(df)
    print(did_df.to_string(index=False))

    print("\n── 3. Propensity-Score Matching ──────────────────")
    psm_df, _, _ = psm_analysis(df)
    print(psm_df.to_string(index=False))

    print("\n── 4. Saving figures ────────────────────────────")
    plot_pre_post(pp_df)
    plot_did_psm(did_df, psm_df)
    plot_effectiveness_heatmap(pp_df)

    print("\n── 5. Saving reports ────────────────────────────")
    save_results(pp_df, did_df, psm_df)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
