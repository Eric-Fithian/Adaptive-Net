# Create a single, self-contained script with everything I ran (plus light refactoring)
"""
Correlation Analysis Pipeline
-----------------------------
Computes Spearman correlations (pairwise deletion) between all metric columns and
delta_test_loss_at_h* targets, applies Benjamini–Hochberg FDR, exports CSVs, and
produces summary plots. Includes a small family analysis for post_activation_grad_mean
sub-metrics and a utility to print top-k metrics for chosen horizons.

Usage:
    python analysis_pipeline.py --csv /path/to/correlation_experiment_results_synthetic_linear.csv

Outputs (written next to the input CSV by default if permissions allow; here, saved to /mnt/data):
    - spearman_rho_by_metric_by_horizon.csv
    - spearman_fdr_q_by_metric_by_horizon.csv
    - summary_by_horizon.csv
    - summary_by_metric.csv
    - top10_metrics_per_horizon.csv
    - heatmap_top20_metrics.png
    - effect_size_vs_horizon.png
    - n_significant_per_horizon.png
    - rho_subfamilies_vs_h.png
"""

import argparse
import os
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

# -------------------------
# Helpers
# -------------------------

def _hnum(col: str) -> int:
    try:
        return int(col.split("_h")[-1])
    except Exception:
        return 10**9

def zscore_nan(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score with NaN support; returns (Z, mean, std)."""
    m = np.nanmean(a, axis=0)
    s = np.nanstd(a, axis=0, ddof=1)
    s[s == 0] = np.nan
    z = (a - m) / s
    return z, m, s

def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction on an array of p-values (any shape)."""
    p_flat = p.flatten()
    valid = np.isfinite(p_flat)
    pv = p_flat[valid]
    m = pv.size
    if m == 0:
        return np.full_like(p, np.nan)
    order = np.argsort(pv)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = pv * m / ranks
    # Enforce monotonicity
    q_monotone = np.minimum.accumulate(q[order[::-1]])[::-1]
    q_adj = np.empty_like(pv)
    q_adj[order] = np.minimum(q_monotone, 1.0)
    q_full = np.full_like(p_flat, np.nan, dtype=float)
    q_full[valid] = q_adj
    return q_full.reshape(p.shape)

def compute_spearman_pairwise(df: pd.DataFrame, metric_cols: List[str], h_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Spearman rho and FDR q-values between metrics and horizons."""
    ranked = df[metric_cols + h_cols].rank(method="average")
    X = ranked[metric_cols].to_numpy(dtype=float)
    Y = ranked[h_cols].to_numpy(dtype=float)

    Zx, _, _ = zscore_nan(X)
    Zy, _, _ = zscore_nan(Y)

    n_metrics = Zx.shape[1]
    n_h = Zy.shape[1]

    rho = np.full((n_metrics, n_h), np.nan, dtype=float)
    n_eff = np.zeros((n_metrics, n_h), dtype=int)

    # Pearson on z-scored ranks = Spearman
    valid_y_all = ~np.isnan(Zy)
    for j in range(n_h):
        yj = Zy[:, j]
        valid_y = ~np.isnan(yj)
        for i in range(n_metrics):
            xi = Zx[:, i]
            mask = valid_y & ~np.isnan(xi)
            n = int(mask.sum())
            if n >= 10:
                r = float(np.nanmean(xi[mask] * yj[mask]))
                r = max(min(r, 1.0), -1.0)
                rho[i, j] = r
                n_eff[i, j] = n

    # p-values via t approx
    pvals = np.full_like(rho, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = rho * np.sqrt(np.maximum(n_eff - 2, 0) / np.maximum(1.0 - rho**2, 1e-12))
    for i in range(n_metrics):
        for j in range(n_h):
            if np.isfinite(rho[i, j]) and n_eff[i, j] >= 10:
                tstat = numerator[i, j]
                dfree = max(n_eff[i, j] - 2, 1)
                p = 2.0 * (1.0 - t_dist.cdf(abs(tstat), dfree))
                pvals[i, j] = p

    qvals = bh_fdr(pvals)

    rho_df = pd.DataFrame(rho, index=metric_cols, columns=h_cols)
    n_df = pd.DataFrame(n_eff, index=metric_cols, columns=h_cols)
    q_df = pd.DataFrame(qvals, index=metric_cols, columns=h_cols)
    return rho_df, q_df, n_df

def summarize_results(rho_df: pd.DataFrame, q_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    abs_rho = rho_df.abs()

    # Per-horizon summary
    sig_per_h = (q_df < 0.05).sum(axis=0).rename("n_significant")
    median_abs_rho_per_h = abs_rho.median(axis=0).rename("median_|rho|")
    p95_abs_rho_per_h = abs_rho.quantile(0.95, axis=0).rename("p95_|rho|")
    summary_per_h = pd.concat([sig_per_h, median_abs_rho_per_h, p95_abs_rho_per_h], axis=1)
    summary_per_h.index.name = "horizon"

    # Per-metric robustness
    sig_counts_per_metric = (q_df < 0.05).sum(axis=1).rename("n_horizons_q<0.05")
    mean_abs_rho_per_metric = abs_rho.mean(axis=1).rename("mean_|rho|")
    max_abs_rho_per_metric = abs_rho.max(axis=1).rename("max_|rho|")
    argmax_h = abs_rho.values.argmax(axis=1)
    h_cols = rho_df.columns.tolist()
    best_h_per_metric = pd.Series([h_cols[k] if np.isfinite(max_abs_rho_per_metric.iloc[i]) else np.nan
                                   for i, k in enumerate(argmax_h)],
                                  index=abs_rho.index, name="h_of_max")
    summary_per_metric = pd.concat(
        [sig_counts_per_metric, mean_abs_rho_per_metric, max_abs_rho_per_metric, best_h_per_metric],
        axis=1
    ).sort_values(["n_horizons_q<0.05", "mean_|rho|", "max_|rho|"], ascending=[False, False, False])

    return summary_per_h, summary_per_metric

def export_topk_per_horizon(rho_df: pd.DataFrame, q_df: pd.DataFrame, n_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    records = []
    for h in rho_df.columns:
        msk = (q_df[h] < 0.05) & rho_df[h].abs().notna()
        ranked_metrics = rho_df[h][msk].abs().sort_values(ascending=False).head(k).index.tolist()
        for rank, m in enumerate(ranked_metrics, start=1):
            records.append({
                "horizon": h,
                "rank": rank,
                "metric": m,
                "rho": rho_df.at[m, h],
                "q_value": q_df.at[m, h],
                "n": int(n_df.at[m, h])
            })
    return pd.DataFrame(records)

def plot_top20_heatmap(rho_df: pd.DataFrame, summary_per_metric: pd.DataFrame, outpath: str):
    top20 = summary_per_metric.head(20).index.tolist()
    h_cols = rho_df.columns.tolist()
    hnums = [_hnum(h) for h in h_cols]
    H = rho_df.loc[top20, h_cols]
    plt.figure(figsize=(12, 8))
    plt.imshow(H, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Spearman rho')
    plt.yticks(ticks=np.arange(len(top20)), labels=top20)
    plt.xticks(ticks=np.arange(len(h_cols)), labels=hnums, rotation=90)
    plt.title("Top 20 metrics by mean |rho| across horizons")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_effect_sizes(summary_per_h: pd.DataFrame, outpath: str):
    hnums = [_hnum(h) for h in summary_per_h.index.tolist()]
    plt.figure(figsize=(9, 5))
    plt.plot(hnums, summary_per_h["median_|rho|"].values, label="Median |rho|")
    plt.plot(hnums, summary_per_h["p95_|rho|"].values, label="P95 |rho|")
    plt.title("Effect size vs horizon")
    plt.xlabel("Horizon (h)")
    plt.ylabel("|rho|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_n_significant(summary_per_h: pd.DataFrame, outpath: str):
    hnums = [_hnum(h) for h in summary_per_h.index.tolist()]
    plt.figure(figsize=(10, 5))
    plt.bar(hnums, summary_per_h["n_significant"].values)
    plt.title("Number of significant metrics per horizon (FDR q<0.05)")
    plt.xlabel("Horizon (h)")
    plt.ylabel("# metrics")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_subfamilies(rho_df: pd.DataFrame, outpath: str):
    h_cols = [c for c in rho_df.columns if c.startswith("delta_test_loss_at_h")]
    hnums = np.array([_hnum(c) for c in h_cols])
    fam_idx = rho_df.index.str.contains("post_activation_grad_mean")
    rho_fam = rho_df.loc[fam_idx, h_cols]

    def rows(pattern):
        return rho_fam[rho_fam.index.str.contains(pattern, regex=True, na=False)]

    groups = {
        "q90": rows(r"__q90__"),
        "q10": rows(r"__q10__"),
        "min": rows(r"__min__"),
        "mad": rows(r"__mad__"),
        "abs2diff": rows(r"abs_mean_second_diff"),
    }
    avg_rhos = {k: v.mean(axis=0).values for k, v in groups.items() if not v.empty}

    plt.figure(figsize=(10, 6))
    for label, vals in avg_rhos.items():
        plt.plot(hnums, vals, label=label)
    plt.axhline(0, linestyle="--")
    plt.title("Average rho by subfamily (post_activation_grad_mean)")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Spearman rho")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def top_for(topk_df: pd.DataFrame, hnum: int, k: int = 8) -> pd.DataFrame:
    hcol = f"delta_test_loss_at_h{hnum}"
    sub = topk_df[topk_df["horizon"] == hcol].copy()
    return sub.sort_values("rank").head(k)[["metric","rho","q_value","n"]]

# -------------------------
# Main
# -------------------------

def main(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)

    # Identify horizons and metric cols
    h_cols = [c for c in df.columns if c.startswith("delta_test_loss_at_h")]
    h_cols = sorted(h_cols, key=_hnum)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in num_cols if c not in h_cols]

    exclude_like = {"regime_name","starting_width","init_id","neuron_idx","seed","trial","epoch","step"}
    metric_cols = [c for c in metric_cols if c not in exclude_like]

    # Core stats
    rho_df, q_df, n_df = compute_spearman_pairwise(df, metric_cols, h_cols)

    # Summaries
    summary_per_h, summary_per_metric = summarize_results(rho_df, q_df)

    # Top-K per horizon
    topk_df = export_topk_per_horizon(rho_df, q_df, n_df, k=10)

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    rho_csv = os.path.join(out_dir, "spearman_rho_by_metric_by_horizon.csv")
    q_csv = os.path.join(out_dir, "spearman_fdr_q_by_metric_by_horizon.csv")
    summary_h_csv = os.path.join(out_dir, "summary_by_horizon.csv")
    summary_m_csv = os.path.join(out_dir, "summary_by_metric.csv")
    topk_csv = os.path.join(out_dir, "top10_metrics_per_horizon.csv")

    rho_df.to_csv(rho_csv)
    q_df.to_csv(q_csv)
    summary_per_h.to_csv(summary_h_csv)
    summary_per_metric.to_csv(summary_m_csv)
    topk_df.to_csv(topk_csv, index=False)

    # Plots
    plot_top20_heatmap(rho_df, summary_per_metric, os.path.join(out_dir, "heatmap_top20_metrics.png"))
    plot_effect_sizes(summary_per_h, os.path.join(out_dir, "effect_size_vs_horizon.png"))
    plot_n_significant(summary_per_h, os.path.join(out_dir, "n_significant_per_horizon.png"))
    plot_subfamilies(rho_df, os.path.join(out_dir, "rho_subfamilies_vs_h.png"))

    # Print a quick digest for a few horizons
    for h in [0, 1, 2, 4, 5, 10, 20, 30, 40, 49]:
        print(f"\n=== Top metrics at delta_test_loss_at_h{h} ===")
        print(top_for(topk_df, h, 8).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to correlation_experiment_results_synthetic_linear.csv")
    parser.add_argument("--out_dir", type=str, help="Output directory for CSVs and plots")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.csv) + "/analysis"
    main(args.csv, out_dir)