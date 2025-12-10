"""
Correlation Analysis for FashionMNIST and CIFAR-10 Experiments
---------------------------------------------------------------
Computes Spearman correlations (pairwise deletion) between all metric columns and
delta_test_loss_at_h* targets, applies Benjamini-Hochberg FDR, exports CSVs, and
produces summary plots.

Now includes within-regime analysis to avoid confounding from network size differences.

Usage:
    python s2_analysis.py

Analyzes both FashionMNIST (mnist/) and CIFAR-10 (cifar-10/) datasets.

Outputs (saved to output_local/x1_correlation/{mnist,cifar-10}/analysis/):
    - spearman_rho_by_metric_by_horizon.csv (pooled)
    - spearman_fdr_q_by_metric_by_horizon.csv (pooled)
    - within_regime_correlations.csv (NEW)
    - summary_by_horizon.csv
    - summary_by_metric.csv
    - top10_metrics_per_horizon.csv
    - heatmap_top20_metrics.png
    - effect_size_vs_horizon.png
    - n_significant_per_horizon.png
    - rho_subfamilies_vs_h.png
    - within_regime_comparison.png (NEW)
"""

import os
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

# -------------------------
# Main
# -------------------------


def main(csv_path: str, out_dir: str):
    """Run the full correlation analysis pipeline."""
    dataset_name = Path(csv_path).stem.replace("correlation_experiment_results_", "")
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Correlation Analysis")
    print(f"{'='*60}")
    print(f"Input CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Store analysis directory in DataFrame attrs for use in plotting functions
    df.attrs['analysis_dir'] = out_dir
    print(f"Loaded {len(df)} rows")

    # Check for regime column
    if 'regime_name' not in df.columns and 'starting_width' not in df.columns:
        print("\nWARNING: No regime information found. Skipping within-regime analysis.")
        has_regime_info = False
    else:
        has_regime_info = True
        regime_col = 'regime_name' if 'regime_name' in df.columns else 'starting_width'
        regimes = df[regime_col].unique()
        print(f"\nFound {len(regimes)} regimes: {sorted(regimes)}")

    # Identify horizons and metric cols
    h_cols = [c for c in df.columns if c.startswith("delta_test_loss_at_h")]
    h_cols_pct = [c for c in df.columns if c.startswith("delta_test_loss_pct_at_h")]
    h_cols = sorted(h_cols, key=_hnum)
    h_cols_pct = sorted(h_cols_pct, key=_hnum)
    
    print(f"Found {len(h_cols)} horizon columns (raw delta): {[_hnum(h) for h in h_cols[:5]]}...")
    if h_cols_pct:
        print(f"Found {len(h_cols_pct)} horizon columns (percentage): {[_hnum(h) for h in h_cols_pct[:5]]}...")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in num_cols if c not in h_cols + h_cols_pct]

    # Exclude non-metric columns
    exclude_like = {"regime_name","starting_width","init_id","neuron_idx","seed","trial","epoch","step","action_epoch"}
    metric_cols = [c for c in metric_cols if c not in exclude_like]
    print(f"Analyzing {len(metric_cols)} metric columns")
    
    # Check if action_epoch is variable (random split epochs)
    has_variable_action_epoch = 'action_epoch' in df.columns and df['action_epoch'].nunique() > 1
    if has_variable_action_epoch:
        print(f"Variable action_epoch detected: range [{df['action_epoch'].min()}, {df['action_epoch'].max()}]")

    # ===========================
    # PRIMARY: Within-Regime Analysis
    # ===========================
    if has_regime_info:
        print(f"\n{'='*60}")
        print("WITHIN-REGIME CORRELATION ANALYSIS")
        print(f"{'='*60}")
        
        within_regime_results = compute_within_regime_correlations(
            df, metric_cols, h_cols, regime_col
        )
        
        # Save within-regime results
        within_regime_csv = os.path.join(out_dir, "within_regime_correlations.csv")
        os.makedirs(out_dir, exist_ok=True)
        within_regime_results.to_csv(within_regime_csv, index=False)
        print(f"Saved within-regime results to: {within_regime_csv}")
        
        # Print summary
        print("\nWithin-regime correlation summary:")
        print_within_regime_summary(within_regime_results)

    # ===========================
    # POOLED Analysis (all regimes together)
    # ===========================
    print(f"\n{'='*60}")
    print("POOLED CORRELATION ANALYSIS (All Regimes)")
    print(f"{'='*60}")

    # Core stats - pooled
    print("\nComputing pooled Spearman correlations (raw delta)...")
    rho_df, q_df, n_df = compute_spearman_pairwise(df, metric_cols, h_cols)

    # If percentage columns exist, also compute pooled correlations on those
    if h_cols_pct:
        print("Computing pooled Spearman correlations (percentage delta)...")
        rho_pct_df, q_pct_df, n_pct_df = compute_spearman_pairwise(df, metric_cols, h_cols_pct)

    # Summaries
    print("Generating summaries...")
    summary_per_h, summary_per_metric = summarize_results(rho_df, q_df)

    # Top-K per horizon
    print("Extracting top metrics per horizon...")
    topk_df = export_topk_per_horizon(rho_df, q_df, n_df, k=10)

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving pooled results to: {out_dir}")
    
    rho_csv = os.path.join(out_dir, "spearman_rho_by_metric_by_horizon_pooled.csv")
    q_csv = os.path.join(out_dir, "spearman_fdr_q_by_metric_by_horizon_pooled.csv")
    summary_h_csv = os.path.join(out_dir, "summary_by_horizon.csv")
    summary_m_csv = os.path.join(out_dir, "summary_by_metric.csv")
    topk_csv = os.path.join(out_dir, "top10_metrics_per_horizon.csv")

    rho_df.to_csv(rho_csv)
    q_df.to_csv(q_csv)
    summary_per_h.to_csv(summary_h_csv)
    summary_per_metric.to_csv(summary_m_csv)
    topk_df.to_csv(topk_csv, index=False)

    if h_cols_pct:
        rho_pct_csv = os.path.join(out_dir, "spearman_rho_by_metric_by_horizon_pooled_pct.csv")
        q_pct_csv = os.path.join(out_dir, "spearman_fdr_q_by_metric_by_horizon_pooled_pct.csv")
        rho_pct_df.to_csv(rho_pct_csv)
        q_pct_df.to_csv(q_pct_csv)
        print(f"Saved percentage-based pooled results")

    # Plots
    print("Generating plots...")
    plot_top20_heatmap(rho_df, summary_per_metric, os.path.join(out_dir, "heatmap_top20_metrics_pooled.png"))
    plot_effect_sizes(summary_per_h, os.path.join(out_dir, "effect_size_vs_horizon.png"))
    plot_n_significant(summary_per_h, os.path.join(out_dir, "n_significant_per_horizon.png"))
    plot_subfamilies(rho_df, os.path.join(out_dir, "rho_subfamilies_vs_h.png"))
    
    # Within-regime comparison plot
    if has_regime_info:
        plot_within_regime_comparison(
            within_regime_results, 
            rho_df, 
            os.path.join(out_dir, "within_regime_comparison.png")
        )

    # Print summary statistics
    print(f"\n{'='*60}")
    print("POOLED ANALYSIS - Summary Statistics")
    print(f"{'='*60}")
    print("\nPer-horizon summary:")
    print(summary_per_h.to_string())
    
    print(f"\n\nTop 20 metrics by robustness across horizons (pooled):")
    print(summary_per_metric.head(20).to_string())

    # Print top metrics for selected horizons
    available_h = [_hnum(h) for h in h_cols]
    display_horizons = [h for h in [2, 4, 8, 16, 32] if h in available_h]
    
    for h in display_horizons:
        print(f"\n{'='*60}")
        print(f"Top 8 metrics at delta_test_loss_at_h{h} (pooled)")
        print(f"{'='*60}")
        top_h = top_for(topk_df, h, 8)
        if not top_h.empty:
            print(top_h.to_string(index=False))
        else:
            print("No significant metrics found")

    # Comparison: Within-regime vs Pooled
    if has_regime_info:
        print(f"\n{'='*60}")
        print("COMPARISON: Within-Regime vs Pooled Analysis")
        print(f"{'='*60}")
        compare_within_vs_pooled(within_regime_results, rho_df, q_df)

    analyze_split_effects(df)
    
    # Analyze action_epoch effects if variable
    if has_variable_action_epoch:
        analyze_action_epoch_effects(df, metric_cols, h_cols, out_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


# -------------------------
# Within-Regime Analysis Functions
# -------------------------

def analyze_action_epoch_effects(
    df: pd.DataFrame, 
    metric_cols: List[str], 
    h_cols: List[str],
    out_dir: str
):
    """
    Analyze how variable action_epoch affects split outcomes and correlations.
    
    Since action_epoch is now randomly sampled, this analysis checks:
    1. Distribution of action_epochs
    2. Whether split effectiveness varies with action_epoch
    3. Correlation between action_epoch and delta_test_loss
    4. Whether metric correlations are stable across action_epoch bins
    """
    print(f"\n{'='*60}")
    print("ACTION EPOCH EFFECTS ANALYSIS")
    print(f"{'='*60}")
    
    if 'action_epoch' not in df.columns:
        print("action_epoch column not found - skipping analysis")
        return
    
    action_epochs = df['action_epoch']
    print(f"\nAction epoch distribution:")
    print(f"  Min: {action_epochs.min()}")
    print(f"  Max: {action_epochs.max()}")
    print(f"  Mean: {action_epochs.mean():.1f}")
    print(f"  Std: {action_epochs.std():.1f}")
    print(f"  Unique values: {action_epochs.nunique()}")
    
    # Check if delta_test_loss correlates with action_epoch
    # (it might - later splits have less time to recover)
    print(f"\nCorrelation between action_epoch and delta_test_loss:")
    for h_col in h_cols[:5]:  # First 5 horizons
        h_num = _hnum(h_col)
        if h_col in df.columns:
            valid_mask = df[h_col].notna() & df['action_epoch'].notna()
            if valid_mask.sum() >= 10:
                rho = df.loc[valid_mask, 'action_epoch'].corr(
                    df.loc[valid_mask, h_col], method='spearman'
                )
                print(f"  h={h_num}: ρ = {rho:.3f}")
    
    # Bin action_epochs and check if correlations are stable
    print(f"\nStability check: correlations by action_epoch tercile")
    
    # Create tercile bins
    tercile_labels = ['early', 'middle', 'late']
    df_copy = df.copy()
    df_copy['action_epoch_tercile'] = pd.qcut(
        df_copy['action_epoch'], 
        q=3, 
        labels=tercile_labels,
        duplicates='drop'
    )
    
    # Pick a representative horizon and metric for illustration
    h_col_example = 'delta_test_loss_at_h8' if 'delta_test_loss_at_h8' in h_cols else h_cols[min(8, len(h_cols)-1)]
    
    # Find top 3 metrics by overall correlation
    if len(metric_cols) > 0:
        overall_corrs = {}
        for m in metric_cols[:50]:  # Check first 50 metrics
            valid = df[h_col_example].notna() & df[m].notna()
            if valid.sum() >= 10:
                overall_corrs[m] = abs(df.loc[valid, m].corr(df.loc[valid, h_col_example], method='spearman'))
        
        if overall_corrs:
            top_metrics = sorted(overall_corrs.keys(), key=lambda x: overall_corrs[x], reverse=True)[:3]
            
            print(f"\n  Checking stability for top 3 metrics at {h_col_example}:")
            for metric in top_metrics:
                metric_short = metric[:40] + "..." if len(metric) > 40 else metric
                print(f"\n  {metric_short}:")
                for tercile in tercile_labels:
                    tercile_df = df_copy[df_copy['action_epoch_tercile'] == tercile]
                    valid = tercile_df[h_col_example].notna() & tercile_df[metric].notna()
                    if valid.sum() >= 10:
                        rho = tercile_df.loc[valid, metric].corr(
                            tercile_df.loc[valid, h_col_example], method='spearman'
                        )
                        print(f"    {tercile}: ρ = {rho:.3f} (n={valid.sum()})")
                    else:
                        print(f"    {tercile}: insufficient data (n={valid.sum()})")
    
    # Plot action_epoch distribution and its effect on delta_test_loss
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Histogram of action_epochs
    ax = axes[0]
    ax.hist(action_epochs, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Action Epoch')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Split Times')
    ax.axvline(action_epochs.mean(), color='red', linestyle='--', label=f'Mean={action_epochs.mean():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Scatter plot of action_epoch vs delta_test_loss at h=8
    ax = axes[1]
    if h_col_example in df.columns:
        valid = df[h_col_example].notna()
        ax.scatter(df.loc[valid, 'action_epoch'], df.loc[valid, h_col_example], alpha=0.3, s=10)
        ax.set_xlabel('Action Epoch')
        ax.set_ylabel(f'Delta Test Loss (h={_hnum(h_col_example)})')
        ax.set_title('Split Time vs Outcome')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # 3. Mean delta_test_loss by action_epoch tercile
    ax = axes[2]
    if h_col_example in df.columns:
        tercile_means = df_copy.groupby('action_epoch_tercile')[h_col_example].agg(['mean', 'std', 'count'])
        x_pos = np.arange(len(tercile_labels))
        bars = ax.bar(x_pos, tercile_means['mean'], yerr=tercile_means['std']/np.sqrt(tercile_means['count']), 
                      capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tercile_labels)
        ax.set_xlabel('Action Epoch Tercile')
        ax.set_ylabel(f'Mean Delta Test Loss (h={_hnum(h_col_example)})')
        ax.set_title('Outcome by Split Time')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'action_epoch_effects.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved action_epoch effects plot to: {plot_path}")


def analyze_split_effects(df: pd.DataFrame):
    """Analyze whether splits help on average and variance in outcomes."""
    
    print(f"\n{'='*60}")
    print("SPLIT EFFECTIVENESS ANALYSIS")
    print(f"{'='*60}")
    
    # Look at a representative horizon (h=8)
    delta_col = 'delta_test_loss_at_h8'
    
    if delta_col not in df.columns:
        print("Delta column not found")
        return
    
    # Overall statistics
    deltas = df[delta_col].dropna()
    print(f"\nOverall (all regimes, n={len(deltas)}):")
    print(f"  Mean delta: {deltas.mean():.6f} (negative = improvement)")
    print(f"  Median delta: {deltas.median():.6f}")
    print(f"  Std dev: {deltas.std():.6f}")
    print(f"  % improved (delta < 0): {(deltas < 0).sum() / len(deltas) * 100:.1f}%")
    print(f"  % harmed (delta > 0): {(deltas > 0).sum() / len(deltas) * 100:.1f}%")
    print(f"  % neutral (|delta| < 0.001): {(deltas.abs() < 0.001).sum() / len(deltas) * 100:.1f}%")
    
    # By regime
    print(f"\nBy regime:")
    for regime in sorted(df['regime_name'].unique()):
        regime_deltas = df[df['regime_name'] == regime][delta_col].dropna()
        print(f"\n  {regime} (n={len(regime_deltas)}):")
        print(f"    Mean delta: {regime_deltas.mean():.6f}")
        print(f"    Std dev: {regime_deltas.std():.6f}")
        print(f"    % improved: {(regime_deltas < 0).sum() / len(regime_deltas) * 100:.1f}%")
        print(f"    Min (best): {regime_deltas.min():.6f}")
        print(f"    Max (worst): {regime_deltas.max():.6f}")
    
    # Variance analysis
    print(f"\n{'='*60}")
    print("VARIANCE ANALYSIS: Is there signal to learn?")
    print(f"{'='*60}")
    
    # If all splits have similar outcomes, there's no signal to predict
    # If outcomes vary widely, there's signal to learn from
    
    for regime in sorted(df['regime_name'].unique()):
        regime_deltas = df[df['regime_name'] == regime][delta_col].dropna()
        signal_to_noise = regime_deltas.std() / abs(regime_deltas.mean() + 1e-10)
        
        print(f"\n  {regime}:")
        print(f"    Coefficient of variation: {signal_to_noise:.2f}")
        print(f"    Range: [{regime_deltas.min():.6f}, {regime_deltas.max():.6f}]")
        print(f"    IQR: {regime_deltas.quantile(0.75) - regime_deltas.quantile(0.25):.6f}")
        
        if signal_to_noise > 2:
            print(f"    → HIGH variance: Strong signal to learn from")
        elif signal_to_noise > 0.5:
            print(f"    → MODERATE variance: Some signal to learn from")
        else:
            print(f"    → LOW variance: Weak signal to learn from")
    
    # Plot distributions
    regimes = sorted(df['regime_name'].unique())
    n_regimes = len(regimes)
    fig, axes = plt.subplots(1, n_regimes, figsize=(5 * n_regimes, 4))
    
    # Handle case where there's only 1 regime (axes won't be an array)
    if n_regimes == 1:
        axes = [axes]
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        regime_deltas = df[df['regime_name'] == regime][delta_col].dropna()
        
        ax.hist(regime_deltas, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='No effect')
        ax.axvline(regime_deltas.mean(), color='green', linestyle='--', label=f'Mean={regime_deltas.mean():.4f}')
        ax.set_xlabel('Delta test loss (lower = better)')
        ax.set_ylabel('Count')
        ax.set_title(f'{regime} regime')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Get the analysis directory from the current context
    analysis_dir = Path(df.attrs.get('analysis_dir', 'output_local/analysis'))
    dist_plot_path = analysis_dir / 'split_effect_distributions.png'
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved distribution plot to: {dist_plot_path}")


def compute_within_regime_correlations(
    df: pd.DataFrame, 
    metric_cols: List[str], 
    h_cols: List[str],
    regime_col: str
) -> pd.DataFrame:
    """
    Compute correlations separately within each regime to avoid confounding
    from network size differences.
    
    Returns DataFrame with columns:
    - regime: regime name
    - metric: metric name
    - horizon: horizon column name
    - rho: Spearman correlation
    - p_value: raw p-value
    - q_value: FDR-corrected q-value
    - n: sample size
    """
    results = []
    regimes = df[regime_col].unique()
    
    for regime in sorted(regimes):
        regime_df = df[df[regime_col] == regime]
        n_samples = len(regime_df)
        
        print(f"\n  Regime: {regime} (n={n_samples} samples)")
        
        if n_samples < 10:
            print(f"    WARNING: Only {n_samples} samples - skipping")
            continue
        
        # Compute correlations for this regime
        rho_df, q_df, n_df = compute_spearman_pairwise(regime_df, metric_cols, h_cols)
        
        # Convert to long format
        for metric in metric_cols:
            for h_col in h_cols:
                rho = rho_df.loc[metric, h_col]
                q = q_df.loc[metric, h_col]
                n = n_df.loc[metric, h_col]
                
                if np.isfinite(rho):
                    # Compute raw p-value from rho and n
                    if n >= 10:
                        t_stat = rho * np.sqrt(max(n - 2, 0) / max(1.0 - rho**2, 1e-12))
                        p_value = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), max(n - 2, 1)))
                    else:
                        p_value = np.nan
                    
                    results.append({
                        'regime': regime,
                        'metric': metric,
                        'horizon': h_col,
                        'horizon_num': _hnum(h_col),
                        'rho': rho,
                        'p_value': p_value,
                        'q_value': q,
                        'n': int(n)
                    })
    
    return pd.DataFrame(results)


def print_within_regime_summary(within_results: pd.DataFrame):
    """Print summary of within-regime correlations."""
    if within_results.empty:
        print("No within-regime results to summarize")
        return
    
    # For each regime, show top metrics at a few horizons
    regimes = within_results['regime'].unique()
    display_horizons = [4, 8, 16]
    
    for regime in sorted(regimes):
        regime_data = within_results[within_results['regime'] == regime]
        print(f"\n  Regime: {regime}")
        
        for h in display_horizons:
            h_data = regime_data[regime_data['horizon_num'] == h]
            h_data_sig = h_data[h_data['q_value'] < 0.05]
            h_data_sig = h_data_sig.sort_values('rho', key=abs, ascending=False)
            
            if len(h_data_sig) > 0:
                top_metric = h_data_sig.iloc[0]
                print(f"    h={h}: top metric = {top_metric['metric'][:40]}... "
                      f"(ρ={top_metric['rho']:.3f}, q={top_metric['q_value']:.4f}, n={top_metric['n']})")
            else:
                print(f"    h={h}: no significant metrics")


def compare_within_vs_pooled(
    within_results: pd.DataFrame,
    pooled_rho: pd.DataFrame,
    pooled_q: pd.DataFrame
):
    """Compare within-regime vs pooled correlation results."""
    
    # Find metrics that are significant in MOST regimes
    regimes = within_results['regime'].unique()
    n_regimes = len(regimes)
    
    # For horizon 8 (example)
    h_col = 'delta_test_loss_at_h8'
    h_num = 8
    
    if h_col not in pooled_rho.columns:
        print(f"Horizon {h_num} not available")
        return
    
    print(f"\nAnalyzing horizon h={h_num}:")
    
    # Get within-regime results for this horizon
    h_within = within_results[within_results['horizon'] == h_col]
    
    # Count how many regimes each metric is significant in
    sig_counts = h_within[h_within['q_value'] < 0.05].groupby('metric').size()
    sig_counts = sig_counts.sort_values(ascending=False)
    
    print(f"\nMetrics significant in multiple regimes:")
    print(f"{'Metric':<50} {'# Regimes Sig':<15} {'Pooled ρ':<12} {'Pooled q':<12}")
    print("-" * 90)
    
    for metric in sig_counts.head(10).index:
        n_sig = sig_counts[metric]
        pooled_rho_val = pooled_rho.loc[metric, h_col]
        pooled_q_val = pooled_q.loc[metric, h_col]
        
        metric_short = metric[:47] + "..." if len(metric) > 50 else metric
        print(f"{metric_short:<50} {n_sig}/{n_regimes:<15} {pooled_rho_val:>10.3f}  {pooled_q_val:>10.4f}")
    
    # Show regime-specific correlations for top metric
    if len(sig_counts) > 0:
        top_metric = sig_counts.index[0]
        print(f"\n\nRegime-specific correlations for top metric: {top_metric[:60]}")
        print(f"{'Regime':<20} {'ρ':<10} {'q-value':<12} {'n':<8}")
        print("-" * 50)
        
        metric_data = h_within[h_within['metric'] == top_metric]
        for _, row in metric_data.iterrows():
            print(f"{str(row['regime']):<20} {row['rho']:>8.3f}  {row['q_value']:>10.4f}  {int(row['n']):<8}")


def plot_within_regime_comparison(
    within_results: pd.DataFrame,
    pooled_rho: pd.DataFrame,
    outpath: str
):
    """
    Plot comparison of within-regime vs pooled correlations for top metrics.
    Shows how correlation strength varies across regimes.
    """
    # Select a few representative horizons
    horizons = [4, 8, 16]
    
    # Find top 5 metrics by pooled correlation at h=8
    h_col_8 = 'delta_test_loss_at_h8'
    if h_col_8 not in pooled_rho.columns:
        print("Cannot create within-regime comparison plot: h=8 not available")
        return
    
    top_metrics = pooled_rho[h_col_8].abs().nlargest(5).index.tolist()
    
    fig, axes = plt.subplots(1, len(horizons), figsize=(15, 5), sharey=True)
    
    regimes = sorted(within_results['regime'].unique())
    x_pos = np.arange(len(regimes))
    
    for idx, h in enumerate(horizons):
        ax = axes[idx]
        h_col = f'delta_test_loss_at_h{h}'
        
        # For each top metric, plot its correlation across regimes
        for metric in top_metrics[:3]:  # Limit to 3 for readability
            metric_vals = []
            for regime in regimes:
                regime_metric = within_results[
                    (within_results['regime'] == regime) & 
                    (within_results['metric'] == metric) &
                    (within_results['horizon'] == h_col)
                ]
                if len(regime_metric) > 0:
                    metric_vals.append(regime_metric.iloc[0]['rho'])
                else:
                    metric_vals.append(np.nan)
            
            # Shorten metric name for legend
            metric_short = metric.split('__')[-1][:20] if '__' in metric else metric[:20]
            ax.plot(x_pos, metric_vals, marker='o', label=metric_short, linewidth=2)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Regime')
        ax.set_title(f'Horizon h={h}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regimes, rotation=45)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Spearman ρ')
        if idx == len(horizons) - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle('Within-Regime Correlations: Top 3 Metrics Across Regimes', fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved within-regime comparison plot to: {outpath}")


# -------------------------
# Original Helper Functions (Unchanged)
# -------------------------

def _hnum(col: str) -> int:
    """Extract horizon number from column name like 'delta_test_loss_at_h4'."""
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
    """Benjamini-Hochberg FDR correction on an array of p-values (any shape)."""
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

    # p-values via t approximation
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
    """Generate per-horizon and per-metric summary statistics."""
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
    """Export top-k significant metrics for each horizon."""
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
    """Plot heatmap of top 20 metrics across horizons."""
    top20 = summary_per_metric.head(20).index.tolist()
    h_cols = rho_df.columns.tolist()
    hnums = [_hnum(h) for h in h_cols]
    H = rho_df.loc[top20, h_cols]
    plt.figure(figsize=(12, 8))
    plt.imshow(H, aspect='auto', interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Spearman rho')
    plt.yticks(ticks=np.arange(len(top20)), labels=top20, fontsize=8)
    plt.xticks(ticks=np.arange(len(h_cols)), labels=hnums, rotation=90)
    plt.title("Top 20 metrics by mean |rho| across horizons (pooled)")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_effect_sizes(summary_per_h: pd.DataFrame, outpath: str):
    """Plot median and 95th percentile effect sizes vs horizon."""
    hnums = [_hnum(h) for h in summary_per_h.index.tolist()]
    plt.figure(figsize=(9, 5))
    plt.plot(hnums, summary_per_h["median_|rho|"].values, marker='o', label="Median |rho|")
    plt.plot(hnums, summary_per_h["p95_|rho|"].values, marker='s', label="P95 |rho|")
    plt.title("Effect size vs horizon (pooled)")
    plt.xlabel("Horizon (h)")
    plt.ylabel("|rho|")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_n_significant(summary_per_h: pd.DataFrame, outpath: str):
    """Plot number of significant metrics per horizon."""
    hnums = [_hnum(h) for h in summary_per_h.index.tolist()]
    plt.figure(figsize=(10, 5))
    plt.bar(hnums, summary_per_h["n_significant"].values)
    plt.title("Number of significant metrics per horizon (FDR q<0.05, pooled)")
    plt.xlabel("Horizon (h)")
    plt.ylabel("# metrics")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_subfamilies(rho_df: pd.DataFrame, outpath: str):
    """Plot average correlation for subfamilies of post_activation_grad_mean metrics."""
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
        plt.plot(hnums, vals, marker='o', label=label)
    plt.axhline(0, linestyle="--", color='gray', alpha=0.5)
    plt.title("Average rho by subfamily (post_activation_grad_mean, pooled)")
    plt.xlabel("Horizon (h)")
    plt.ylabel("Spearman rho")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def top_for(topk_df: pd.DataFrame, hnum: int, k: int = 8) -> pd.DataFrame:
    """Get top-k metrics for a specific horizon."""
    hcol = f"delta_test_loss_at_h{hnum}"
    sub = topk_df[topk_df["horizon"] == hcol].copy()
    return sub.sort_values("rank").head(k)[["metric","rho","q_value","n"]]

if __name__ == "__main__":
    # Get experiment directory name
    experiment_dir_name = Path(__file__).parent.name
    base_dir = f"output_local/{experiment_dir_name}"
    
    # Define datasets to analyze
    datasets = [
        {
            "name": "FashionMNIST",
            "folder": "mnist",
            "csv_filename": "correlation_experiment_results_FashionMNIST.csv"
        },
        {
            "name": "CIFAR-10",
            "folder": "cifar-10",
            "csv_filename": "correlation_experiment_results_CIFAR10.csv"
        }
    ]
    
    # Run analysis for each dataset
    for dataset in datasets:
        csv_path = f"{base_dir}/{dataset['folder']}/{dataset['csv_filename']}"
        out_dir = f"{base_dir}/{dataset['folder']}/analysis"
        
        # Check if CSV exists
        if not os.path.exists(csv_path):
            print(f"\n{'='*60}")
            print(f"Skipping {dataset['name']}: CSV not found")
            print(f"Expected: {csv_path}")
            print(f"{'='*60}\n")
            continue
        
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset['name']} DATASET")
        print(f"{'='*80}\n")
        
        main(csv_path, out_dir)
    
    print(f"\n{'='*80}")
    print("ALL DATASETS ANALYZED SUCCESSFULLY!")
    print(f"{'='*80}\n")