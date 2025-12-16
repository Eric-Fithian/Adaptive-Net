"""
Correlation Analysis for FashionMNIST and CIFAR-10 Experiments
===============================================================

Computes Spearman correlations WITHIN each regime to avoid confounding from
network size differences (architecture size correlates with both metrics and outcomes).

Analysis Structure (Progressive Complexity):
    Part 0: Definitions and experimental setup documentation
    Part 1: Data overview (including non-IID structure documentation)
    Part 2: Atomic metrics (13 base metrics at split time) - PRIMARY ANALYSIS
    Part 3: Temporal metrics with basic aggregations (min, max, mean, median, var)
    Part 4: Temporal metrics with all aggregation methods - EXPLORATORY
    Part 5: Cross-regime consistency analysis
    Part 6: Horizon trend analysis
    Part 7: Cross-dataset transfer validation (FashionMNIST vs CIFAR10)
    Part 8: Practical significance / decision-theoretic analysis

Statistical Methods:
    - Spearman rank correlation (robust to outliers, captures monotonic relationships)
    - Benjamini-Hochberg FDR correction (controls false discovery rate at q < 0.05)
    - Within-regime analysis (avoids architecture size confounding)

Output Structure:
    - Large CSV files → experiments/x1_correlation/output_local/{dataset}/ (not committed)
    - Graphs (PNG) → experiments/x1_correlation/output/{dataset}/ (committed)

Usage:
    python s2_analysis.py
"""

import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, spearmanr

# -------------------------
# Constants
# -------------------------

# The 13 atomic metrics (measured at split time, no temporal aggregation)
ATOMIC_METRICS = [
    'input_weights_mean',
    'input_weights_var',
    'output_weights_mean',
    'output_weights_var',
    'bias',
    'pre_activation_mean',
    'post_activation_mean',
    'input_weight_grads_mean',
    'input_weight_grads_var',
    'output_weight_grads_mean',
    'output_weight_grads_var',
    'pre_activation_grad_mean',
    'post_activation_grad_mean',
]

# Basic aggregation methods for initial exploration
BASIC_AGGREGATIONS = ['min', 'max', 'mean', 'median', 'var']

# Aggregation method families for organizing results
AGGREGATION_FAMILIES = {
    'central': ['mean', 'median', 'ema_mean_hl'],
    'spread': ['std', 'var', 'mad', 'iqr', 'range'],
    'extremes': ['min', 'max', 'q10', 'q90', 'tail_exceed_p90_rate'],
    'trend': ['slope_ols', 'trend_tstat', 'delta_last_first', 'pct_change_last_first', 'spearman_rho_time'],
    'dynamics': ['autocorr_lag1', 'vol_cluster_acf', 'zero_cross_rate', 'sign_persistence'],
    'curvature': ['mean_second_diff', 'abs_mean_second_diff'],
    'energy': ['l1norm', 'l2norm'],
    'regime_shift': ['half_diff_mean', 'ema_crossover_rate'],
    'spectral': ['dom_freq_idx', 'spectral_entropy', 'low_high_bandpower_ratio'],
    'robust': ['above_1_sigma_rate', 'above_2_sigma_rate', 'fano_factor'],
    'distribution': ['skew', 'kurtosis'],
}

# Primary analysis parameters (pre-specified to avoid multiple testing inflation)
PRIMARY_HORIZON = 8  # Pre-specified primary horizon for main conclusions
PRIMARY_METRICS = ATOMIC_METRICS + [f'tmp8__{agg}__{m}' for agg in BASIC_AGGREGATIONS for m in ATOMIC_METRICS]


# -------------------------
# Main Entry Point
# -------------------------

def main(csv_path: str, csv_out_dir: str, graph_out_dir: str) -> Dict:
    """
    Run the full correlation analysis pipeline with progressive complexity.
    
    Args:
        csv_path: Path to input CSV
        csv_out_dir: Directory for large CSV output files (output_local/)
        graph_out_dir: Directory for graph/PNG output files (experiments/output/)
    
    Returns results dict for cross-dataset analysis.
    """
    dataset_name = Path(csv_path).stem.replace("correlation_experiment_results_", "")
    
    print(f"\n{'='*70}")
    print(f" {dataset_name} CORRELATION ANALYSIS")
    print(f" Within-Regime Analysis (avoids architecture size confounding)")
    print(f"{'='*70}")
    print(f"Input: {csv_path}")
    print(f"CSV output: {csv_out_dir}")
    print(f"Graph output: {graph_out_dir}")
    
    # Load data
    df = pd.read_csv(csv_path)
    os.makedirs(csv_out_dir, exist_ok=True)
    os.makedirs(graph_out_dir, exist_ok=True)
    
    # Identify regime column
    regime_col = 'regime_name' if 'regime_name' in df.columns else 'starting_width'
    if regime_col not in df.columns:
        raise ValueError("No regime column found. Cannot perform within-regime analysis.")
    
    # Identify horizon columns
    h_cols = sorted([c for c in df.columns if c.startswith("delta_test_loss_at_h")], key=_hnum)
    
    # =========================================================================
    # PART 0: Definitions
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 0: DEFINITIONS & EXPERIMENTAL SETUP")
    print(f"{'='*70}")
    part0_definitions(graph_out_dir)
    
    # =========================================================================
    # PART 1: Data Overview (including non-IID structure)
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 1: DATA OVERVIEW & NON-IID STRUCTURE")
    print(f"{'='*70}")
    part1_data_overview(df, regime_col, h_cols, graph_out_dir)
    
    # =========================================================================
    # PART 2: Atomic Metrics Analysis (PRIMARY ANALYSIS)
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 2: ATOMIC METRICS (PRIMARY ANALYSIS)")
    print(f" Pre-specified: 13 base metrics at h={PRIMARY_HORIZON}")
    print(f"{'='*70}")
    atomic_cols = [c for c in ATOMIC_METRICS if c in df.columns]
    if atomic_cols:
        atomic_results = part2_atomic_metrics(df, atomic_cols, h_cols, regime_col, csv_out_dir, graph_out_dir)
    else:
        print("WARNING: No atomic metrics found in data")
        atomic_results = pd.DataFrame()
    
    # =========================================================================
    # PART 3: Temporal Metrics with Basic Aggregations
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 3: TEMPORAL METRICS - BASIC AGGREGATIONS")
    print(f" (Using: {', '.join(BASIC_AGGREGATIONS)})")
    print(f"{'='*70}")
    temporal_basic_results = part3_temporal_basic(df, h_cols, regime_col, csv_out_dir, graph_out_dir)
    
    # =========================================================================
    # PART 4: Temporal Metrics with All Aggregations (EXPLORATORY)
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 4: TEMPORAL METRICS - ALL AGGREGATIONS (EXPLORATORY)")
    print(f" Note: ~2300 metrics × {len(h_cols)} horizons - interpret with caution")
    print(f"{'='*70}")
    temporal_full_results = part4_temporal_full(df, h_cols, regime_col, csv_out_dir, graph_out_dir)
    
    # =========================================================================
    # PART 5: Cross-Regime Consistency
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 5: CROSS-REGIME CONSISTENCY ANALYSIS")
    print(f"{'='*70}")
    part5_cross_regime_consistency(atomic_results, temporal_full_results, regime_col, csv_out_dir, graph_out_dir)
    
    # =========================================================================
    # PART 6: Horizon Trend Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 6: HORIZON TREND ANALYSIS")
    print(f"{'='*70}")
    part6_horizon_analysis(temporal_full_results, h_cols, csv_out_dir, graph_out_dir)
    
    # =========================================================================
    # PART 8: Practical Significance / Decision-Theoretic Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print(" PART 8: PRACTICAL SIGNIFICANCE ANALYSIS")
    print(f"{'='*70}")
    part8_practical_significance(df, atomic_results, temporal_basic_results, regime_col, h_cols, csv_out_dir, graph_out_dir)
    
    print(f"\n{'='*70}")
    print(f" SINGLE-DATASET ANALYSIS COMPLETE")
    print(f" CSV results: {csv_out_dir}")
    print(f" Graph results: {graph_out_dir}")
    print(f"{'='*70}\n")
    
    # Return results for cross-dataset analysis
    return {
        'dataset_name': dataset_name,
        'df': df,
        'atomic_results': atomic_results,
        'temporal_basic_results': temporal_basic_results,
        'temporal_full_results': temporal_full_results,
        'regime_col': regime_col,
        'h_cols': h_cols,
    }


# -------------------------
# Part 0: Definitions
# -------------------------

def part0_definitions(graph_out_dir: str):
    """Print and save experimental setup definitions."""
    
    definitions = """
EXPERIMENTAL SETUP DEFINITIONS
==============================

1. SAMPLE UNIT
   Each row = one neuron at one split decision point
   - A neuron is selected for potential splitting
   - Its metrics are recorded at the moment of decision
   - The split is performed and outcomes are measured

2. TARGET VARIABLE: delta_test_loss_at_h{N}
   delta = (test_loss_after_split at epoch+N) - (test_loss_of_unsplit_baseline at epoch+N)
   
   Interpretation:
   - delta < 0 → Split HELPED (test loss decreased)
   - delta > 0 → Split HURT (test loss increased)
   - delta ≈ 0 → No significant effect
   
   We want metrics with NEGATIVE correlation (higher metric → lower delta → better outcome)

3. REGIME
   Network architecture width category:
   - tiny: 10 hidden neurons
   - small: 20 hidden neurons
   - medium: 40 hidden neurons
   - large: 80 hidden neurons
   
   Analysis is done WITHIN each regime to avoid confounding from architecture size.

4. SPEARMAN CORRELATION (ρ)
   Rank-based correlation measuring monotonic relationship.
   - ρ = -0.3: Higher metric values predict better outcomes (split helps)
   - ρ = +0.3: Higher metric values predict worse outcomes (split hurts)
   - |ρ| > 0.1: Small effect
   - |ρ| > 0.3: Medium effect  
   - |ρ| > 0.5: Large effect

5. STATISTICAL TESTING
   - Benjamini-Hochberg FDR correction applied
   - q < 0.05 threshold for significance
   - FDR controls expected proportion of false discoveries among rejected hypotheses

6. NON-IID DATA STRUCTURE
   Neurons are nested within networks:
   - Multiple neurons per network
   - Multiple networks per regime
   - Correlations within network may exist
   
   Report: number of unique networks (init_id) per regime

7. PRIMARY vs EXPLORATORY ANALYSIS
   PRIMARY (pre-specified): 13 atomic metrics at h=8
   EXPLORATORY: All temporal metrics, all horizons
   
   Primary analysis conclusions are confirmatory.
   Exploratory analysis generates hypotheses for future testing.
"""
    
    print(definitions)
    
    # Save to file
    with open(os.path.join(graph_out_dir, 'part0_definitions.txt'), 'w') as f:
        f.write(definitions)
    print(f"Saved: part0_definitions.txt")


# -------------------------
# Part 1: Data Overview
# -------------------------

def part1_data_overview(df: pd.DataFrame, regime_col: str, h_cols: List[str], graph_out_dir: str):
    """Print data overview including non-IID structure documentation."""
    print(f"\nTotal samples (neuron-split observations): {len(df)}")
    print(f"Regime column: {regime_col}")
    
    # Non-IID structure: count unique networks
    print(f"\n--- NON-IID DATA STRUCTURE ---")
    if 'init_id' in df.columns:
        n_networks = df['init_id'].nunique()
        print(f"Total unique networks (init_id): {n_networks}")
        print(f"Average neurons per network: {len(df) / n_networks:.1f}")
        
        print(f"\nPer-regime breakdown:")
        for regime in sorted(df[regime_col].unique()):
            regime_df = df[df[regime_col] == regime]
            n_regime_networks = regime_df['init_id'].nunique()
            n_samples = len(regime_df)
            print(f"  {regime}: {n_samples} samples from {n_regime_networks} networks "
                  f"({n_samples/n_regime_networks:.1f} neurons/network)")
    else:
        print("  WARNING: 'init_id' column not found - cannot assess network clustering")
        print("  Statistical tests assume independence which may be violated")
    
    # Regime distribution
    print(f"\n--- SAMPLE SIZES ---")
    regime_counts = df[regime_col].value_counts().sort_index()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} samples")
    
    # Horizons
    print(f"\n--- TARGET VARIABLES ---")
    print(f"Horizons available: {len(h_cols)}")
    print(f"  Range: h{_hnum(h_cols[0])} to h{_hnum(h_cols[-1])}")
    print(f"  Primary horizon (pre-specified): h={PRIMARY_HORIZON}")
    
    # Column breakdown
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    atomic_count = sum(1 for c in ATOMIC_METRICS if c in df.columns)
    temporal_count = sum(1 for c in num_cols if c.startswith('tmp'))
    
    print(f"\n--- PREDICTOR METRICS ---")
    print(f"  Atomic metrics: {atomic_count}")
    print(f"  Temporal metrics: {temporal_count}")
    print(f"  Total hypothesis tests (all horizons): ~{(atomic_count + temporal_count) * len(h_cols):,}")
    
    # Outcome distribution by regime
    example_h = f'delta_test_loss_at_h{PRIMARY_HORIZON}'
    if example_h not in h_cols:
        example_h = h_cols[min(8, len(h_cols)-1)]
    
    print(f"\n--- OUTCOME DISTRIBUTION at {example_h} ---")
    for regime in sorted(df[regime_col].unique()):
        regime_data = df[df[regime_col] == regime][example_h].dropna()
        if len(regime_data) > 0:
            pct_improved = (regime_data < 0).mean() * 100
            print(f"  {regime}: mean={regime_data.mean():.6f}, std={regime_data.std():.6f}, "
                  f"{pct_improved:.1f}% improved")
    
    # Plot outcome distributions by regime
    _plot_outcome_distributions(df, regime_col, example_h, graph_out_dir)


def _plot_outcome_distributions(df: pd.DataFrame, regime_col: str, h_col: str, graph_out_dir: str):
    """Plot outcome distributions by regime."""
    regimes = sorted(df[regime_col].unique())
    n_regimes = len(regimes)
    
    fig, axes = plt.subplots(1, n_regimes, figsize=(4*n_regimes, 4))
    if n_regimes == 1:
        axes = [axes]
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        data = df[df[regime_col] == regime][h_col].dropna()
        
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='No effect')
        ax.axvline(data.mean(), color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean={data.mean():.4f}')
        ax.set_xlabel('Delta test loss (negative=improvement)')
        ax.set_ylabel('Count')
        ax.set_title(f'{regime} (n={len(data)})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Outcome Distribution at {h_col}', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, 'part1_outcome_distributions.png'), dpi=150)
    plt.close()
    print(f"\nSaved: part1_outcome_distributions.png")


# -------------------------
# Part 2: Atomic Metrics (PRIMARY ANALYSIS)
# -------------------------

def part2_atomic_metrics(
    df: pd.DataFrame, 
    atomic_cols: List[str], 
    h_cols: List[str], 
    regime_col: str, 
    csv_out_dir: str,
    graph_out_dir: str
) -> pd.DataFrame:
    """PRIMARY ANALYSIS: Correlations for atomic (non-temporal) metrics."""
    print(f"\n*** PRIMARY ANALYSIS: {len(atomic_cols)} atomic metrics ***")
    print(f"Pre-specified hypothesis: These metrics predict split outcomes at h={PRIMARY_HORIZON}")
    
    results = compute_within_regime_correlations(df, atomic_cols, h_cols, regime_col)
    
    # Save results
    results.to_csv(os.path.join(csv_out_dir, 'part2_atomic_correlations.csv'), index=False)
    print(f"Saved: part2_atomic_correlations.csv")
    
    # Primary horizon results
    primary_h_col = f'delta_test_loss_at_h{PRIMARY_HORIZON}'
    if primary_h_col in [r['horizon'] for r in results.to_dict('records') if results.size > 0]:
        print(f"\n--- PRIMARY RESULTS at h={PRIMARY_HORIZON} ---")
        _print_primary_results(results, PRIMARY_HORIZON)
    
    # Plot
    _plot_atomic_heatmap(results, atomic_cols, h_cols, regime_col, graph_out_dir)
    
    return results


def _print_primary_results(results: pd.DataFrame, h_num: int):
    """Print results for primary hypothesis (pre-specified horizon)."""
    h_data = results[results['horizon_num'] == h_num]
    
    for regime in sorted(results['regime'].unique()):
        regime_data = h_data[h_data['regime'] == regime]
        sig_data = regime_data[regime_data['q_value'] < 0.05].sort_values('rho')
        
        print(f"\n  Regime: {regime} (n={regime_data['n'].iloc[0] if len(regime_data) > 0 else 0})")
        print(f"  Significant metrics (q<0.05): {len(sig_data)}/{len(regime_data)}")
        
        if len(sig_data) > 0:
            print(f"  {'Metric':<35} {'ρ':>8} {'q-value':>10} {'Direction':<12}")
            print(f"  {'-'*65}")
            for _, row in sig_data.head(10).iterrows():
                direction = "SPLIT HELPS" if row['rho'] < 0 else "SPLIT HURTS"
                print(f"  {row['metric']:<35} {row['rho']:>+8.3f} {row['q_value']:>10.4f} {direction:<12}")


def _plot_atomic_heatmap(
    results: pd.DataFrame, 
    atomic_cols: List[str], 
    h_cols: List[str],
    regime_col: str,
    graph_out_dir: str
):
    """Plot heatmap of atomic metric correlations by regime."""
    regimes = sorted(results['regime'].unique())
    n_regimes = len(regimes)
    
    h_nums = sorted(results['horizon_num'].unique())
    if len(h_nums) > 15:
        h_subset = h_nums[::max(1, len(h_nums)//15)]
    else:
        h_subset = h_nums
    
    fig, axes = plt.subplots(1, n_regimes, figsize=(5*n_regimes, 6), sharey=True)
    if n_regimes == 1:
        axes = [axes]
    
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        regime_data = results[results['regime'] == regime]
        
        pivot = regime_data.pivot_table(
            values='rho', 
            index='metric', 
            columns='horizon_num',
            aggfunc='first'
        )
        
        pivot = pivot.reindex(index=[c for c in atomic_cols if c in pivot.index])
        pivot = pivot[[h for h in h_subset if h in pivot.columns]]
        
        if pivot.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{regime}')
            continue
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_xlabel('Horizon')
        ax.set_title(f'{regime}')
        
        if idx == 0:
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
    
    plt.colorbar(im, ax=axes, label='Spearman ρ (negative=split helps)', shrink=0.8)
    plt.suptitle('Atomic Metric Correlations by Regime (PRIMARY ANALYSIS)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, 'part2_atomic_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: part2_atomic_heatmap.png")


# -------------------------
# Part 3: Temporal Basic Aggregations
# -------------------------

def part3_temporal_basic(
    df: pd.DataFrame,
    h_cols: List[str],
    regime_col: str,
    csv_out_dir: str,
    graph_out_dir: str
) -> pd.DataFrame:
    """Analyze temporal metrics using only basic aggregations."""
    temporal_basic_cols = []
    for col in df.columns:
        if col.startswith('tmp'):
            parts = col.split('__')
            if len(parts) >= 2 and parts[1] in BASIC_AGGREGATIONS:
                temporal_basic_cols.append(col)
    
    print(f"\nFound {len(temporal_basic_cols)} temporal metrics with basic aggregations")
    
    if not temporal_basic_cols:
        print("WARNING: No temporal basic metrics found")
        return pd.DataFrame()
    
    results = compute_within_regime_correlations(df, temporal_basic_cols, h_cols, regime_col)
    results = _parse_temporal_columns(results)
    
    results.to_csv(os.path.join(csv_out_dir, 'part3_temporal_basic_correlations.csv'), index=False)
    print(f"Saved: part3_temporal_basic_correlations.csv")
    
    _analyze_temporal_patterns(results, "basic", graph_out_dir)
    
    return results


def _parse_temporal_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Parse temporal column names to extract window, aggregation, and base metric."""
    if results.empty:
        return results
    
    windows = []
    aggregations = []
    base_metrics = []
    
    for metric in results['metric']:
        parts = metric.split('__')
        if len(parts) >= 3 and parts[0].startswith('tmp'):
            window = int(parts[0].replace('tmp', ''))
            agg = parts[1]
            base = '__'.join(parts[2:])
        else:
            window, agg, base = None, None, metric
        
        windows.append(window)
        aggregations.append(agg)
        base_metrics.append(base)
    
    results = results.copy()
    results['temporal_window'] = windows
    results['aggregation'] = aggregations
    results['base_metric'] = base_metrics
    
    return results


def _analyze_temporal_patterns(results: pd.DataFrame, analysis_type: str, graph_out_dir: str):
    """Analyze patterns in temporal metrics by window and aggregation."""
    if results.empty or 'temporal_window' not in results.columns:
        return
    
    sig_results = results[results['q_value'] < 0.05]
    
    if sig_results.empty:
        print("No significant correlations found")
        return
    
    print(f"\n  Correlation strength by temporal window:")
    window_stats = sig_results.groupby('temporal_window')['rho'].agg(['mean', 'std', 'count'])
    window_stats['abs_mean'] = sig_results.groupby('temporal_window')['rho'].apply(lambda x: x.abs().mean())
    for window, row in window_stats.iterrows():
        print(f"    Window {window}: mean |ρ|={row['abs_mean']:.3f}, n_sig={int(row['count'])}")
    
    print(f"\n  Correlation strength by aggregation method:")
    agg_stats = sig_results.groupby('aggregation')['rho'].agg(['mean', 'count'])
    agg_stats['abs_mean'] = sig_results.groupby('aggregation')['rho'].apply(lambda x: x.abs().mean())
    agg_stats = agg_stats.sort_values('abs_mean', ascending=False)
    for agg, row in agg_stats.head(10).iterrows():
        print(f"    {agg}: mean |ρ|={row['abs_mean']:.3f}, n_sig={int(row['count'])}")
    
    _plot_window_comparison(results, analysis_type, graph_out_dir)


def _plot_window_comparison(results: pd.DataFrame, analysis_type: str, graph_out_dir: str):
    """Plot correlation strength by temporal window."""
    if results.empty or 'temporal_window' not in results.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    for regime in sorted(results['regime'].unique()):
        regime_data = results[results['regime'] == regime]
        window_means = regime_data.groupby('temporal_window')['rho'].apply(lambda x: x.abs().mean())
        ax.plot(window_means.index, window_means.values, marker='o', label=regime)
    
    ax.set_xlabel('Temporal Window')
    ax.set_ylabel('Mean |Spearman ρ|')
    ax.set_title('Correlation Strength by Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    sig_results = results[results['q_value'] < 0.05]
    for regime in sorted(results['regime'].unique()):
        regime_sig = sig_results[sig_results['regime'] == regime]
        window_counts = regime_sig.groupby('temporal_window').size()
        ax.plot(window_counts.index, window_counts.values, marker='s', label=regime)
    
    ax.set_xlabel('Temporal Window')
    ax.set_ylabel('# Significant Metrics (q<0.05)')
    ax.set_title('Significant Metrics by Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Temporal Window Analysis ({analysis_type})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, f'part3_window_comparison_{analysis_type}.png'), dpi=150)
    plt.close()
    print(f"Saved: part3_window_comparison_{analysis_type}.png")


# -------------------------
# Part 4: Temporal Full (EXPLORATORY)
# -------------------------

def part4_temporal_full(
    df: pd.DataFrame,
    h_cols: List[str],
    regime_col: str,
    csv_out_dir: str,
    graph_out_dir: str
) -> pd.DataFrame:
    """EXPLORATORY: Analyze ALL temporal metrics with all aggregation methods."""
    temporal_cols = [c for c in df.columns if c.startswith('tmp')]
    print(f"\nFound {len(temporal_cols)} total temporal metrics")
    print(f"*** EXPLORATORY ANALYSIS - results should be validated ***")
    
    if not temporal_cols:
        return pd.DataFrame()
    
    results = compute_within_regime_correlations(df, temporal_cols, h_cols, regime_col)
    results = _parse_temporal_columns(results)
    results['aggregation_family'] = results['aggregation'].apply(_get_aggregation_family)
    
    results.to_csv(os.path.join(csv_out_dir, 'part4_temporal_full_correlations.csv'), index=False)
    print(f"Saved: part4_temporal_full_correlations.csv")
    
    _analyze_aggregation_families(results, graph_out_dir)
    _print_top_temporal_metrics(results, h_cols)
    
    return results


def _get_aggregation_family(agg: str) -> str:
    """Map aggregation method to its family."""
    if agg is None:
        return 'unknown'
    for family, methods in AGGREGATION_FAMILIES.items():
        if agg in methods:
            return family
    return 'other'


def _analyze_aggregation_families(results: pd.DataFrame, graph_out_dir: str):
    """Analyze performance by aggregation family."""
    if results.empty:
        return
    
    sig_results = results[results['q_value'] < 0.05]
    if sig_results.empty:
        print("No significant correlations found")
        return
    
    print(f"\n  Performance by Aggregation Family:")
    family_stats = sig_results.groupby('aggregation_family').agg({
        'rho': [lambda x: x.abs().mean(), 'count'],
        'metric': 'nunique'
    })
    family_stats.columns = ['mean_abs_rho', 'n_significant', 'n_unique_metrics']
    family_stats = family_stats.sort_values('mean_abs_rho', ascending=False)
    
    print(f"  {'Family':<15} {'Mean |ρ|':<12} {'# Sig':<10}")
    print(f"  {'-'*37}")
    for family, row in family_stats.iterrows():
        print(f"  {family:<15} {row['mean_abs_rho']:.3f}        {int(row['n_significant']):<10}")
    
    _plot_family_comparison(results, graph_out_dir)


def _plot_family_comparison(results: pd.DataFrame, graph_out_dir: str):
    """Plot aggregation family comparison."""
    sig_results = results[results['q_value'] < 0.05]
    if sig_results.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    family_stats = sig_results.groupby('aggregation_family')['rho'].apply(lambda x: x.abs().mean())
    family_stats = family_stats.sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(family_stats)))
    ax.barh(range(len(family_stats)), family_stats.values, color=colors)
    ax.set_yticks(range(len(family_stats)))
    ax.set_yticklabels(family_stats.index)
    ax.set_xlabel('Mean |Spearman ρ|')
    ax.set_title('Aggregation Family Performance')
    ax.grid(True, alpha=0.3, axis='x')
    
    ax = axes[1]
    agg_stats = sig_results.groupby('aggregation')['rho'].apply(lambda x: x.abs().mean())
    agg_stats = agg_stats.sort_values(ascending=True).tail(15)
    ax.barh(range(len(agg_stats)), agg_stats.values, color='steelblue')
    ax.set_yticks(range(len(agg_stats)))
    ax.set_yticklabels(agg_stats.index)
    ax.set_xlabel('Mean |Spearman ρ|')
    ax.set_title('Top 15 Aggregation Methods')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, 'part4_aggregation_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: part4_aggregation_comparison.png")


def _print_top_temporal_metrics(results: pd.DataFrame, h_cols: List[str]):
    """Print top temporal metrics at primary horizon."""
    if results.empty:
        return
    
    print(f"\n  Top 5 Temporal Metrics per Regime at h={PRIMARY_HORIZON}:")
    
    for regime in sorted(results['regime'].unique()):
        print(f"\n    Regime: {regime}")
        h_data = results[(results['regime'] == regime) & (results['horizon_num'] == PRIMARY_HORIZON)]
        top = h_data.sort_values('rho', key=abs, ascending=False).head(5)
        
        for _, row in top.iterrows():
            metric_short = row['metric'][:50] + "..." if len(row['metric']) > 50 else row['metric']
            q_str = f"q={row['q_value']:.4f}" if row['q_value'] < 0.05 else "NS"
            print(f"      ρ={row['rho']:+.3f} ({q_str}): {metric_short}")


# -------------------------
# Part 5: Cross-Regime Consistency
# -------------------------

def part5_cross_regime_consistency(
    atomic_results: pd.DataFrame,
    temporal_results: pd.DataFrame,
    regime_col: str,
    csv_out_dir: str,
    graph_out_dir: str
):
    """Analyze which metrics are consistently predictive across regimes."""
    all_results = pd.concat([atomic_results, temporal_results], ignore_index=True)
    
    if all_results.empty:
        print("No results to analyze")
        return
    
    regimes = sorted(all_results['regime'].unique())
    n_regimes = len(regimes)
    print(f"\nAnalyzing consistency across {n_regimes} regimes")
    
    h_data = all_results[all_results['horizon_num'] == PRIMARY_HORIZON]
    sig_data = h_data[h_data['q_value'] < 0.05]
    
    # Metrics significant in ALL regimes with SAME SIGN
    metric_stats = []
    for metric in h_data['metric'].unique():
        metric_data = h_data[h_data['metric'] == metric]
        sig_in = (metric_data['q_value'] < 0.05).sum()
        
        if sig_in == n_regimes:
            rhos = metric_data['rho'].values
            same_sign = all(r > 0 for r in rhos) or all(r < 0 for r in rhos)
            metric_stats.append({
                'metric': metric,
                'mean_rho': rhos.mean(),
                'std_rho': rhos.std(),
                'n_regimes_sig': sig_in,
                'same_sign': same_sign,
            })
    
    if metric_stats:
        consistent_df = pd.DataFrame(metric_stats)
        same_sign_df = consistent_df[consistent_df['same_sign']]
        
        print(f"\nMetrics significant in ALL regimes: {len(consistent_df)}")
        print(f"  ...with SAME SIGN (most reliable): {len(same_sign_df)}")
        
        if len(same_sign_df) > 0:
            same_sign_df = same_sign_df.sort_values('mean_rho')
            print(f"\n  Top same-sign consistent metrics (negative ρ = split helps):")
            for _, row in same_sign_df.head(10).iterrows():
                m_short = row['metric'][:45] + "..." if len(row['metric']) > 45 else row['metric']
                print(f"    ρ={row['mean_rho']:+.3f} (±{row['std_rho']:.3f}): {m_short}")
            
            same_sign_df.to_csv(os.path.join(csv_out_dir, 'part5_cross_regime_consistent.csv'), index=False)
            print(f"Saved: part5_cross_regime_consistent.csv")
    
    _plot_consistency_heatmap(h_data, regimes, graph_out_dir, PRIMARY_HORIZON)


def _shorten_metric_name(metric: str, max_len: int = 30) -> str:
    """Create a short but informative metric label."""
    if not metric.startswith('tmp'):
        # Atomic metric - just truncate
        return metric[:max_len]
    
    parts = metric.split('__')
    if len(parts) >= 3:
        window = parts[0]  # e.g., 'tmp32'
        agg = parts[1]     # e.g., 'pct_change_last_first'
        base = parts[2]    # e.g., 'input_weights_var'
        
        # Shorten base metric name
        base_abbrev = {
            'input_weights_mean': 'in_w_μ',
            'input_weights_var': 'in_w_σ²',
            'output_weights_mean': 'out_w_μ',
            'output_weights_var': 'out_w_σ²',
            'bias': 'bias',
            'pre_activation_mean': 'pre_act',
            'post_activation_mean': 'post_act',
            'input_weight_grads_mean': 'in_g_μ',
            'input_weight_grads_var': 'in_g_σ²',
            'output_weight_grads_mean': 'out_g_μ',
            'output_weight_grads_var': 'out_g_σ²',
            'pre_activation_grad_mean': 'pre_g',
            'post_activation_grad_mean': 'post_g',
        }
        base_short = base_abbrev.get(base, base[:8])
        
        # Shorten aggregation
        agg_abbrev = {
            'pct_change_last_first': 'pct_Δ',
            'abs_mean_second_diff': 'abs_2nd',
            'delta_last_first': 'Δ_lf',
            'slope_ols': 'slope',
            'half_diff_mean': 'half_Δ',
            'ema_mean_hl': 'ema',
        }
        agg_short = agg_abbrev.get(agg, agg[:6])
        
        return f"{window}_{agg_short}_{base_short}"
    
    return metric[:max_len]


def _plot_consistency_heatmap(results: pd.DataFrame, regimes: List, graph_out_dir: str, h_num: int):
    """Plot heatmap showing metric correlations across regimes."""
    sig_results = results[results['q_value'] < 0.05]
    if sig_results.empty:
        return
    
    metric_avg_rho = sig_results.groupby('metric')['rho'].apply(lambda x: x.abs().mean())
    top_metrics = metric_avg_rho.sort_values(ascending=False).head(30).index.tolist()
    
    presence_matrix = np.zeros((len(top_metrics), len(regimes)))
    for i, metric in enumerate(top_metrics):
        for j, regime in enumerate(regimes):
            metric_regime = results[(results['metric'] == metric) & (results['regime'] == regime)]
            if len(metric_regime) > 0:
                presence_matrix[i, j] = metric_regime.iloc[0]['rho']
    
    # Use improved label shortening
    short_names = [_shorten_metric_name(m) for m in top_metrics]
    
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(presence_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes)
    ax.set_yticks(range(len(top_metrics)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Regime')
    ax.set_title(f'Top 30 Metrics by Regime (h={h_num})')
    plt.colorbar(im, label='Spearman ρ', shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, 'part5_consistency_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: part5_consistency_heatmap.png")


# -------------------------
# Part 6: Horizon Analysis
# -------------------------

def part6_horizon_analysis(results: pd.DataFrame, h_cols: List[str], csv_out_dir: str, graph_out_dir: str):
    """Analyze how correlations change across horizons."""
    if results.empty:
        return
    
    sig_results = results[results['q_value'] < 0.05]
    if sig_results.empty:
        print("No significant correlations")
        return
    
    horizon_stats = sig_results.groupby('horizon_num').agg({
        'rho': [lambda x: x.abs().mean(), lambda x: x.abs().max(), 'count'],
        'metric': 'nunique'
    })
    horizon_stats.columns = ['mean_abs_rho', 'max_abs_rho', 'n_significant', 'n_unique_metrics']
    
    print(f"\n  Correlation Summary by Horizon:")
    print(f"  {'H':<5} {'Mean |ρ|':<10} {'Max |ρ|':<10} {'# Sig':<8}")
    print(f"  {'-'*33}")
    for h, row in horizon_stats.iterrows():
        print(f"  {h:<5} {row['mean_abs_rho']:.3f}      {row['max_abs_rho']:.3f}      {int(row['n_significant']):<8}")
    
    horizon_stats.to_csv(os.path.join(csv_out_dir, 'part6_horizon_summary.csv'))
    _plot_horizon_trends(results, graph_out_dir)


def _plot_horizon_trends(results: pd.DataFrame, graph_out_dir: str):
    """Plot correlation trends across horizons."""
    sig_results = results[results['q_value'] < 0.05]
    if sig_results.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    for regime in sorted(results['regime'].unique()):
        regime_sig = sig_results[sig_results['regime'] == regime]
        h_means = regime_sig.groupby('horizon_num')['rho'].apply(lambda x: x.abs().mean())
        ax.plot(h_means.index, h_means.values, marker='o', label=regime)
    ax.axvline(PRIMARY_HORIZON, color='red', linestyle='--', alpha=0.5, label=f'Primary (h={PRIMARY_HORIZON})')
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Mean |Spearman ρ|')
    ax.set_title('Correlation Strength by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for regime in sorted(results['regime'].unique()):
        regime_sig = sig_results[sig_results['regime'] == regime]
        h_counts = regime_sig.groupby('horizon_num').size()
        ax.plot(h_counts.index, h_counts.values, marker='s', label=regime)
    ax.set_xlabel('Horizon')
    ax.set_ylabel('# Significant Metrics')
    ax.set_title('Number of Predictive Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    if 'aggregation_family' in sig_results.columns:
        for family in ['extremes', 'spread', 'central'][:3]:
            family_data = sig_results[sig_results['aggregation_family'] == family]
            if len(family_data) > 0:
                h_means = family_data.groupby('horizon_num')['rho'].apply(lambda x: x.abs().mean())
                ax.plot(h_means.index, h_means.values, marker='o', label=family)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Mean |Spearman ρ|')
        ax.set_title('Top Aggregation Families')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if 'temporal_window' in sig_results.columns:
        for window in sorted([w for w in sig_results['temporal_window'].unique() if w is not None]):
            window_data = sig_results[sig_results['temporal_window'] == window]
            h_means = window_data.groupby('horizon_num')['rho'].apply(lambda x: x.abs().mean())
            if len(h_means) > 0:
                ax.plot(h_means.index, h_means.values, marker='o', label=f'w={window}')
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Mean |Spearman ρ|')
        ax.set_title('Temporal Windows')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Horizon Trend Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, 'part6_horizon_trends.png'), dpi=150)
    plt.close()
    print(f"Saved: part6_horizon_trends.png")


# -------------------------
# Part 7: Cross-Dataset Transfer
# -------------------------

def part7_cross_dataset_transfer(
    results_dict1: Dict,
    results_dict2: Dict,
    csv_out_dir: str,
    graph_out_dir: str
):
    """
    CRITICAL VALIDATION: Test if metrics that predict on one dataset also predict on another.
    """
    name1 = results_dict1['dataset_name']
    name2 = results_dict2['dataset_name']
    
    print(f"\n{'='*70}")
    print(f" PART 7: CROSS-DATASET TRANSFER VALIDATION")
    print(f" {name1} vs {name2}")
    print(f"{'='*70}")
    
    r1 = pd.concat([results_dict1['atomic_results'], results_dict1['temporal_basic_results']], ignore_index=True)
    r2 = pd.concat([results_dict2['atomic_results'], results_dict2['temporal_basic_results']], ignore_index=True)
    
    r1_h = r1[r1['horizon_num'] == PRIMARY_HORIZON]
    r2_h = r2[r2['horizon_num'] == PRIMARY_HORIZON]
    
    r1_avg = r1_h.groupby('metric')['rho'].mean().rename(f'rho_{name1}')
    r2_avg = r2_h.groupby('metric')['rho'].mean().rename(f'rho_{name2}')
    
    combined = pd.concat([r1_avg, r2_avg], axis=1).dropna()
    
    if len(combined) < 10:
        print("Insufficient overlapping metrics for transfer analysis")
        return
    
    print(f"\nOverlapping metrics analyzed: {len(combined)}")
    
    rho_transfer, p_transfer = spearmanr(combined[f'rho_{name1}'], combined[f'rho_{name2}'])
    print(f"\nCORRELATION OF CORRELATIONS:")
    print(f"  Spearman ρ = {rho_transfer:.3f} (p = {p_transfer:.2e})")
    
    if rho_transfer > 0.5:
        print(f"  → STRONG positive transfer: metrics predictive on {name1} also predictive on {name2}")
    elif rho_transfer > 0.3:
        print(f"  → MODERATE positive transfer")
    elif rho_transfer > 0:
        print(f"  → WEAK positive transfer")
    else:
        print(f"  → NO transfer or negative transfer (findings may not generalize)")
    
    r1_sig = r1_h[r1_h['q_value'] < 0.05].groupby('metric')['rho'].mean()
    r2_sig = r2_h[r2_h['q_value'] < 0.05].groupby('metric')['rho'].mean()
    
    both_sig = set(r1_sig.index) & set(r2_sig.index)
    same_sign_metrics = []
    
    for m in both_sig:
        if (r1_sig[m] > 0 and r2_sig[m] > 0) or (r1_sig[m] < 0 and r2_sig[m] < 0):
            same_sign_metrics.append({
                'metric': m,
                f'rho_{name1}': r1_sig[m],
                f'rho_{name2}': r2_sig[m],
                'mean_rho': (r1_sig[m] + r2_sig[m]) / 2,
            })
    
    print(f"\nMETRICS SIGNIFICANT ON BOTH DATASETS:")
    print(f"  Total: {len(both_sig)}")
    print(f"  With same sign (TRANSFERABLE): {len(same_sign_metrics)}")
    
    if same_sign_metrics:
        transfer_df = pd.DataFrame(same_sign_metrics).sort_values('mean_rho')
        
        print(f"\n  Top transferable metrics (negative ρ = split helps on both):")
        for _, row in transfer_df.head(10).iterrows():
            m_short = row['metric'][:40] + "..." if len(row['metric']) > 40 else row['metric']
            print(f"    {name1}: {row[f'rho_{name1}']:+.3f}, {name2}: {row[f'rho_{name2}']:+.3f} → {m_short}")
        
        transfer_df.to_csv(os.path.join(csv_out_dir, 'part7_transferable_metrics.csv'), index=False)
        print(f"\nSaved: part7_transferable_metrics.csv")
    
    _plot_transfer_scatter(combined, name1, name2, rho_transfer, graph_out_dir)
    
    combined.to_csv(os.path.join(csv_out_dir, 'part7_cross_dataset_comparison.csv'))
    print(f"Saved: part7_cross_dataset_comparison.csv")


def _plot_transfer_scatter(combined: pd.DataFrame, name1: str, name2: str, rho: float, graph_out_dir: str):
    """Plot scatter of correlations across datasets."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = combined[f'rho_{name1}']
    y = combined[f'rho_{name2}']
    
    ax.scatter(x, y, alpha=0.5, s=30)
    
    lims = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect transfer')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    ax.fill_between([-1, 0], [-1, -1], [0, 0], alpha=0.1, color='green', 
                    label='Both predict improvement')
    ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.1, color='red',
                    label='Both predict harm')
    
    ax.set_xlabel(f'Spearman ρ on {name1}')
    ax.set_ylabel(f'Spearman ρ on {name2}')
    ax.set_title(f'Cross-Dataset Transfer (r={rho:.3f})\nAt h={PRIMARY_HORIZON}')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(graph_out_dir, 'part7_transfer_scatter.png'), dpi=150)
    plt.close()
    print(f"Saved: part7_transfer_scatter.png")


# -------------------------
# Part 8: Practical Significance
# -------------------------

def part8_practical_significance(
    df: pd.DataFrame,
    atomic_results: pd.DataFrame,
    temporal_basic_results: pd.DataFrame,
    regime_col: str,
    h_cols: List[str],
    csv_out_dir: str,
    graph_out_dir: str
):
    """Decision-theoretic analysis: What's the practical value of using these metrics?"""
    h_col = f'delta_test_loss_at_h{PRIMARY_HORIZON}'
    if h_col not in df.columns:
        print(f"Primary horizon column {h_col} not found")
        return
    
    all_results = pd.concat([atomic_results, temporal_basic_results], ignore_index=True)
    if all_results.empty:
        return
    
    print(f"\n--- QUARTILE ANALYSIS: Selecting neurons by metric value ---")
    print(f"If you split only neurons in the TOP or BOTTOM quartile by metric,")
    print(f"what's the expected improvement vs random selection?")
    
    quartile_results = []
    
    for regime in sorted(df[regime_col].unique()):
        regime_df = df[df[regime_col] == regime].copy()
        regime_results = all_results[all_results['regime'] == regime]
        h_results = regime_results[regime_results['horizon_num'] == PRIMARY_HORIZON]
        
        best_neg = h_results.sort_values('rho').head(1)
        best_pos = h_results.sort_values('rho', ascending=False).head(1)
        
        print(f"\n  Regime: {regime}")
        
        baseline_delta = regime_df[h_col].mean()
        print(f"    Baseline (random): mean delta = {baseline_delta:.6f}")
        
        for best, direction in [(best_neg, 'high'), (best_pos, 'low')]:
            if len(best) == 0:
                continue
            
            metric = best.iloc[0]['metric']
            rho = best.iloc[0]['rho']
            
            if metric not in regime_df.columns:
                continue
            
            q25 = regime_df[metric].quantile(0.25)
            q75 = regime_df[metric].quantile(0.75)
            
            if direction == 'high':
                selected = regime_df[regime_df[metric] >= q75]
                quartile_name = 'top'
            else:
                selected = regime_df[regime_df[metric] <= q25]
                quartile_name = 'bottom'
            
            if len(selected) < 10:
                continue
            
            selected_delta = selected[h_col].mean()
            improvement = baseline_delta - selected_delta
            pct_improvement = (improvement / abs(baseline_delta)) * 100 if baseline_delta != 0 else 0
            
            m_short = metric[:35] + "..." if len(metric) > 35 else metric
            print(f"    Select {quartile_name} quartile by {m_short}:")
            print(f"      ρ = {rho:.3f}, n_selected = {len(selected)}")
            print(f"      Selected mean delta = {selected_delta:.6f}")
            print(f"      Improvement over random: {improvement:.6f} ({pct_improvement:+.1f}%)")
            
            quartile_results.append({
                'regime': regime,
                'metric': metric,
                'rho': rho,
                'selection': f'{quartile_name}_quartile',
                'n_selected': len(selected),
                'baseline_delta': baseline_delta,
                'selected_delta': selected_delta,
                'improvement': improvement,
                'pct_improvement': pct_improvement,
            })
    
    if quartile_results:
        pd.DataFrame(quartile_results).to_csv(
            os.path.join(csv_out_dir, 'part8_quartile_analysis.csv'), index=False
        )
        print(f"\nSaved: part8_quartile_analysis.csv")
    
    print(f"\n--- CLASSIFICATION ANALYSIS ---")
    print(f"If we use metric as classifier: 'split if metric > median'")
    print(f"Target: delta < 0 (split helped)")
    
    _classification_analysis(df, all_results, regime_col, h_col, csv_out_dir)


def _classification_analysis(
    df: pd.DataFrame,
    results: pd.DataFrame,
    regime_col: str,
    h_col: str,
    csv_out_dir: str
):
    """Treat metric as binary classifier for 'should split'."""
    classification_results = []
    
    for regime in sorted(df[regime_col].unique()):
        regime_df = df[df[regime_col] == regime].copy()
        regime_results = results[(results['regime'] == regime) & (results['horizon_num'] == PRIMARY_HORIZON)]
        
        truth = (regime_df[h_col] < 0).values
        base_rate = truth.mean()
        
        top_metrics = regime_results.sort_values('rho').head(3)['metric'].tolist()
        
        for metric in top_metrics:
            if metric not in regime_df.columns:
                continue
            
            median_val = regime_df[metric].median()
            rho = regime_results[regime_results['metric'] == metric]['rho'].iloc[0]
            
            if rho < 0:
                pred = (regime_df[metric] > median_val).values
            else:
                pred = (regime_df[metric] < median_val).values
            
            tp = (pred & truth).sum()
            fp = (pred & ~truth).sum()
            fn = (~pred & truth).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            classification_results.append({
                'regime': regime,
                'metric': metric,
                'rho': rho,
                'base_rate': base_rate,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
            
            m_short = metric[:30] + "..." if len(metric) > 30 else metric
            print(f"    {regime} - {m_short}:")
            print(f"      Base rate (% splits help): {base_rate*100:.1f}%")
            print(f"      Precision: {precision*100:.1f}%, Recall: {recall*100:.1f}%, F1: {f1:.3f}")
    
    if classification_results:
        pd.DataFrame(classification_results).to_csv(
            os.path.join(csv_out_dir, 'part8_classification_analysis.csv'), index=False
        )


# -------------------------
# Core Computation Functions
# -------------------------

def compute_within_regime_correlations(
    df: pd.DataFrame, 
    metric_cols: List[str], 
    h_cols: List[str],
    regime_col: str
) -> pd.DataFrame:
    """Compute Spearman correlations WITHIN each regime."""
    results = []
    regimes = sorted(df[regime_col].unique())
    
    for regime in regimes:
        regime_df = df[df[regime_col] == regime]
        n_samples = len(regime_df)
        
        if n_samples < 10:
            print(f"  Skipping {regime}: only {n_samples} samples")
            continue
        
        rho_df, q_df, n_df = _compute_spearman_pairwise(regime_df, metric_cols, h_cols)
        
        for metric in metric_cols:
            if metric not in rho_df.index:
                continue
            for h_col in h_cols:
                rho = rho_df.loc[metric, h_col]
                q = q_df.loc[metric, h_col]
                n = n_df.loc[metric, h_col]
                
                if np.isfinite(rho):
                    results.append({
                        'regime': regime,
                        'metric': metric,
                        'horizon': h_col,
                        'horizon_num': _hnum(h_col),
                        'rho': rho,
                        'q_value': q,
                        'n': int(n)
                    })
    
    return pd.DataFrame(results)


def _compute_spearman_pairwise(
    df: pd.DataFrame, 
    metric_cols: List[str], 
    h_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Spearman rho and BH-FDR q-values."""
    metric_cols = [c for c in metric_cols if c in df.columns]
    h_cols = [c for c in h_cols if c in df.columns]
    
    ranked = df[metric_cols + h_cols].rank(method="average")
    X = ranked[metric_cols].to_numpy(dtype=float)
    Y = ranked[h_cols].to_numpy(dtype=float)

    Zx, _, _ = _zscore_nan(X)
    Zy, _, _ = _zscore_nan(Y)

    n_metrics = Zx.shape[1]
    n_h = Zy.shape[1]

    rho = np.full((n_metrics, n_h), np.nan, dtype=float)
    n_eff = np.zeros((n_metrics, n_h), dtype=int)

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

    qvals = _bh_fdr(pvals)

    rho_df = pd.DataFrame(rho, index=metric_cols, columns=h_cols)
    n_df = pd.DataFrame(n_eff, index=metric_cols, columns=h_cols)
    q_df = pd.DataFrame(qvals, index=metric_cols, columns=h_cols)
    return rho_df, q_df, n_df


def _hnum(col: str) -> int:
    """Extract horizon number from column name."""
    try:
        return int(col.split("_h")[-1])
    except Exception:
        return 10**9


def _zscore_nan(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise z-score with NaN support."""
    m = np.nanmean(a, axis=0)
    s = np.nanstd(a, axis=0, ddof=1)
    s[s == 0] = np.nan
    z = (a - m) / s
    return z, m, s


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
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
    q_monotone = np.minimum.accumulate(q[order[::-1]])[::-1]
    q_adj = np.empty_like(pv)
    q_adj[order] = np.minimum(q_monotone, 1.0)
    q_full = np.full_like(p_flat, np.nan, dtype=float)
    q_full[valid] = q_adj
    return q_full.reshape(p.shape)


# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    # Get the experiment directory (where this script lives)
    experiment_dir = Path("experiments/x1_correlation")
    
    # Output directories (both inside experiments/x1_correlation/)
    csv_base_dir = experiment_dir / "output_local"    # Large CSV files (not committed)
    graph_base_dir = experiment_dir / "output"        # Graphs and logs (committed)
    
    datasets = [
        {"name": "FashionMNIST", "folder": "mnist", "csv_filename": "correlation_experiment_results_FashionMNIST.csv"},
        {"name": "CIFAR-10", "folder": "cifar-10", "csv_filename": "correlation_experiment_results_CIFAR10.csv"}
    ]
    
    # Run analysis for each dataset
    all_results = {}
    for dataset in datasets:
        # Input CSV is in output_local (where training saves it)
        csv_path = csv_base_dir / dataset['folder'] / dataset['csv_filename']
        # CSV outputs stay in output_local
        csv_out_dir = csv_base_dir / dataset['folder'] / "analysis"
        # Graph outputs go to output (committed)
        graph_out_dir = graph_base_dir / dataset['folder']
        
        if not csv_path.exists():
            print(f"\n{'='*70}")
            print(f"Skipping {dataset['name']}: CSV not found at {csv_path}")
            print(f"{'='*70}\n")
            continue
        
        all_results[dataset['name']] = main(str(csv_path), str(csv_out_dir), str(graph_out_dir))
    
    # =========================================================================
    # PART 7: Cross-Dataset Transfer (requires both datasets)
    # =========================================================================
    if len(all_results) >= 2:
        dataset_names = list(all_results.keys())
        csv_out_dir = csv_base_dir / "cross_dataset"
        graph_out_dir = graph_base_dir / "cross_dataset"
        csv_out_dir.mkdir(parents=True, exist_ok=True)
        graph_out_dir.mkdir(parents=True, exist_ok=True)
        
        part7_cross_dataset_transfer(
            all_results[dataset_names[0]],
            all_results[dataset_names[1]],
            str(csv_out_dir),
            str(graph_out_dir)
        )
    else:
        print("\n[Skipping Part 7: Need both datasets for cross-dataset transfer analysis]")
    
    print(f"\n{'='*70}")
    print(" ALL ANALYSES COMPLETE")
    print(f" CSV outputs (large, not committed): {csv_base_dir}/")
    print(f" Graph outputs (committed): {graph_base_dir}/")
    print(f"{'='*70}\n")
