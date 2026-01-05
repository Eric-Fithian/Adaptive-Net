"""
Step 4: Meta-Analysis
Aggregate results from all test datasets.
Calculate relative improvement and win rates.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel

if __name__ == "__main__":
    experiment_dir = Path("experiments/x3_openml")
    eval_dir = experiment_dir / "output_local" / "evaluation"
    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = list(eval_dir.glob("*_results.csv"))
    if not all_files:
        print("No evaluation results found.")
        exit(1)
        
    dfs = []
    for f in all_files:
        dfs.append(pd.read_csv(f))
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 1. Normalize results per dataset
    # Calculate % improvement over baseline for each run
    # Problem: Baseline and Variations have different init_ids (random seeds might differ, 
    # but usually we compare means).
    # Let's aggregate by (dataset, variation) -> mean_loss
    
    agg = full_df.groupby(['dataset', 'variation'])['final_test_loss'].mean().reset_index()
    
    # Pivot to have columns for each variation
    pivot = agg.pivot(index='dataset', columns='variation', values='final_test_loss')
    
    # Calculate relative improvement vs baseline
    # (baseline - var) / baseline * 100
    # Positive = Improvement
    for var in ['random', 'greedy', 'anti-greedy']:
        if var in pivot.columns:
            pivot[f'{var}_imp_pct'] = (pivot['baseline'] - pivot[var]) / pivot['baseline'] * 100
            
    print("\nSummary Statistics (Mean % Improvement over Baseline):")
    print(pivot[[c for c in pivot.columns if 'imp_pct' in c]].describe())
    
    # Win Rates
    print("\nWin Rates (Variation < Baseline):")
    for var in ['random', 'greedy', 'anti-greedy']:
        if var in pivot.columns:
            wins = (pivot[var] < pivot['baseline']).mean()
            print(f"{var}: {wins*100:.1f}%")
            
    # T-Tests (Paired, across datasets)
    print("\nPaired T-Tests (vs Baseline):")
    for var in ['random', 'greedy', 'anti-greedy']:
        if var in pivot.columns:
            # Dropna in case some runs failed
            data = pivot[['baseline', var]].dropna()
            t, p = ttest_rel(data['baseline'], data[var])
            print(f"{var}: t={t:.3f}, p={p:.4e}")
            
    # Plotting
    plot_df = pivot[[c for c in pivot.columns if 'imp_pct' in c]].melt(var_name='Strategy', value_name='Improvement (%)')
    plot_df['Strategy'] = plot_df['Strategy'].str.replace('_imp_pct', '')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x='Strategy', y='Improvement (%)')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Relative Improvement over Baseline across Datasets')
    plt.tight_layout()
    plt.savefig(output_dir / "meta_analysis_improvement.png")
    
    # Save aggregate
    pivot.to_csv(output_dir / "meta_analysis_summary.csv")
    print(f"Saved analysis to {output_dir}")

