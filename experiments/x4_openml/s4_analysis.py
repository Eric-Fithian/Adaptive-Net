"""
Step 4: Meta-Analysis
Aggregate results from all test datasets.
Calculate relative improvement and win rates.
Analyze impact of network depth.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel


def main():
    experiment_dir = Path("experiments/x4_openml")
    eval_dir = experiment_dir / "output_local" / "evaluation"
    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_files = list(eval_dir.glob("*_results.csv"))
    if not all_files:
        print("No evaluation results found. Run s3 first.")
        return
        
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    print(f"{'='*60}")
    print("DEEP NETWORKS - HALF SPLIT + NOISE")
    print(f"{'='*60}")
    print(f"Total evaluation runs: {len(full_df)}")
    
    agg = full_df.groupby(['dataset', 'variation'])['final_test_loss'].mean().reset_index()
    pivot = agg.pivot(index='dataset', columns='variation', values='final_test_loss')
    
    for var in ['random', 'greedy', 'anti-greedy']:
        if var in pivot.columns:
            pivot[f'{var}_imp_pct'] = (pivot['baseline'] - pivot[var]) / pivot['baseline'] * 100
            
    print("\nSummary Statistics (Mean % Improvement over Baseline):")
    imp_cols = [c for c in pivot.columns if 'imp_pct' in c]
    print(pivot[imp_cols].describe())
    
    print("\nWin Rates (Variation < Baseline):")
    for var in ['random', 'greedy', 'anti-greedy']:
        if var in pivot.columns:
            wins = (pivot[var] < pivot['baseline']).mean()
            print(f"  {var}: {wins*100:.1f}%")
            
    print("\nMean % Improvement over Baseline:")
    for var in ['random', 'greedy', 'anti-greedy']:
        col = f'{var}_imp_pct'
        if col in pivot.columns:
            mean_imp = pivot[col].mean()
            print(f"  {var}: {mean_imp:.2f}%")
            
    print("\nPaired T-Tests (vs Baseline):")
    for var in ['random', 'greedy', 'anti-greedy']:
        if var in pivot.columns:
            data = pivot[['baseline', var]].dropna()
            t, p = ttest_rel(data['baseline'], data[var])
            print(f"  {var}: t={t:.3f}, p={p:.4e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS BY NETWORK DEPTH")
    print(f"{'='*60}")
    
    if 'n_hidden_layers' in full_df.columns:
        depth_analysis = full_df.groupby(['n_hidden_layers', 'variation']).agg({
            'final_test_loss': ['mean', 'std', 'count']
        }).round(4)
        print("\nLoss by network depth and variation:")
        print(depth_analysis)
        
        depth_pivot = full_df.pivot_table(
            index='n_hidden_layers',
            columns='variation',
            values='final_test_loss',
            aggfunc='mean'
        )
        
        for var in ['random', 'greedy', 'anti-greedy']:
            if var in depth_pivot.columns and 'baseline' in depth_pivot.columns:
                depth_pivot[f'{var}_imp_pct'] = (
                    (depth_pivot['baseline'] - depth_pivot[var]) / depth_pivot['baseline'] * 100
                )
        
        print("\n% Improvement over baseline by depth:")
        imp_cols_depth = [c for c in depth_pivot.columns if 'imp_pct' in c]
        print(depth_pivot[imp_cols_depth].round(2))
    
    print(f"\n{'='*60}")
    print("KEY METRICS")
    print(f"{'='*60}")
    
    if 'greedy_imp_pct' in pivot.columns and 'random_imp_pct' in pivot.columns:
        greedy_advantage = pivot['greedy_imp_pct'].mean() - pivot['random_imp_pct'].mean()
        print(f"\nGreedy advantage over Random: {greedy_advantage:+.2f}%")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    imp_cols = [c for c in pivot.columns if 'imp_pct' in c]
    plot_df = pivot[imp_cols].melt(var_name='Strategy', value_name='Improvement (%)')
    plot_df['Strategy'] = plot_df['Strategy'].str.replace('_imp_pct', '')
    
    sns.boxplot(data=plot_df, x='Strategy', y='Improvement (%)', ax=axes[0])
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Relative Improvement over Baseline')
    
    if 'n_hidden_layers' in full_df.columns:
        depth_means = full_df.groupby(['n_hidden_layers', 'variation'])['final_test_loss'].mean().reset_index()
        sns.lineplot(
            data=depth_means,
            x='n_hidden_layers',
            y='final_test_loss',
            hue='variation',
            marker='o',
            ax=axes[1]
        )
        axes[1].set_title('Loss by Network Depth')
        axes[1].set_xlabel('Number of Hidden Layers')
        axes[1].set_ylabel('Test Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / "meta_analysis_deep_networks.png", bbox_inches='tight')
    
    pivot.to_csv(output_dir / "meta_analysis_summary.csv")
    
    if 'n_hidden_layers' in full_df.columns:
        depth_summary = full_df.groupby('n_hidden_layers').agg({
            'final_test_loss': ['mean', 'std', 'count']
        })
        depth_summary.to_csv(output_dir / "depth_analysis.csv")
    
    print(f"\nSaved analysis to {output_dir}")


if __name__ == "__main__":
    main()
