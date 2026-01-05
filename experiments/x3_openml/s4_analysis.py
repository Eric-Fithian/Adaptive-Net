"""
Step 4: Meta-Analysis
Aggregate results from all test datasets.
Calculate relative improvement and win rates.
Compare between different split methods.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel

SPLIT_METHODS = ["half_noise", "orthogonal"]

if __name__ == "__main__":
    experiment_dir = Path("experiments/x3_openml")
    eval_dir = experiment_dir / "output_local" / "evaluation"
    output_dir = experiment_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results for each split method
    all_results = {}
    
    for split_method in SPLIT_METHODS:
        all_files = list(eval_dir.glob(f"*_{split_method}_results.csv"))
        if not all_files:
            print(f"No evaluation results found for {split_method}.")
            continue
            
        dfs = []
        for f in all_files:
            df = pd.read_csv(f)
            dfs.append(df)
            
        full_df = pd.concat(dfs, ignore_index=True)
        all_results[split_method] = full_df
        
    if not all_results:
        print("No evaluation results found for any split method.")
        exit(1)
    
    # Analyze each split method
    summary_tables = {}
    
    for split_method, full_df in all_results.items():
        print(f"\n{'='*60}")
        print(f"SPLIT METHOD: {split_method.upper()}")
        print(f"{'='*60}")
        
        # Aggregate by (dataset, variation) -> mean_loss
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
        imp_cols = [c for c in pivot.columns if 'imp_pct' in c]
        print(pivot[imp_cols].describe())
        
        # Win Rates
        print("\nWin Rates (Variation < Baseline):")
        for var in ['random', 'greedy', 'anti-greedy']:
            if var in pivot.columns:
                wins = (pivot[var] < pivot['baseline']).mean()
                print(f"  {var}: {wins*100:.1f}%")
                
        # Mean improvement
        print("\nMean % Improvement over Baseline:")
        for var in ['random', 'greedy', 'anti-greedy']:
            col = f'{var}_imp_pct'
            if col in pivot.columns:
                mean_imp = pivot[col].mean()
                print(f"  {var}: {mean_imp:.2f}%")
                
        # T-Tests (Paired, across datasets)
        print("\nPaired T-Tests (vs Baseline):")
        for var in ['random', 'greedy', 'anti-greedy']:
            if var in pivot.columns:
                data = pivot[['baseline', var]].dropna()
                t, p = ttest_rel(data['baseline'], data[var])
                print(f"  {var}: t={t:.3f}, p={p:.4e}")
        
        summary_tables[split_method] = pivot
    
    # Compare methods side-by-side
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN SPLIT METHODS")
    print(f"{'='*60}")
    
    comparison_data = []
    for split_method, pivot in summary_tables.items():
        for var in ['random', 'greedy', 'anti-greedy']:
            col = f'{var}_imp_pct'
            if col in pivot.columns:
                mean_imp = pivot[col].mean()
                comparison_data.append({
                    'split_method': split_method,
                    'strategy': var,
                    'mean_improvement_pct': mean_imp
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    if not comparison_df.empty:
        comparison_pivot = comparison_df.pivot(
            index='strategy', 
            columns='split_method', 
            values='mean_improvement_pct'
        )
        print("\nMean % Improvement by Split Method and Strategy:")
        print(comparison_pivot.to_string())
        
        # Key comparison: Greedy - Random differential
        print("\n[KEY METRIC] Greedy advantage over Random (Greedy_imp - Random_imp):")
        for split_method in SPLIT_METHODS:
            if split_method in comparison_pivot.columns:
                greedy_imp = comparison_pivot.loc['greedy', split_method] if 'greedy' in comparison_pivot.index else 0
                random_imp = comparison_pivot.loc['random', split_method] if 'random' in comparison_pivot.index else 0
                diff = greedy_imp - random_imp
                print(f"  {split_method}: {diff:+.2f}%")
    
    # Plotting: Create comparison visualization
    # Figure 1: Box plots for each split method (side by side)
    fig, axes = plt.subplots(1, len(summary_tables), figsize=(6*len(summary_tables), 6), sharey=True)
    if len(summary_tables) == 1:
        axes = [axes]
        
    for ax, (split_method, pivot) in zip(axes, summary_tables.items()):
        imp_cols = [c for c in pivot.columns if 'imp_pct' in c]
        plot_df = pivot[imp_cols].melt(var_name='Strategy', value_name='Improvement (%)')
        plot_df['Strategy'] = plot_df['Strategy'].str.replace('_imp_pct', '')
        
        sns.boxplot(data=plot_df, x='Strategy', y='Improvement (%)', ax=ax)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_title(f'{split_method.upper()} Method')
    
    plt.suptitle('Relative Improvement over Baseline by Split Method', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "meta_analysis_improvement.png", bbox_inches='tight')
    
    # Figure 2: Bar chart comparing methods
    if not comparison_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=comparison_df, x='strategy', y='mean_improvement_pct', hue='split_method')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Selection Strategy')
        plt.ylabel('Mean % Improvement over Baseline')
        plt.title('Comparison of Split Methods')
        plt.legend(title='Split Method')
        plt.tight_layout()
        plt.savefig(output_dir / "meta_analysis_method_comparison.png")
    
    # Save aggregate data
    for split_method, pivot in summary_tables.items():
        pivot.to_csv(output_dir / f"meta_analysis_summary_{split_method}.csv")
    
    if not comparison_df.empty:
        comparison_pivot.to_csv(output_dir / "meta_analysis_comparison.csv")
    
    print(f"\nSaved analysis to {output_dir}")
