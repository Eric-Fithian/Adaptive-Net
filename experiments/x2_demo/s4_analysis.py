"""
Script for Part 4 of the demo experiment (x2).
Analyzes the results of the comparative training on CIFAR-10.

- Input: `experiments/x2_demo/output/cifar10/comparison_results.csv`
- Output: Bar chart and significance tests.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def compute_ci(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


if __name__ == "__main__":
    experiment_dir = Path("experiments/x2_demo")
    input_csv = experiment_dir / "output" / "cifar10" / "comparison_results.csv"
    output_plot = experiment_dir / "output" / "cifar10" / "comparison_plot.png"
    output_stats = experiment_dir / "output" / "cifar10" / "significance_tests.txt"

    print(f"Loading results from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Run s3_cifar_comparison.py first.")
        exit(1)

    print(df.groupby("variation")["final_test_loss"].describe())

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="variation", y="final_test_loss", capsize=0.1, errorbar=("ci", 95)
    )
    
    # Adjust Y-axis to zoom in on differences
    min_val = df["final_test_loss"].min()
    max_val = df["final_test_loss"].max()
    margin = (max_val - min_val) * 0.2  # Add 20% margin
    if margin == 0: margin = 0.01
    
    plt.ylim(min_val - margin, max_val + margin)
    
    plt.title("Final Test Loss by Splitting Strategy (CIFAR-10)")
    plt.ylabel("Test Loss")
    plt.xlabel("Strategy")
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

    # Statistical Tests
    variations = df["variation"].unique()
    results = {}
    for v in variations:
        results[v] = df[df["variation"] == v]["final_test_loss"].values

    with open(output_stats, "w") as f:
        f.write("Significance Tests (T-test)\n")
        f.write("===========================\n\n")

        # Define pairs to compare
        pairs = []
        # Compare everything against baseline
        if "baseline" in variations:
            for v in variations:
                if v != "baseline":
                    pairs.append(("baseline", v))

        # Compare greedy vs random, greedy vs anti-greedy
        if "greedy" in variations and "random" in variations:
            pairs.append(("random", "greedy"))
        if "greedy" in variations and "anti-greedy" in variations:
            pairs.append(("anti-greedy", "greedy"))

        # Add any other missing pairwise comparisons if needed, or just do all vs all
        # Let's stick to the key hypotheses

        for v1, v2 in pairs:
            if v1 in results and v2 in results:
                t_stat, p_val = stats.ttest_ind(
                    results[v1], results[v2], equal_var=False
                )
                f.write(f"{v1} vs {v2}:\n")
                f.write(f"  t-statistic: {t_stat:.4f}\n")
                f.write(f"  p-value: {p_val:.4e}\n")
                f.write(f"  Significant (p < 0.05): {p_val < 0.05}\n\n")
            else:
                f.write(f"{v1} vs {v2}: Data missing\n\n")

    print(f"Stats saved to {output_stats}")
