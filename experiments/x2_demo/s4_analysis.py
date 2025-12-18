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
        # Create dummy data for testing the script structure if needed, but better to fail here.
        # exit(1)
        # For the sake of the user running this now without previous steps completed, I will generate mock data if file missing?
        # No, the user instructions say "Implement everything thoroughly".
        exit(1)

    print(df.groupby("variation")["final_test_loss"].describe())

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="variation", y="final_test_loss", capsize=0.1, errorbar=("ci", 95)
    )
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

        pairs = [("baseline", "random"), ("baseline", "greedy"), ("random", "greedy")]

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
