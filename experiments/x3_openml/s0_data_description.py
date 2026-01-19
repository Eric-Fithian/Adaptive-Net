"""
Step 0: Exploratory Data Analysis on OpenML-CC18 Datasets
Generates descriptive statistics about the datasets used in the x3_openml experiment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
from scipy import sparse
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from experiments.x3_openml.utils import CC18_DATASET_IDS, get_dataset_splits


def analyze_dataset(dataset_id: int) -> dict:
    """
    Fetch and analyze a single OpenML dataset.
    Returns a dictionary of descriptive statistics.
    """
    try:
        # Fetch the dataset with frame to preserve type info
        bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X_df = bunch.data
        y = bunch.target
        
        # Also get the raw array for variance calculations
        bunch_array = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')
        X_raw = bunch_array.data
        
        # Handle sparse matrices
        if sparse.issparse(X_raw):
            X_raw = X_raw.toarray()
        
        # Basic info
        n_samples = X_df.shape[0]
        n_features = X_df.shape[1]
        
        # Determine task type from OpenML metadata
        # CC18 is Curated Classification 18, so all are classification
        task_type = "classification"
        
        # Number of classes
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        n_classes = len(le.classes_)
        
        # Class balance (imbalance ratio: max_class_count / min_class_count)
        class_counts = np.bincount(y_encoded)
        class_balance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else np.inf
        
        # Feature types
        n_numeric = len(X_df.select_dtypes(include=['number']).columns)
        n_categorical = len(X_df.select_dtypes(include=['object', 'category']).columns)
        
        # Missing values
        n_missing = X_df.isna().sum().sum()
        missing_pct = (n_missing / (n_samples * n_features)) * 100
        
        # Compute variance statistics for numeric columns only
        numeric_cols = X_df.select_dtypes(include=['number'])
        if len(numeric_cols.columns) > 0:
            # Convert to numpy for variance calculation, handling NaNs
            numeric_values = numeric_cols.values.astype(float)
            col_variances = np.nanvar(numeric_values, axis=0)
            mean_variance = np.nanmean(col_variances)
            median_variance = np.nanmedian(col_variances)
            min_variance = np.nanmin(col_variances)
            max_variance = np.nanmax(col_variances)
        else:
            mean_variance = np.nan
            median_variance = np.nan
            min_variance = np.nan
            max_variance = np.nan
        
        # Dataset metadata from OpenML
        dataset_name = bunch.details.get('name', f'dataset_{dataset_id}') if hasattr(bunch, 'details') else f'dataset_{dataset_id}'
        
        return {
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'task_type': task_type,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes,
            'n_numeric_features': n_numeric,
            'n_categorical_features': n_categorical,
            'missing_pct': missing_pct,
            'class_balance_ratio': class_balance_ratio,
            'mean_variance': mean_variance,
            'median_variance': median_variance,
            'min_variance': min_variance,
            'max_variance': max_variance,
            'samples_per_feature': n_samples / n_features if n_features > 0 else np.nan,
            'samples_per_class': n_samples / n_classes if n_classes > 0 else np.nan,
        }
        
    except Exception as e:
        print(f"Error processing dataset {dataset_id}: {e}")
        return {
            'dataset_id': dataset_id,
            'dataset_name': f'FAILED_{dataset_id}',
            'task_type': 'unknown',
            'n_samples': np.nan,
            'n_features': np.nan,
            'n_classes': np.nan,
            'n_numeric_features': np.nan,
            'n_categorical_features': np.nan,
            'missing_pct': np.nan,
            'class_balance_ratio': np.nan,
            'mean_variance': np.nan,
            'median_variance': np.nan,
            'min_variance': np.nan,
            'max_variance': np.nan,
            'samples_per_feature': np.nan,
            'samples_per_class': np.nan,
            'error': str(e),
        }


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics from the dataset descriptions."""
    numeric_cols = [
        'n_samples', 'n_features', 'n_classes', 'n_numeric_features',
        'n_categorical_features', 'missing_pct', 'class_balance_ratio',
        'mean_variance', 'median_variance', 'samples_per_feature', 'samples_per_class'
    ]
    
    summary = df[numeric_cols].describe()
    return summary


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create and save visualizations of the dataset descriptions."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Dataset sizes distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Samples distribution
    ax = axes[0, 0]
    df['n_samples'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dataset Sizes (Samples)')
    ax.axvline(df['n_samples'].median(), color='red', linestyle='--', label=f"Median: {df['n_samples'].median():.0f}")
    ax.legend()
    
    # Features distribution
    ax = axes[0, 1]
    df['n_features'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Input Dimensions (Features)')
    ax.axvline(df['n_features'].median(), color='red', linestyle='--', label=f"Median: {df['n_features'].median():.0f}")
    ax.legend()
    
    # Classes distribution
    ax = axes[1, 0]
    df['n_classes'].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Number of Classes')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Output Dimensions (Classes)')
    ax.axvline(df['n_classes'].median(), color='red', linestyle='--', label=f"Median: {df['n_classes'].median():.0f}")
    ax.legend()
    
    # Feature types pie chart
    ax = axes[1, 1]
    total_numeric = df['n_numeric_features'].sum()
    total_categorical = df['n_categorical_features'].sum()
    ax.pie([total_numeric, total_categorical], labels=['Numeric', 'Categorical'], 
           autopct='%1.1f%%', colors=['steelblue', 'coral'])
    ax.set_title('Overall Feature Type Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Samples vs Features scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['n_features'], df['n_samples'], 
                        c=df['n_classes'], cmap='viridis', 
                        s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Number of Features (Input Dimension)')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Size: Samples vs Features (colored by # Classes)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.colorbar(scatter, label='Number of Classes')
    plt.savefig(output_dir / 'samples_vs_features.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Class balance distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    # Cap extreme values for visualization
    balance_capped = df['class_balance_ratio'].clip(upper=100)
    balance_capped.hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Class Balance Ratio (max_count/min_count, capped at 100)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Class Imbalance')
    ax.axvline(1, color='green', linestyle='--', linewidth=2, label='Perfectly Balanced')
    ax.axvline(balance_capped.median(), color='red', linestyle='--', label=f"Median: {balance_capped.median():.1f}")
    ax.legend()
    plt.savefig(output_dir / 'class_balance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Variance distribution (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_variance = df['mean_variance'].dropna()
    if len(valid_variance) > 0:
        # Use log scale for variance
        log_variance = np.log10(valid_variance.clip(lower=1e-10))
        log_variance.hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='teal')
        ax.set_xlabel('Log10(Mean Feature Variance)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Feature Variance (Log Scale)')
    plt.savefig(output_dir / 'variance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Missing data distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    df['missing_pct'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7, color='crimson')
    ax.set_xlabel('Missing Data Percentage')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Missing Data')
    n_complete = (df['missing_pct'] == 0).sum()
    ax.annotate(f'{n_complete} datasets have no missing data', 
                xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12)
    plt.savefig(output_dir / 'missing_data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Train/Test split visualization
    train_ids, test_ids = get_dataset_splits()
    df['split'] = df['dataset_id'].apply(lambda x: 'train' if x in train_ids else 'test')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for split, color in [('train', 'blue'), ('test', 'orange')]:
        subset = df[df['split'] == split]
        ax.scatter(subset['n_features'], subset['n_samples'], 
                  c=color, label=f'{split.capitalize()} ({len(subset)})', 
                  s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Train/Test Dataset Split')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(output_dir / 'train_test_split.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    experiment_dir = Path("experiments/x3_openml")
    output_dir = experiment_dir / "output" / "data_description"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing {len(CC18_DATASET_IDS)} OpenML-CC18 datasets...")
    print(f"Output directory: {output_dir}")
    
    # Collect statistics for each dataset
    results = []
    for dataset_id in tqdm(CC18_DATASET_IDS, desc="Analyzing datasets"):
        stats = analyze_dataset(dataset_id)
        results.append(stats)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add train/test split info
    train_ids, test_ids = get_dataset_splits()
    df['split'] = df['dataset_id'].apply(lambda x: 'train' if x in train_ids else 'test')
    
    # Save full dataset descriptions
    df.to_csv(output_dir / 'dataset_descriptions.csv', index=False)
    print(f"\nSaved dataset descriptions to {output_dir / 'dataset_descriptions.csv'}")
    
    # Create and save summary statistics
    summary = create_summary_statistics(df)
    summary.to_csv(output_dir / 'summary_statistics.csv')
    print(f"Saved summary statistics to {output_dir / 'summary_statistics.csv'}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("OPENML-CC18 DATASET SUMMARY")
    print("="*60)
    
    print(f"\nTotal datasets: {len(df)}")
    print(f"  - Successfully analyzed: {df['n_samples'].notna().sum()}")
    print(f"  - Failed: {df['n_samples'].isna().sum()}")
    
    print(f"\nTrain/Test Split:")
    print(f"  - Train: {len(train_ids)} datasets")
    print(f"  - Test: {len(test_ids)} datasets")
    
    print(f"\nTask Types:")
    print(df['task_type'].value_counts().to_string())
    
    print(f"\nDataset Sizes:")
    print(f"  Samples:  min={df['n_samples'].min():.0f}, max={df['n_samples'].max():.0f}, median={df['n_samples'].median():.0f}")
    print(f"  Features: min={df['n_features'].min():.0f}, max={df['n_features'].max():.0f}, median={df['n_features'].median():.0f}")
    print(f"  Classes:  min={df['n_classes'].min():.0f}, max={df['n_classes'].max():.0f}, median={df['n_classes'].median():.0f}")
    
    print(f"\nFeature Types (total across all datasets):")
    print(f"  Numeric:     {df['n_numeric_features'].sum():.0f} ({df['n_numeric_features'].sum() / (df['n_numeric_features'].sum() + df['n_categorical_features'].sum()) * 100:.1f}%)")
    print(f"  Categorical: {df['n_categorical_features'].sum():.0f} ({df['n_categorical_features'].sum() / (df['n_numeric_features'].sum() + df['n_categorical_features'].sum()) * 100:.1f}%)")
    
    print(f"\nMissing Data:")
    print(f"  Datasets with no missing data: {(df['missing_pct'] == 0).sum()}")
    print(f"  Datasets with >10% missing: {(df['missing_pct'] > 10).sum()}")
    print(f"  Mean missing percentage: {df['missing_pct'].mean():.2f}%")
    
    print(f"\nClass Balance (max/min class count ratio):")
    print(f"  Perfectly balanced (ratio=1): {(df['class_balance_ratio'] == 1).sum()}")
    print(f"  Moderately imbalanced (1-10): {((df['class_balance_ratio'] > 1) & (df['class_balance_ratio'] <= 10)).sum()}")
    print(f"  Highly imbalanced (>10): {(df['class_balance_ratio'] > 10).sum()}")
    
    print(f"\nFeature Variance (numeric features only):")
    print(f"  Mean variance across datasets: {df['mean_variance'].mean():.4f}")
    print(f"  Median variance across datasets: {df['mean_variance'].median():.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, output_dir)
    print(f"Saved visualizations to {output_dir}")
    
    # Save a text summary report
    report = []
    report.append("="*60)
    report.append("OPENML-CC18 DATASET EXPLORATORY DATA ANALYSIS")
    report.append("="*60)
    report.append("")
    report.append(f"Total datasets: {len(df)}")
    report.append(f"Successfully analyzed: {df['n_samples'].notna().sum()}")
    report.append(f"Failed to analyze: {df['n_samples'].isna().sum()}")
    report.append("")
    report.append("TRAIN/TEST SPLIT")
    report.append("-"*40)
    report.append(f"Train datasets: {len(train_ids)} (70%)")
    report.append(f"Test datasets:  {len(test_ids)} (30%)")
    report.append("")
    report.append("TASK TYPES")
    report.append("-"*40)
    report.append("All datasets are classification (CC18 = Curated Classification 18)")
    report.append("")
    report.append("DATASET SIZE STATISTICS")
    report.append("-"*40)
    report.append(f"Samples:  min={df['n_samples'].min():.0f}, max={df['n_samples'].max():.0f}, median={df['n_samples'].median():.0f}, mean={df['n_samples'].mean():.0f}")
    report.append(f"Features: min={df['n_features'].min():.0f}, max={df['n_features'].max():.0f}, median={df['n_features'].median():.0f}, mean={df['n_features'].mean():.0f}")
    report.append(f"Classes:  min={df['n_classes'].min():.0f}, max={df['n_classes'].max():.0f}, median={df['n_classes'].median():.0f}, mean={df['n_classes'].mean():.1f}")
    report.append("")
    report.append("FEATURE TYPE DISTRIBUTION")
    report.append("-"*40)
    total_features = df['n_numeric_features'].sum() + df['n_categorical_features'].sum()
    report.append(f"Numeric features:     {df['n_numeric_features'].sum():.0f} ({df['n_numeric_features'].sum() / total_features * 100:.1f}%)")
    report.append(f"Categorical features: {df['n_categorical_features'].sum():.0f} ({df['n_categorical_features'].sum() / total_features * 100:.1f}%)")
    report.append("")
    report.append("MISSING DATA")
    report.append("-"*40)
    report.append(f"Datasets with no missing data: {(df['missing_pct'] == 0).sum()} ({(df['missing_pct'] == 0).sum() / len(df) * 100:.1f}%)")
    report.append(f"Datasets with >10% missing:    {(df['missing_pct'] > 10).sum()} ({(df['missing_pct'] > 10).sum() / len(df) * 100:.1f}%)")
    report.append(f"Mean missing percentage:       {df['missing_pct'].mean():.2f}%")
    report.append("")
    report.append("CLASS BALANCE")
    report.append("-"*40)
    report.append(f"Perfectly balanced (ratio=1):   {(df['class_balance_ratio'] == 1).sum()}")
    report.append(f"Moderately imbalanced (1-10):   {((df['class_balance_ratio'] > 1) & (df['class_balance_ratio'] <= 10)).sum()}")
    report.append(f"Highly imbalanced (>10):        {(df['class_balance_ratio'] > 10).sum()}")
    report.append("")
    report.append("FEATURE VARIANCE (numeric features)")
    report.append("-"*40)
    report.append(f"Mean variance:   {df['mean_variance'].mean():.4f}")
    report.append(f"Median variance: {df['mean_variance'].median():.4f}")
    report.append("")
    report.append("FILES GENERATED")
    report.append("-"*40)
    report.append("- dataset_descriptions.csv: Full statistics for each dataset")
    report.append("- summary_statistics.csv: Aggregate statistics")
    report.append("- dataset_distributions.png: Histograms of key metrics")
    report.append("- samples_vs_features.png: Scatter plot of dataset sizes")
    report.append("- class_balance_distribution.png: Class imbalance histogram")
    report.append("- variance_distribution.png: Feature variance histogram")
    report.append("- missing_data_distribution.png: Missing data histogram")
    report.append("- train_test_split.png: Visualization of train/test split")
    report.append("")
    
    with open(output_dir / 'eda_report.txt', 'w') as f:
        f.write('\n'.join(report))
    print(f"Saved text report to {output_dir / 'eda_report.txt'}")
    
    print("\nData description complete!")

