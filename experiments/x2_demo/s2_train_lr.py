"""
Script for Part 2 of the demo experiment (x2).
Trains a logistic regression model to predict test loss improvement.

- Input: `experiments/x2_demo/output/mnist/training_metrics.csv`
- Target: Probability of improving test loss at horizon 8 (delta_test_loss < 0)
- Features: Neuron statistics
- Output: `experiments/x2_demo/output/mnist/lr_model.joblib`
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    experiment_dir = Path("experiments/x2_demo")
    input_csv = experiment_dir / "output" / "mnist" / "training_metrics.csv"
    output_model = experiment_dir / "output" / "mnist" / "lr_model.joblib"

    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Run s1_dataset_creation.py first.")
        exit(1)

    print(f"Data shape: {df.shape}")

    # Define target
    # Horizon 8 corresponds to h0 because we only requested one window [8]
    if "delta_test_loss_at_h0" not in df.columns:
        print("Error: 'delta_test_loss_at_h0' not found in columns.")
        print("Available columns:", df.columns.tolist())
        exit(1)

    # Target: 1 if delta < 0 (improvement), 0 otherwise
    # First drop rows where target is NaN
    df = df.dropna(subset=["delta_test_loss_at_h0"])
    y = (df["delta_test_loss_at_h0"] < 0).astype(int)
    print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")

    # Define features
    # Exclude metadata and targets
    exclude_cols = [
        "regime_name",
        "starting_width",
        "init_id",
        "neuron_idx",
        "action_epoch",
        "delta_test_loss_at_h0",
    ]
    # Also exclude any other potential target columns if they exist
    exclude_cols.extend([c for c in df.columns if "delta_test_loss" in c])

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    print(f"Number of features: {len(feature_cols)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline
    # Impute NaNs with mean, scale features, then LR
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # Train
    print("Training Logistic Regression model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, output_model)
    print(f"\nModel saved to {output_model}")
