"""
Step 2: Meta-Training
Train the policy (Logistic Regression) on the collected meta-data.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    experiment_dir = Path("experiments/x3_openml")
    data_dir = experiment_dir / "output_local" / "data_collection"
    output_model = experiment_dir / "output_local" / "policy_model.joblib"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading meta-training data...")
    all_files = list(data_dir.glob("*.csv"))
    
    if not all_files:
        print("No data files found. Run s1 first.")
        exit(1)
        
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Add source dataset ID just in case
            df['source_dataset'] = f.stem
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    
    # Target: delta_test_loss_at_h7 (8 epochs later)
    # Note: s1 uses 'temporal_windows': [8] and splits s.t. we have enough epochs.
    # So h0 is +1 epoch, h7 is +8 epochs.
    target_col = "delta_test_loss_at_h7"
    
    if target_col not in full_df.columns:
        print(f"Target {target_col} not found. Available keys: {full_df.columns[:10]}")
        # Fallback if names differ slightly
        cols = [c for c in full_df.columns if "delta_test_loss" in c]
        if cols:
            target_col = cols[-1] # Pick the longest horizon available
            print(f"Falling back to {target_col}")
    
    full_df = full_df.dropna(subset=[target_col])
    
    # Binary Target: Improvement (delta < 0)
    y = (full_df[target_col] < 0).astype(int)
    print(f"Positive class (Improvement) rate: {y.mean():.3f}")
    
    # Features
    exclude_cols = [
        "regime_name", "starting_width", "init_id", "neuron_idx", "action_epoch",
        "source_dataset"
    ]
    exclude_cols.extend([c for c in full_df.columns if "delta_test_loss" in c])
    
    feature_cols = [c for c in full_df.columns if c not in exclude_cols]
    X = full_df[feature_cols]
    
    print(f"Training on {len(feature_cols)} features.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Eval
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(pipeline, output_model)
    print(f"Model saved to {output_model}")

