"""
Step 2: Meta-Training
Train the policy (Logistic Regression) on the collected meta-data.
Single split method: half_noise
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


def main():
    experiment_dir = Path("experiments/x4_openml")
    data_dir = experiment_dir / "output_local" / "data_collection"
    output_model = experiment_dir / "output_local" / "policy_model.joblib"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading meta-training data...")
    all_files = list(data_dir.glob("*.csv"))
    
    if not all_files:
        print("No data files found. Run s1 first.")
        return
        
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['source_dataset'] = f.stem
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(full_df)}")
    
    target_col = "delta_test_loss_at_h7"
    
    if target_col not in full_df.columns:
        print(f"Target {target_col} not found. Available keys: {list(full_df.columns[:10])}")
        cols = [c for c in full_df.columns if "delta_test_loss" in c]
        if cols:
            target_col = cols[-1]
            print(f"Falling back to {target_col}")
    
    full_df = full_df.dropna(subset=[target_col])
    
    y = (full_df[target_col] < 0).astype(int)
    print(f"Positive class (Improvement) rate: {y.mean():.3f}")
    
    exclude_cols = [
        "n_hidden_layers", "hidden_layer_split", "input_layer_idx", "output_layer_idx",
        "hidden_width", "init_id", "neuron_idx", "action_epoch", "source_dataset"
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
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(pipeline, output_model)
    print(f"Model saved to {output_model}")
    
    print("\n=== Analysis of network depth impact ===")
    if 'n_hidden_layers' in full_df.columns:
        depth_analysis = full_df.groupby('n_hidden_layers').agg({
            target_col: ['mean', 'std', 'count']
        }).round(4)
        print("Delta loss by network depth:")
        print(depth_analysis)
        
    if 'hidden_layer_split' in full_df.columns:
        layer_analysis = full_df.groupby('hidden_layer_split').agg({
            target_col: ['mean', 'std', 'count']
        }).round(4)
        print("\nDelta loss by split layer:")
        print(layer_analysis)


if __name__ == "__main__":
    main()
