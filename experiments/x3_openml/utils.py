
# OpenML-CC18 Benchmark
# https://www.openml.org/s/99

# Hardcoded list of task IDs for reproducibility and to avoid dependency on openml API query at runtime
# These are the task IDs for the OpenML-CC18 suite.
# Note: We use Task IDs (which define train/test splits usually), but here we just want the Dataset IDs.
# Actually, let's use Dataset IDs (d_id).
# Source: OpenML-CC18 definition.

CC18_DATASET_IDS = [
    3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188,
    300, 307, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1111, 1112, 1114, 1116, 1119,
    1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1168, 1216, 1217, 1218, 1233, 1235, 1236,
    1237, 1238, 1240, 1241, 1242, 1457, 1461, 1462, 1464, 1466, 1467, 1468, 1475, 1476, 1478,
    1479, 1480, 1485, 1486, 1487, 1489, 1491, 1492, 1493, 1494, 1497, 1501, 1504, 1510, 1515,
    1590, 23381, 40496, 40498, 40499, 40509, 40511, 40515, 40517, 40518, 40536, 40668, 40670,
    40685, 40691, 40701, 40923, 40927, 40966, 40975, 40978, 40979, 40981, 40982, 40983, 40984,
    40994, 40996, 41027, 4134, 4135, 42
]

# Split 70/30 deterministically
def get_dataset_splits(seed=42):
    import random
    ids = sorted(CC18_DATASET_IDS)
    random.seed(seed)
    random.shuffle(ids)
    
    split_idx = int(len(ids) * 0.7)
    train_ids = ids[:split_idx]
    test_ids = ids[split_idx:]
    
    return train_ids, test_ids

def get_openml_dataset(dataset_id, device):
    """
    Fetch dataset from OpenML, preprocess, and return DataLoaders.
    Standardizes input (mean=0, std=1).
    """
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    print(f"Fetching OpenML dataset ID={dataset_id}...")
    try:
        # fetch_openml handles caching automatically
        bunch = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')
        X, y = bunch.data, bunch.target
    except Exception as e:
        print(f"Failed to fetch dataset {dataset_id}: {e}")
        return None, None, None, None

    # Handle sparse matrices
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()

    # Handle categorical columns by using DataFrame + encoding
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    try:
        # Convert to DataFrame to handle mixed types safely
        # If X is already numpy array of object, this helps.
        # If fetch_openml returned array, we assume columns might be mixed.
        df_X = pd.DataFrame(X)
        
        # Identify categorical columns (object/category) vs numerical
        # Since we converted from numpy array, all might be object if mixed.
        # Attempt to infer numeric types safely
        for col in df_X.columns:
            try:
                # 'coerce' turns non-numeric to NaN. If a column is genuinely numeric mixed with strings,
                # this cleans it. If it's fully strings (categorical), it becomes all NaN.
                # We check if it converted successfully.
                numeric_series = pd.to_numeric(df_X[col], errors='coerce')
                # If we lost too much data (e.g. it was actually categorical), keep as object
                if numeric_series.isna().mean() < 0.5: # arbitrary threshold: if >50% data preserved
                    df_X[col] = numeric_series
            except:
                pass
        
        # Now define selectors
        cat_cols = df_X.select_dtypes(include=['object', 'category']).columns
        num_cols = df_X.select_dtypes(include=['number']).columns
        
        # Simple pipeline: Impute/Encode -> Standardize
        # Note: We do encoding first to get numbers, then StandardScaler later (or here).
        
        transformers = []
        if len(num_cols) > 0:
            transformers.append(
                ('num', SimpleImputer(strategy='mean'), num_cols)
            )
        if len(cat_cols) > 0:
            transformers.append(
                ('cat', Pipeline([
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]), cat_cols)
            )
            
        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        
        # This returns a numpy array of numbers
        X_processed = preprocessor.fit_transform(df_X)
        X = X_processed
        
    except Exception as e:
        print(f"Preprocessing failed for dataset {dataset_id}: {e}")
        return None, None, None, None

    # Final safe check for NaNs (should be handled by imputers above, but robust check)
    if np.isnan(X).any():
        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    # Normalize inputs
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to Tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    # Train/Test Split (Standard 80/20 for local training)
    # Note: This is splitting the SAMPLES of the dataset, 
    # not to be confused with splitting the DATASETS themselves.
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
    
    # DataLoaders
    batch_size = 128
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    input_dim = X.shape[1]
    
    return train_loader, test_loader, input_dim, n_classes

