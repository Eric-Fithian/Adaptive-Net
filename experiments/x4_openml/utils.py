"""
Utilities for x4 experiment: Deep networks with batch normalization.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Optional

from anet import WidenableLinear, StatsWrapper


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


def get_dataset_splits(seed=42):
    ids = sorted(CC18_DATASET_IDS)
    random.seed(seed)
    random.shuffle(ids)
    split_idx = int(len(ids) * 0.7)
    train_ids = ids[:split_idx]
    test_ids = ids[split_idx:]
    return train_ids, test_ids


def get_openml_dataset(dataset_id, device):
    """Fetch dataset from OpenML, preprocess, and return DataLoaders."""
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from scipy import sparse
    
    print(f"Fetching OpenML dataset ID={dataset_id}...")
    try:
        bunch = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')
        X, y = bunch.data, bunch.target
    except Exception as e:
        print(f"Failed to fetch dataset {dataset_id}: {e}")
        return None, None, None, None

    if sparse.issparse(X):
        X = X.toarray()

    try:
        df_X = pd.DataFrame(X)
        for col in df_X.columns:
            try:
                numeric_series = pd.to_numeric(df_X[col], errors='coerce')
                if numeric_series.isna().mean() < 0.5:
                    df_X[col] = numeric_series
            except:
                pass
        
        cat_cols = df_X.select_dtypes(include=['object', 'category']).columns
        num_cols = df_X.select_dtypes(include=['number']).columns
        
        transformers = []
        if len(num_cols) > 0:
            transformers.append(('num', SimpleImputer(strategy='mean'), num_cols))
        if len(cat_cols) > 0:
            transformers.append(
                ('cat', Pipeline([
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]), cat_cols)
            )
            
        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        X_processed = preprocessor.fit_transform(df_X)
        X = X_processed
        
    except Exception as e:
        print(f"Preprocessing failed for dataset {dataset_id}: {e}")
        return None, None, None, None

    if np.isnan(X).any():
        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
    
    batch_size = 128
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    input_dim = X.shape[1]
    return train_loader, test_loader, input_dim, n_classes


def build_deep_network(
    n_features: int,
    n_outputs: int,
    hidden_width: int,
    n_hidden_layers: int,
    buffer_size: Optional[int] = None,
) -> StatsWrapper:
    """
    Build a deep network with batch normalization.
    
    Architecture: [Linear -> BatchNorm -> GELU] x n_hidden_layers -> Linear
    
    Args:
        n_features: Number of input features
        n_outputs: Number of output classes
        hidden_width: Width of all hidden layers (same for all)
        n_hidden_layers: Number of hidden layers (3-8)
        buffer_size: Buffer size for StatsWrapper temporal tracking
    
    Returns:
        StatsWrapper containing the deep network
        
    Layer indexing for a network with N hidden layers:
        Layer 0: Linear(n_features -> hidden_width)
        Layer 1: BatchNorm1d(hidden_width)
        Layer 2: GELU
        Layer 3: Linear(hidden_width -> hidden_width)
        Layer 4: BatchNorm1d(hidden_width)
        Layer 5: GELU
        ...
        Layer 3*(N-1): Linear(hidden_width -> hidden_width)
        Layer 3*(N-1)+1: BatchNorm1d(hidden_width)
        Layer 3*(N-1)+2: GELU
        Layer 3*N: Linear(hidden_width -> n_outputs)  # Output layer
    """
    layers = []
    
    # First hidden layer: input -> hidden
    layers.append(WidenableLinear(n_features, hidden_width))
    layers.append(nn.BatchNorm1d(hidden_width))
    layers.append(nn.GELU())
    
    # Middle hidden layers: hidden -> hidden
    for _ in range(n_hidden_layers - 1):
        layers.append(WidenableLinear(hidden_width, hidden_width))
        layers.append(nn.BatchNorm1d(hidden_width))
        layers.append(nn.GELU())
    
    # Output layer: hidden -> output
    layers.append(WidenableLinear(hidden_width, n_outputs))
    
    model = StatsWrapper(nn.Sequential(*layers), buffer_size=buffer_size)
    return model


def get_splittable_layer_pairs(n_hidden_layers: int) -> List[Tuple[int, int]]:
    """
    Get valid (input_layer_idx, output_layer_idx) pairs for splitting.
    
    We skip the first two hidden layers to ensure we only split layers 
    that have learned high-level features.
    
    For a network with N hidden layers (indexed 0 to N-1):
    - Hidden layers 0 and 1 are NOT splittable
    - Hidden layers 2 through N-1 ARE splittable
    
    Args:
        n_hidden_layers: Total number of hidden layers in the network
        
    Returns:
        List of (input_layer_idx, output_layer_idx) tuples for valid splits
        
    Example for n_hidden_layers=5:
        Hidden layer 2: input_idx=6, output_idx=9
        Hidden layer 3: input_idx=9, output_idx=12
        Hidden layer 4: input_idx=12, output_idx=15 (output layer)
    """
    splittable_pairs = []
    
    # Skip first two hidden layers (indices 0 and 1)
    # Splittable hidden layers: 2, 3, ..., n_hidden_layers - 1
    for hidden_idx in range(2, n_hidden_layers):
        # Each hidden layer block is 3 layers: Linear, BatchNorm, GELU
        input_layer_idx = hidden_idx * 3
        
        # Output layer is the next Linear layer
        if hidden_idx == n_hidden_layers - 1:
            # Last hidden layer connects to output layer
            output_layer_idx = n_hidden_layers * 3
        else:
            # Connects to next hidden layer's Linear
            output_layer_idx = (hidden_idx + 1) * 3
        
        splittable_pairs.append((input_layer_idx, output_layer_idx))
    
    return splittable_pairs


def sample_network_depth(min_layers: int = 3, max_layers: int = 8) -> int:
    """Sample a random network depth."""
    return random.randint(min_layers, max_layers)


def sample_split_layer(n_hidden_layers: int) -> Tuple[int, int]:
    """
    Randomly sample a layer to split from valid splittable layers.
    
    Args:
        n_hidden_layers: Total number of hidden layers
        
    Returns:
        (input_layer_idx, output_layer_idx) tuple for the split
    """
    pairs = get_splittable_layer_pairs(n_hidden_layers)
    if not pairs:
        raise ValueError(f"No splittable layers for network with {n_hidden_layers} hidden layers")
    return random.choice(pairs)
