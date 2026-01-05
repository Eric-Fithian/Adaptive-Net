"""
Step 3: Meta-Test Evaluation
Run the 4 variations on the 'Meta-Test' subset of OpenML datasets.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import random
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from anet import (
    split_neuron, ExactCopy, OrthogonalDecomp, 
    WidenableLinear, StatsWrapper, Trainer
)
from experiments.x3_openml.utils import get_dataset_splits, get_openml_dataset

def get_neuron_features(wrapper, neuron_idx, temporal_windows=[2, 4, 8, 16, 32]):
    """Extract features matching training format"""
    # Assuming Linear(0) -> Act -> Linear(2)
    # wrapper.model[0] is input layer, wrapper.model[2] is output layer
    # Index 0 and 2 are usually consistent for single hidden layer
    stats = wrapper.get_neuron_stats(0, 2, neuron_idx)
    for w in temporal_windows:
        stats.update(wrapper.get_neuron_stats_temporal(0, 2, neuron_idx, w))
    return stats

def run_evaluation(d_id, dataset_name, train_loader, test_loader, input_dim, n_classes, lr_model, device):
    results = []
    
    # Config
    EPOCHS = 30
    WARMUP_EPOCHS = 5
    STARTING_WIDTH = 50
    LR = 0.001
    
    # Split Schedule: Every 4 epochs starting from 6
    # [6, 10, 14, 18, 22, 26]
    SPLIT_START = 6
    SPLIT_INTERVAL = 4
    split_epochs = list(range(SPLIT_START, EPOCHS, SPLIT_INTERVAL))
    
    VARIATIONS = ["baseline", "random", "greedy", "anti-greedy"]
    N_INITS = 3 # Small number of seeds per dataset
    
    feature_columns = None
    if lr_model is not None:
         if hasattr(lr_model["classifier"], "feature_names_in_"):
            feature_columns = lr_model["classifier"].feature_names_in_
         elif hasattr(lr_model, "feature_names_in_"):
            feature_columns = lr_model.feature_names_in_

    print(f"Evaluating {dataset_name}...")
    
    # Calculate total runs for this dataset
    total_runs = len(VARIATIONS) * N_INITS
    
    with tqdm(total=total_runs, desc=f"  {dataset_name} runs", leave=False) as pbar:
        for var in VARIATIONS:
            for init_id in range(N_INITS):
                
                # Model Setup
                model = StatsWrapper(
                    nn.Sequential(
                        WidenableLinear(input_dim, STARTING_WIDTH),
                        nn.GELU(),
                        WidenableLinear(STARTING_WIDTH, n_classes),
                    ),
                    buffer_size=32 # max temporal window
                )
                
                trainer = Trainer(
                    model=model,
                    loss_fn=nn.CrossEntropyLoss(),
                    device=device,
                    lr=LR,
                    epochs=EPOCHS,
                    warmup_epochs=WARMUP_EPOCHS,
                    use_cosine=True
                )
                
                # Baseline: continuous train
                if var == "baseline":
                    res = trainer.fit(train_loader, test_loader, start_epoch=0, end_epoch=EPOCHS)
                    final_loss = res['test_losses_epoch'][-1]
                    
                else:
                    # Splitting
                    # 1. Train to first split
                    current_epoch = split_epochs[0]
                    res = trainer.fit(train_loader, test_loader, start_epoch=0, end_epoch=current_epoch)
                    final_loss = res['test_losses_epoch'][-1] if res['test_losses_epoch'] else float('inf')
                    
                    # Schedule segments
                    schedule = []
                    for i in range(len(split_epochs)):
                        start = split_epochs[i]
                        end = split_epochs[i+1] if i+1 < len(split_epochs) else EPOCHS
                        schedule.append((start, end))
                    
                    wrapper = trainer.model
                    
                    for start_seg, end_seg in schedule:
                        # Perform Split at start_seg
                        current_width = wrapper.model[0].out_features
                        target_idx = -1
                        
                        if var == "random":
                            target_idx = random.randint(0, current_width - 1)
                            
                        elif var in ["greedy", "anti-greedy"]:
                            # Extract candidates
                            candidates = []
                            for idx in range(current_width):
                                feats = get_neuron_features(wrapper, idx, [2, 4, 8, 16, 32])
                                feats['neuron_idx'] = idx
                                candidates.append(feats)
                                
                            df_c = pd.DataFrame(candidates)
                            
                            # Align cols
                            if feature_columns is not None:
                                for c in feature_columns:
                                    if c not in df_c.columns:
                                        df_c[c] = 0.0 # Impute
                                X_c = df_c[feature_columns]
                            else:
                                # Try to match blindly or fail?
                                # Should use whatever cols available if model doesn't specify
                                X_c = df_c
                                
                            try:
                                probs = lr_model.predict_proba(X_c)[:, 1]
                                if var == "greedy":
                                    target_idx = np.argmax(probs)
                                else:
                                    target_idx = np.argmin(probs)
                            except:
                                # Fallback
                                target_idx = random.randint(0, current_width - 1)
                                
                        # Action
                        split_neuron(
                            network=wrapper,
                            input_layer_idx=0,
                            output_layer_idx=2,
                            neuron_idx=target_idx,
                            input_splitter=ExactCopy(),
                            output_splitter=OrthogonalDecomp()
                        )
                        
                        # Add params
                        existing = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
                        new_params = [p for p in wrapper.parameters() if id(p) not in existing]
                        trainer.add_new_params_follow_scheduler(new_params)
                        
                        # Train next segment
                        res = trainer.fit(train_loader, test_loader, start_epoch=start_seg, end_epoch=end_seg)
                        if res['test_losses_epoch']:
                            final_loss = res['test_losses_epoch'][-1]
                            
                results.append({
                    'dataset': dataset_name,
                    'variation': var,
                    'init_id': init_id,
                    'final_test_loss': final_loss
                })
                pbar.update(1)
            
    return results


if __name__ == "__main__":
    experiment_dir = Path("experiments/x3_openml")
    model_path = experiment_dir / "output_local" / "policy_model.joblib"
    output_dir = experiment_dir / "output_local" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Policy Model...")
    try:
        lr_model = joblib.load(model_path)
    except:
        print("Model not found!")
        exit(1)
        
    _, test_ids = get_dataset_splits()
    print(f"Meta-Test Datasets: {len(test_ids)}")
    
    for d_id in tqdm(test_ids, desc="Eval Datasets"):
        dataset_name = f"openml_{d_id}"
        csv_path = output_dir / f"{dataset_name}_results.csv"
        
        if csv_path.exists():
            continue
            
        train_loader, test_loader, input_dim, n_classes = get_openml_dataset(d_id, DEVICE)
        if train_loader is None:
            continue
            
        results = run_evaluation(d_id, dataset_name, train_loader, test_loader, input_dim, n_classes, lr_model, DEVICE)
        
        pd.DataFrame(results).to_csv(csv_path, index=False)
        
    print("Evaluation Complete.")

