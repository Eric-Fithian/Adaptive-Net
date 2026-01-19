"""
Step 3: Meta-Test Evaluation
Run variations on the 'Meta-Test' subset of OpenML datasets.
Uses deep networks with random depth and random split layer selection.
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from anet import split_neuron, ExactCopy, Half, WithNoise, Trainer
from experiments.x4_openml.utils import (
    get_dataset_splits, 
    get_openml_dataset,
    build_deep_network,
    sample_network_depth,
    sample_split_layer,
)


def get_neuron_features(wrapper, input_layer_idx, output_layer_idx, neuron_idx, temporal_windows=[2, 4, 8, 16, 32]):
    """Extract features matching training format for specified layer pair."""
    stats = wrapper.get_neuron_stats(input_layer_idx, output_layer_idx, neuron_idx)
    for w in temporal_windows:
        stats.update(wrapper.get_neuron_stats_temporal(input_layer_idx, output_layer_idx, neuron_idx, w))
    return stats


def run_evaluation(d_id, dataset_name, train_loader, test_loader, input_dim, n_classes, lr_model, device):
    """Run evaluation for a single dataset with deep networks."""
    results = []
    
    EPOCHS = 30
    WARMUP_EPOCHS = 5
    HIDDEN_WIDTH = 50
    LR = 0.001
    MIN_HIDDEN_LAYERS = 3
    MAX_HIDDEN_LAYERS = 8
    
    SPLIT_START = 6
    SPLIT_INTERVAL = 4
    split_epochs = list(range(SPLIT_START, EPOCHS, SPLIT_INTERVAL))
    
    VARIATIONS = ["baseline", "random", "greedy", "anti-greedy"]
    N_INITS = 3
    
    output_splitter = WithNoise(Half(), sigma_ratio=0.01)
    
    feature_columns = None
    if lr_model is not None:
        if hasattr(lr_model["classifier"], "feature_names_in_"):
            feature_columns = lr_model["classifier"].feature_names_in_
        elif hasattr(lr_model, "feature_names_in_"):
            feature_columns = lr_model.feature_names_in_

    total_runs = len(VARIATIONS) * N_INITS
    
    with tqdm(total=total_runs, desc=f"  {dataset_name} runs", leave=False) as pbar:
        for var in VARIATIONS:
            for init_id in range(N_INITS):
                n_hidden_layers = sample_network_depth(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS)
                
                model = build_deep_network(
                    n_features=input_dim,
                    n_outputs=n_classes,
                    hidden_width=HIDDEN_WIDTH,
                    n_hidden_layers=n_hidden_layers,
                    buffer_size=32
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
                
                if var == "baseline":
                    res = trainer.fit(train_loader, test_loader, start_epoch=0, end_epoch=EPOCHS)
                    final_loss = res['test_losses_epoch'][-1]
                else:
                    current_epoch = split_epochs[0]
                    res = trainer.fit(train_loader, test_loader, start_epoch=0, end_epoch=current_epoch)
                    final_loss = res['test_losses_epoch'][-1] if res['test_losses_epoch'] else float('inf')
                    
                    schedule = []
                    for i in range(len(split_epochs)):
                        start = split_epochs[i]
                        end = split_epochs[i+1] if i+1 < len(split_epochs) else EPOCHS
                        schedule.append((start, end))
                    
                    wrapper = trainer.model
                    
                    for start_seg, end_seg in schedule:
                        input_layer_idx, output_layer_idx = sample_split_layer(n_hidden_layers)
                        current_width = HIDDEN_WIDTH
                        target_idx = -1
                        
                        if var == "random":
                            target_idx = random.randint(0, current_width - 1)
                            
                        elif var in ["greedy", "anti-greedy"]:
                            candidates = []
                            for idx in range(current_width):
                                feats = get_neuron_features(
                                    wrapper, input_layer_idx, output_layer_idx, 
                                    idx, [2, 4, 8, 16, 32]
                                )
                                feats['neuron_idx'] = idx
                                candidates.append(feats)
                                
                            df_c = pd.DataFrame(candidates)
                            
                            if feature_columns is not None:
                                for c in feature_columns:
                                    if c not in df_c.columns:
                                        df_c[c] = 0.0
                                X_c = df_c[feature_columns]
                            else:
                                X_c = df_c
                                
                            try:
                                probs = lr_model.predict_proba(X_c)[:, 1]
                                if var == "greedy":
                                    target_idx = np.argmax(probs)
                                else:
                                    target_idx = np.argmin(probs)
                            except:
                                target_idx = random.randint(0, current_width - 1)
                        
                        split_neuron(
                            network=wrapper,
                            input_layer_idx=input_layer_idx,
                            output_layer_idx=output_layer_idx,
                            neuron_idx=target_idx,
                            input_splitter=ExactCopy(),
                            output_splitter=output_splitter
                        )
                        
                        existing = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
                        new_params = [p for p in wrapper.parameters() if id(p) not in existing]
                        trainer.add_new_params_follow_scheduler(new_params)
                        
                        res = trainer.fit(train_loader, test_loader, start_epoch=start_seg, end_epoch=end_seg)
                        if res['test_losses_epoch']:
                            final_loss = res['test_losses_epoch'][-1]
                            
                results.append({
                    'dataset': dataset_name,
                    'variation': var,
                    'init_id': init_id,
                    'n_hidden_layers': n_hidden_layers,
                    'final_test_loss': final_loss
                })
                pbar.update(1)
            
    return results


def main():
    experiment_dir = Path("experiments/x4_openml")
    output_dir = experiment_dir / "output_local" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
    _, test_ids = get_dataset_splits()
    print(f"Meta-Test Datasets: {len(test_ids)}")
    
    model_path = experiment_dir / "output_local" / "policy_model.joblib"
    print(f"Loading Policy Model: {model_path}")
    try:
        lr_model = joblib.load(model_path)
    except:
        print("Model not found! Run s2 first.")
        return
    
    for d_id in tqdm(test_ids, desc="Eval Datasets"):
        dataset_name = f"openml_{d_id}"
        csv_path = output_dir / f"{dataset_name}_results.csv"
        
        if csv_path.exists():
            continue
            
        train_loader, test_loader, input_dim, n_classes = get_openml_dataset(d_id, DEVICE)
        if train_loader is None:
            continue
            
        results = run_evaluation(
            d_id, dataset_name, train_loader, test_loader, 
            input_dim, n_classes, lr_model, DEVICE
        )
        
        pd.DataFrame(results).to_csv(csv_path, index=False)
        
    print("\nEvaluation Complete.")


if __name__ == "__main__":
    main()
