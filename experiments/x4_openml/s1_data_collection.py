"""
Step 1: Meta-Data Collection for Deep Networks
Run correlation experiments with variable-depth networks (3-8 hidden layers).
Uses batch normalization between layers and random split layer selection.
Parallelized across multiple GPUs at the dataset level.
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from anet import split_neuron, ExactCopy, Half, WithNoise, StatsWrapper, Trainer
from experiments.x4_openml.utils import (
    get_dataset_splits, 
    get_openml_dataset, 
    build_deep_network,
    sample_network_depth,
    sample_split_layer,
)


CONFIG = {
    'batch_size': 128,
    'warmup_epochs': 5,
    'epochs': 30,
    'action_epoch_range': (6, 22),
    'n_action_epoch_slices': 5,
    'n_inits_per_slice': 3,
    'lr': 0.001,
    'n_neurons_per_init': 5,
    'temporal_windows': [2, 4, 8, 16, 32],
    'hidden_width': 50,
    'min_hidden_layers': 3,
    'max_hidden_layers': 8,
}


def run_deep_split_experiment(
    dataset_name: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    n_features: int,
    n_classes: int,
    config: Dict,
    device: str,
) -> List[Dict]:
    """
    Run split correlation experiment with deep networks.
    
    For each initialization:
    1. Sample random network depth (3-8 hidden layers)
    2. Build deep network with batch normalization
    3. Train to action_epoch
    4. Randomly select a splittable layer (skipping first 2 hidden layers)
    5. Split neurons and compare treatment vs control
    """
    output_splitter = WithNoise(Half(), sigma_ratio=0.01)
    
    min_action_epoch, max_action_epoch = config['action_epoch_range']
    n_action_epoch_slices = config['n_action_epoch_slices']
    
    if n_action_epoch_slices == 1:
        action_epochs = [min_action_epoch]
    else:
        action_epochs = np.linspace(min_action_epoch, max_action_epoch, n_action_epoch_slices)
        action_epochs = [int(round(e)) for e in action_epochs]
    
    all_stats = []
    temporal_windows = config['temporal_windows']
    max_buffer = max(temporal_windows) if temporal_windows else 32
    
    for action_epoch in action_epochs:
        for init_id in tqdm(
            range(config['n_inits_per_slice']),
            desc=f"{dataset_name} (split@{action_epoch})",
            leave=False
        ):
            n_hidden_layers = sample_network_depth(
                config['min_hidden_layers'], 
                config['max_hidden_layers']
            )
            
            input_layer_idx, output_layer_idx = sample_split_layer(n_hidden_layers)
            hidden_layer_being_split = input_layer_idx // 3
            
            model = build_deep_network(
                n_features=n_features,
                n_outputs=n_classes,
                hidden_width=config['hidden_width'],
                n_hidden_layers=n_hidden_layers,
                buffer_size=max_buffer,
            )
            
            trainer_pre = Trainer(
                model=deepcopy(model),
                loss_fn=nn.CrossEntropyLoss(),
                device=device,
                lr=config['lr'],
                epochs=config['epochs'],
                warmup_epochs=config['warmup_epochs'],
                use_cosine=True,
            )
            
            trainer_pre.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=0,
                end_epoch=action_epoch,
                deterministic=True,
            )
            pre_state = trainer_pre.snapshot()
            
            trainer_control = Trainer(
                model=deepcopy(model),
                loss_fn=nn.CrossEntropyLoss(),
                device=device,
                lr=config['lr'],
                epochs=config['epochs'],
                warmup_epochs=config['warmup_epochs'],
                use_cosine=True,
            )
            trainer_control.load_snapshot(pre_state)
            control_post_result = trainer_control.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=action_epoch,
                end_epoch=config['epochs'],
                deterministic=True,
            )
            control_test_losses = control_post_result['test_losses_epoch']
            
            current_width = config['hidden_width']
            
            for neuron_id in range(config['n_neurons_per_init']):
                neuron_idx = neuron_id % current_width
                
                trainer_treat = Trainer(
                    model=deepcopy(model),
                    loss_fn=nn.CrossEntropyLoss(),
                    device=device,
                    lr=config['lr'],
                    epochs=config['epochs'],
                    warmup_epochs=config['warmup_epochs'],
                    use_cosine=True,
                )
                trainer_treat.load_snapshot(pre_state)
                
                wrapper: StatsWrapper = trainer_treat.model
                stats = wrapper.get_neuron_stats(input_layer_idx, output_layer_idx, neuron_idx)
                for tw in temporal_windows:
                    stats_temporal = wrapper.get_neuron_stats_temporal(
                        input_layer_idx, output_layer_idx, neuron_idx, tw
                    )
                    stats = {**stats, **stats_temporal}
                
                split_neuron(
                    network=wrapper,
                    input_layer_idx=input_layer_idx,
                    output_layer_idx=output_layer_idx,
                    neuron_idx=neuron_idx,
                    input_splitter=ExactCopy(),
                    output_splitter=output_splitter,
                )
                
                existing = {id(p) for g in trainer_treat.optimizer.param_groups for p in g['params']}
                new_params = [p for p in wrapper.parameters() if id(p) not in existing]
                trainer_treat.add_new_params_follow_scheduler(new_params)
                
                treat_result = trainer_treat.fit(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    start_epoch=action_epoch,
                    end_epoch=config['epochs'],
                    deterministic=True,
                )
                treat_test_losses = treat_result['test_losses_epoch']
                
                row = stats.copy()
                row['n_hidden_layers'] = n_hidden_layers
                row['hidden_layer_split'] = hidden_layer_being_split
                row['input_layer_idx'] = input_layer_idx
                row['output_layer_idx'] = output_layer_idx
                row['hidden_width'] = config['hidden_width']
                row['init_id'] = init_id
                row['neuron_idx'] = neuron_idx
                row['action_epoch'] = action_epoch
                
                for h, (ctrl_loss, treat_loss) in enumerate(
                    zip(control_test_losses, treat_test_losses)
                ):
                    row[f'delta_test_loss_at_h{h}'] = treat_loss - ctrl_loss
                
                all_stats.append(row)
    
    return all_stats


def process_dataset(args: Tuple) -> Optional[str]:
    """Process a single dataset on specified GPU."""
    d_id, gpu_id, output_dir = args
    
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    dataset_name = f"openml_{d_id}"
    csv_path = output_dir / f"{dataset_name}.csv"
    
    if csv_path.exists():
        return f"Skipped {dataset_name} (already exists)"
    
    train_loader, test_loader, input_dim, n_classes = get_openml_dataset(d_id, device)
    
    if train_loader is None:
        return f"Failed to load {dataset_name}"
    
    print(f"[GPU {gpu_id}] Processing {dataset_name}: {input_dim} features, {n_classes} classes")
    
    stats = run_deep_split_experiment(
        dataset_name=dataset_name,
        train_loader=train_loader,
        test_loader=test_loader,
        n_features=input_dim,
        n_classes=n_classes,
        config=CONFIG,
        device=device,
    )
    
    df = pd.DataFrame(stats)
    df.to_csv(csv_path, index=False)
    
    return f"Completed {dataset_name} on GPU {gpu_id}"


def main():
    experiment_dir = Path("experiments/x4_openml")
    output_dir = experiment_dir / "output_local" / "data_collection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    if n_gpus == 0:
        print("No GPUs available. Exiting.")
        return
    
    train_ids, _ = get_dataset_splits()
    print(f"Meta-Train Datasets: {len(train_ids)}")
    print(f"Config: {CONFIG}")
    
    pending_datasets = []
    for d_id in train_ids:
        csv_path = output_dir / f"openml_{d_id}.csv"
        if not csv_path.exists():
            pending_datasets.append(d_id)
    
    print(f"Pending datasets: {len(pending_datasets)}")
    
    if not pending_datasets:
        print("All datasets already processed.")
        return
    
    tasks = []
    for i, d_id in enumerate(pending_datasets):
        gpu_id = i % n_gpus
        tasks.append((d_id, gpu_id, output_dir))
    
    if n_gpus == 1:
        for task in tqdm(tasks, desc="Processing datasets"):
            result = process_dataset(task)
            if result:
                print(result)
    else:
        mp.set_start_method('spawn', force=True)
        
        with mp.Pool(processes=n_gpus) as pool:
            results = list(tqdm(
                pool.imap(process_dataset, tasks),
                total=len(tasks),
                desc="Processing datasets"
            ))
            
            for result in results:
                if result:
                    print(result)
    
    print("\nData collection complete.")


if __name__ == "__main__":
    main()
