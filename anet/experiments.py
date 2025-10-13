"""
Experiment utilities for adaptive network research.

This module provides high-level experiment functions for studying neuron splitting
and other adaptive network behaviors.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict
from copy import deepcopy
from tqdm import tqdm

from anet import split_neuron, ExactCopy, OrthogonalDecomp, WidenableLinear, StatsWrapper, Trainer


def run_split_correlation_experiment(
    dataset_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    regime_dict: Dict[str, int],
    warmup_epochs: int,
    epochs: int,
    action_epoch: int,
    lr: float,
    device: str | torch.device,
    loss_fn: nn.Module | None = None,
    n_outputs: int | None = None,
    n_different_model_initializations: int = 50,
    n_neurons_per_init: int = 4,
    temporal_windows: List[int] = [2, 4, 8, 16, 32, 64, 128],
) -> List[Dict]:
    """
    Run correlation experiment for neuron splitting.
    
    This function implements a controlled experiment to study the correlation between
    neuron statistics at split time and the resulting impact on model performance.
    
    For each regime (architecture width) and initialization:
    1. Pretrain to action_epoch
    2. Split snapshot into control (no split) and treatment (split neuron)
    3. Continue training both to full epochs
    4. Record delta test loss at each horizon
    5. Capture neuron statistics at split time
    
    Args:
        dataset_name: Name of the dataset (for metadata).
        train_loader: Training data loader with TensorDataset.
        test_loader: Test data loader with TensorDataset.
        regime_dict: Dictionary mapping regime names to hidden layer widths.
        warmup_epochs: Number of warmup epochs for learning rate scheduler.
        epochs: Total number of training epochs.
        action_epoch: Epoch at which to split neuron in treatment group.
        lr: Learning rate.
        device: Device to train on ('cuda', 'mps', or 'cpu').
        loss_fn: Loss function to use. If None, infers from target data:
            - Multi-class (>2 unique values): CrossEntropyLoss
            - Binary (2 unique values): BCEWithLogitsLoss
            - Regression (continuous): MSELoss
        n_outputs: Number of output units. If None, infers from target data:
            - Classification: number of unique classes
            - Regression: output dimension (1 for scalar targets)
        n_different_model_initializations: Number of random initializations per regime.
        n_neurons_per_init: Number of neurons to split per initialization.
        temporal_windows: List of temporal window sizes for statistics.
        
    Returns:
        List of dictionaries containing statistics and performance metrics.
        Each dictionary includes:
        - Neuron statistics at split time (spatial and temporal)
        - regime_name, starting_width, init_id, neuron_idx
        - delta_test_loss_at_h{i} for each horizon i (epochs after action)
        
    Example:
        >>> # Multi-class classification (FashionMNIST)
        >>> train_loader, test_loader = get_data_loaders()
        >>> regime_dict = {"small": 20, "medium": 40, "large": 80}
        >>> results = run_split_correlation_experiment(
        ...     dataset_name="FashionMNIST",
        ...     train_loader=train_loader,
        ...     test_loader=test_loader,
        ...     regime_dict=regime_dict,
        ...     warmup_epochs=10,
        ...     epochs=100,
        ...     action_epoch=50,
        ...     lr=0.001,
        ...     device="cuda",
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     n_outputs=10,
        ...     n_different_model_initializations=20,
        ...     n_neurons_per_init=5,
        ...     temporal_windows=[2, 4, 8, 16, 32],
        ... )
        >>> 
        >>> # Regression example
        >>> results = run_split_correlation_experiment(
        ...     dataset_name="housing_prices",
        ...     train_loader=train_loader,
        ...     test_loader=test_loader,
        ...     regime_dict=regime_dict,
        ...     loss_fn=nn.MSELoss(),
        ...     n_outputs=1,
        ...     ...
        ... )
    """
    # Get input dimension from data
    n_features = train_loader.dataset.tensors[0].shape[1]
    y = train_loader.dataset.tensors[1]
    
    # Infer output dimension if not provided
    if n_outputs is None:
        if y.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            # Classification: count unique classes
            n_outputs = int(y.max().item() + 1)
        else:
            # Regression: get output dimension
            if len(y.shape) == 1:
                n_outputs = 1
            else:
                n_outputs = y.shape[1]
    
    # Infer loss function if not provided
    if loss_fn is None:
        if y.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            # Classification task
            n_unique = len(torch.unique(y))
            if n_unique > 2:
                loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = nn.BCEWithLogitsLoss()
        else:
            # Regression task
            loss_fn = nn.MSELoss()
    
    all_stats = []
    
    # Iterate over each regime (architecture width)
    for regime_name, starting_width in regime_dict.items():
        for init_id in tqdm(
            range(n_different_model_initializations), 
            desc=f"Training {dataset_name}->{regime_name} ({starting_width} hidden units)"
        ):
            # Create base model architecture
            model = StatsWrapper(
                nn.Sequential(
                    WidenableLinear(n_features, starting_width),
                    nn.GELU(),
                    WidenableLinear(starting_width, n_outputs)
                ),
                buffer_size=max(temporal_windows) if temporal_windows else None,
            )
            
            # Phase 1: Pretrain to action_epoch (deterministic)
            trainer_pre = Trainer(
                model=deepcopy(model),
                loss_fn=loss_fn,
                device=device,
                lr=lr,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                use_cosine=True,
            )
            
            pre_result = trainer_pre.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=0,
                end_epoch=action_epoch,
                deterministic=True,
            )
            pre_state = trainer_pre.snapshot()
            
            # Phase 2: Control group - continue without split
            trainer_control = Trainer(
                model=deepcopy(model),
                loss_fn=loss_fn,
                device=device,
                lr=lr,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                use_cosine=True,
            )
            trainer_control.load_snapshot(pre_state)
            control_post_result = trainer_control.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=action_epoch,
                end_epoch=epochs,
                deterministic=True,
            )
            control_test_losses_at_horizons = control_post_result['test_losses_epoch']
            
            # Phase 3: Treatment group - split neuron and continue
            raw_feature_rows: List[Dict] = []
            for neuron_id in range(n_neurons_per_init):
                neuron_idx = neuron_id % starting_width
                
                # Create treatment trainer from pre-action snapshot
                trainer_treat = Trainer(
                    model=deepcopy(model),
                    loss_fn=loss_fn,
                    device=device,
                    lr=lr,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    use_cosine=True,
                )
                trainer_treat.load_snapshot(pre_state)
                
                # Capture neuron statistics at action time
                wrapper: StatsWrapper = trainer_treat.model
                stats = wrapper.get_neuron_stats(0, 2, neuron_idx)
                for temporal_window in temporal_windows:
                    stats_temporal = wrapper.get_neuron_stats_temporal(0, 2, neuron_idx, temporal_window)
                    stats = {**stats, **stats_temporal}
                
                # Split the neuron
                split_neuron(
                    network=wrapper,
                    input_layer_idx=0,
                    output_layer_idx=2,
                    neuron_idx=neuron_idx,
                    input_splitter=ExactCopy(),
                    output_splitter=OrthogonalDecomp(),
                )
                
                # Add new parameters to optimizer
                existing = {id(p) for g in trainer_treat.optimizer.param_groups for p in g['params']}
                new_params = [p for p in wrapper.parameters() if id(p) not in existing]
                trainer_treat.add_new_params_follow_scheduler(new_params)
                
                # Continue training from action_epoch to end
                treat_result = trainer_treat.fit(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    start_epoch=action_epoch,
                    end_epoch=epochs,
                    deterministic=True,
                )
                treat_test_losses_at_horizons = treat_result['test_losses_epoch']
                
                # Record statistics and performance deltas
                row = stats.copy()
                row['regime_name'] = regime_name
                row['starting_width'] = starting_width
                row['init_id'] = init_id
                row['neuron_idx'] = neuron_idx
                
                # Calculate delta test loss at each horizon (epochs after action)
                for h, (control_loss, treat_loss) in enumerate(
                    zip(control_test_losses_at_horizons, treat_test_losses_at_horizons)
                ):
                    row[f'delta_test_loss_at_h{h}'] = treat_loss - control_loss
                
                raw_feature_rows.append(row)
            
            all_stats.extend(raw_feature_rows)
    
    return all_stats


__all__ = ['run_split_correlation_experiment']

