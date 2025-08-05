"""
Test script to analyze correlation between training step when neuron is split and final model performance.

This script:
1. Creates a small 2-layer model with no activation functions
2. For each experiment, randomly selects a training step to split a neuron
3. Trains the model with full gradient descent (no batching)
4. Evaluates on validation set to get performance metric
5. Analyzes correlation between split step and final performance
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Callable
import random
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
from actions import split_neuron, ExactCopy, WithNoise, Half, OrthogonalDecomp
from layers.widenable_layer import WidenableLinear
from my_datasets import get_all_datasets, get_dataset_by_name, _DATASET_TASK_TYPE
from stats_wrapper import StatsWrapper
from pathlib import Path
import json
from datetime import datetime


DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def train_model_with_split(
    model: StatsWrapper,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_layer_idx: int,
    output_layer_idx: int,
    neuron_idx: int,
    action_epoch: int,
    *,
    epochs: int,
    lr: float,
    device: str | torch.device,
    loss_fn: Callable,   # swap for CE in classification
    graph_loss: bool = True,
    use_cosine_scheduler: bool = True,
    warmup_epochs: int = 0,
) -> Tuple[Dict, List[float], List[float]]:
    """Fast, noisy training loop; returns best validation loss."""
    stats = {}
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Add cosine learning rate scheduler with warmup
    if use_cosine_scheduler:
        if warmup_epochs > 0:
            # Use PyTorch's built-in schedulers for warmup + cosine annealing
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            # Standard cosine annealing without warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = None
    
    best_test = float("inf")

    train_losses_epoch = []
    test_losses_epoch = []

    for epoch in range(epochs):
        model.train()
        acc_loss, acc_samples = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            acc_loss += loss.item() * yb.size(0)  # Fix #4: accumulate weighted loss
            acc_samples += yb.size(0)
        train_losses_epoch.append(acc_loss / acc_samples)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # --- Split Action ---
        if epoch+1 == action_epoch:
            # Debug: print layer info
            print(f"StatsWrapper layers: {[type(layer).__name__ for layer in model]}")
            
            stats = model.get_neuron_stats(input_layer_idx, output_layer_idx, neuron_idx)
            split_neuron(
                network=model,  # Use the model directly - it's now a Sequential
                input_layer_idx=input_layer_idx,
                output_layer_idx=output_layer_idx,
                neuron_idx=neuron_idx,
                input_splitter=ExactCopy(),
                output_splitter=OrthogonalDecomp(),
            )

            # Add new parameters to optimizer
            opt.param_groups.clear()
            opt.add_param_group({'params': model.parameters()})

        # --- Evaluation --- This should not be affected by the split action if the split is a function preserving split method (e.g. split_neuron)
        model.eval()
        acc_loss, acc_samples = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                acc_loss += loss.item() * yb.size(0)  # Fix #4: accumulate weighted loss
                acc_samples += yb.size(0)
        avg_test_loss = acc_loss / acc_samples
        test_losses_epoch.append(avg_test_loss)

    best_test = min(test_losses_epoch)
    best_test_idx = test_losses_epoch.index(best_test)

    if graph_loss:
        plt.plot(train_losses_epoch, label="Train loss")
        plt.plot(test_losses_epoch, label="Test loss")
        plt.xlabel(f"epoch ({epochs} epochs)")
        plt.ylabel("per-sample loss")
        plt.title(f"Losses for {model.__class__.__name__}")
        plt.plot(best_test_idx, best_test, "ro", label="Best test loss")
        plt.legend()
        plt.show()

    stats['best_test_loss'] = best_test

    return stats, train_losses_epoch, test_losses_epoch


def get_metrics_to_performances_for_split_action(
    dataset_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    starting_width: int,
    warmup_epochs: int,
    epochs: int,
    action_epoch: int,
    lr: float,
    n_different_model_initializations: int = 50,
) -> List[Dict]:
    """
    Run correlation experiment for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to use.
        starting_width: Starting width of the model.
        warmup_epochs: Number of epochs to warmup the model.
        epochs: Number of epochs to train the model.
        action_epoch: Epoch after which to split the neuron.
        lr: Learning rate for the model.
        n_different_model_initializations: Number of different model initializations to run.
    Returns:
        List of dictionaries with results
    """
    task_type = _DATASET_TASK_TYPE[dataset_name]
    n_features = train_loader.dataset.tensors[0].shape[1]
    
    # Fix #5: Determine correct output dimension for classification
    if task_type == "classification":
        # Get number of classes from the dataset
        y = train_loader.dataset.tensors[1]
        n_classes = int(y.max().item() + 1)
    else:
        n_classes = 1

    loss_fn = None
    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    all_stats = []

    for i in range(n_different_model_initializations):
        # Initialize model - StatsWrapper now inherits from nn.Sequential
        model = StatsWrapper(
            WidenableLinear(n_features, starting_width),
            nn.GELU(),
            WidenableLinear(starting_width, n_classes)
        )

        for neuron_idx in tqdm(range(starting_width), desc=f"Training model {i+1} of {n_different_model_initializations} for {dataset_name}"):
            model_copy = deepcopy(model)

            # Train model
            stats, _, _ = train_model_with_split(
                model=model_copy,
                train_loader=train_loader,
                test_loader=test_loader,
                input_layer_idx=0,  # WidenableLinear layer
                output_layer_idx=2,  # WidenableLinear layer (activation is at idx 1)
                neuron_idx=neuron_idx,
                action_epoch=action_epoch, 
                epochs=epochs, 
                lr=lr, 
                loss_fn=loss_fn, 
                device=DEVICE,
                graph_loss=False,  # Disable plotting for batch processing
                use_cosine_scheduler=True, 
                warmup_epochs=warmup_epochs,
            )
            all_stats.append(stats)

    return all_stats

if __name__ == "__main__":
    """Main function to run the correlation experiment."""
    BATCH_SIZE = 128
    TEST_SIZE = 0.2
    WARMUP_EPOCHS = 10
    EPOCHS = 100
    ACTION_EPOCH = 50
    LR = 0.0005
    N_DIFFERENT_MODEL_INITIALIZATIONS = 1
    
    # Get all available datasets
    all_datasets = get_all_datasets(
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
    )
    widths_json_path = Path("experiments/underperforming_models/20250802_231414/underperforming_model_widths.json")

    widths_json = json.load(widths_json_path.open())['Results']
    
    # Create results directory
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'experiments/correlation/{datetime_str}/')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_results = {}
    
    for dataset_name, (train_loader, test_loader) in get_all_datasets()[:-1]:
        starting_width = int(widths_json[dataset_name])  # Convert float to int
        stats = get_metrics_to_performances_for_split_action(
            dataset_name=dataset_name,
            train_loader=train_loader,
            test_loader=test_loader,
            starting_width=starting_width,
            warmup_epochs=WARMUP_EPOCHS,
            epochs=EPOCHS,
            action_epoch=ACTION_EPOCH,
            lr=LR,
            n_different_model_initializations=N_DIFFERENT_MODEL_INITIALIZATIONS,
        )
        experiment_results[dataset_name] = stats
    
    for dataset_name, stats in experiment_results.items():
        # Convert stats to dataframe
        stats_df = pd.DataFrame(stats)
        print(stats_df.head())
        stats_df.to_csv(results_dir / f'correlation_experiment_results_{dataset_name}.csv', index=False)
        starting_width = int(widths_json[dataset_name])

        # Save hyperparameters
        with open(results_dir / 'hyperparameters.txt', 'w') as f:
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"TEST_SIZE: {TEST_SIZE}\n")
            f.write(f"WARMUP_EPOCHS: {WARMUP_EPOCHS}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"ACTION_EPOCH: {ACTION_EPOCH}\n")
            f.write(f"LR: {LR}\n")
            f.write(f"N_DIFFERENT_MODEL_INITIALIZATIONS: {N_DIFFERENT_MODEL_INITIALIZATIONS}\n")
            f.write(f"STARTING_WIDTH: {starting_width}\n")
            f.write(f"DATASET_NAME: {dataset_name}\n")
            f.write(f"WIDTHS_JSON_PATH: {widths_json_path}\n")
        
        print(f"\nDetailed results saved to '{results_dir}/correlation_experiment_results_{dataset_name}.csv'")

        # Calculate correlation between split step and final performance
        if len(stats_df) > 1:  # Need at least 2 points for correlation
            stats_columns = [col for col in stats_df.columns if col != 'best_test_loss']
            for col in stats_columns:
                correlation = stats_df[col].corr(stats_df['best_test_loss'])
                print(f"Correlation between {col} and final performance for {dataset_name}: {correlation}")
        else:
            print(f"Not enough data points for correlation analysis for {dataset_name}")