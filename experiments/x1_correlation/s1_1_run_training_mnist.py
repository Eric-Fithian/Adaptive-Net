"""
Training script for FashionMNIST correlation experiment.

This script trains different network architectures (regimes) on FashionMNIST:
- Each architecture has 1 hidden layer with different widths
- Split epochs are deterministically stratified across the specified range
- For each split epoch, multiple model initializations are trained
- Compares delta test loss between treatment (split) and control (no split)
- Saves results as CSV for downstream analysis
"""

import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from pathlib import Path

from anet import run_split_correlation_experiment
from anet.data_loaders import get_fashionmnist_loaders


# The main experiment function is now imported from anet.experiments
# See: anet/experiments.py for implementation details


if __name__ == "__main__":
    # Get experiment directory (where this script lives)
    experiment_dir = Path("experiments/x1_correlation")
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f'Device available: {DEVICE}')
    
    # Experiment hyperparameters
    BATCH_SIZE = 128
    WARMUP_EPOCHS = 10
    EPOCHS = 50
    ACTION_EPOCH_RANGE = (WARMUP_EPOCHS+1, EPOCHS-1)  # Inclusive range for split epochs
    N_ACTION_EPOCH_SLICES = 10  # Number of evenly spaced split epochs
    N_INITS_PER_SLICE = 20  # Number of model initializations per split epoch
    LR = 0.001
    N_NEURONS_PER_INIT = 10  # Number of neurons to split per initialization
    TEMPORAL_WINDOWS = [2, 4, 8, 16, 32]  # Temporal windows for statistics
    
    # Define 5 architecture regimes with different hidden layer widths
    # Format: 784-<hidden_width>-10
    REGIME_DICT = {
        "tiny": 10,
        "small": 20,
        "medium": 40,
        # "large": 80,
        # "xlarge": 160,
    }
    
    print(f"\n{'='*60}")
    print("FashionMNIST Correlation Experiment")
    print(f"{'='*60}")
    print(f"Dataset: FashionMNIST (784 -> hidden -> 10)")
    print(f"Regimes: {REGIME_DICT}")
    print(f"Epochs: {EPOCHS}")
    print(f"Action epoch range: {ACTION_EPOCH_RANGE} ({N_ACTION_EPOCH_SLICES} slices)")
    print(f"Initializations per slice: {N_INITS_PER_SLICE}")
    print(f"Neurons per initialization: {N_NEURONS_PER_INIT}")
    print(f"Temporal windows: {TEMPORAL_WINDOWS}")
    print(f"{'='*60}\n")
    
    # Load FashionMNIST dataset
    print("Loading FashionMNIST dataset...")
    train_loader, test_loader = get_fashionmnist_loaders(
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Input features: {train_loader.dataset.tensors[0].shape[1]}")
    print(f"Output classes: {train_loader.dataset.tensors[1].max().item() + 1}\n")
    
    # Create results directory with dataset-specific subfolder
    results_dir = experiment_dir / "output_local" / "mnist"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")
    
    # Run experiment
    print("Starting training...")
    stats = run_split_correlation_experiment(
        dataset_name="FashionMNIST",
        train_loader=train_loader,
        test_loader=test_loader,
        regime_dict=REGIME_DICT,
        warmup_epochs=WARMUP_EPOCHS,
        epochs=EPOCHS,
        action_epoch_range=ACTION_EPOCH_RANGE,
        n_action_epoch_slices=N_ACTION_EPOCH_SLICES,
        n_inits_per_slice=N_INITS_PER_SLICE,
        lr=LR,
        device=DEVICE,
        loss_fn=nn.CrossEntropyLoss(),  # Multi-class classification
        n_outputs=10,  # 10 FashionMNIST classes
        n_neurons_per_init=N_NEURONS_PER_INIT,
        temporal_windows=TEMPORAL_WINDOWS,
    )
    
    # Convert results to DataFrame and save
    print("\nProcessing results...")
    stats_df = pd.DataFrame(stats)
    print(f"Total samples collected: {len(stats_df)}")
    print(f"\nFirst few rows:")
    print(stats_df.head())
    
    # Save results
    output_csv = results_dir / 'correlation_experiment_results_FashionMNIST.csv'
    stats_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Save hyperparameters
    hyperparams_file = results_dir / 'hyperparameters.txt'
    with open(hyperparams_file, 'w') as f:
        f.write(f"DATASET: FashionMNIST\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"WARMUP_EPOCHS: {WARMUP_EPOCHS}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"ACTION_EPOCH_RANGE: {ACTION_EPOCH_RANGE}\n")
        f.write(f"N_ACTION_EPOCH_SLICES: {N_ACTION_EPOCH_SLICES}\n")
        f.write(f"N_INITS_PER_SLICE: {N_INITS_PER_SLICE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"N_NEURONS_PER_INIT: {N_NEURONS_PER_INIT}\n")
        f.write(f"TEMPORAL_WINDOWS: {TEMPORAL_WINDOWS}\n")
        f.write(f"REGIMES: {REGIME_DICT}\n")
        f.write(f"DEVICE: {DEVICE}\n")
    print(f"Hyperparameters saved to: {hyperparams_file}")
    
    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")

