"""
Script for Part 1 of the demo experiment (x2).
Creates a dataset of metrics -> delta test losses for MNIST.

- Dataset: MNIST
- Hidden layer size: 10
- Train 40 different model inits.
- Split epoch: 25
- Horizon: 8
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

from anet import run_split_correlation_experiment
from anet.data_loaders import get_mnist_loaders

if __name__ == "__main__":
    # Get experiment directory
    experiment_dir = Path("experiments/x2_demo")

    # Device configuration
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device available: {DEVICE}")

    # Experiment hyperparameters
    BATCH_SIZE = 128
    WARMUP_EPOCHS = 10
    EPOCHS = 50  # Total epochs to run
    ACTION_EPOCH_RANGE = (25, 25)  # Fixed split epoch at 25
    N_ACTION_EPOCH_SLICES = 1
    N_INITS_PER_SLICE = 40
    LR = 0.001
    N_NEURONS_PER_INIT = 10  # All neurons in the layer
    TEMPORAL_WINDOWS = [8]  # Fixed horizon

    REGIME_DICT = {
        "demo": 10,
    }

    print(f"\n{'='*60}")
    print("MNIST Dataset Creation Experiment (Part 1)")
    print(f"{'='*60}")
    print(f"Dataset: MNIST")
    print(f"Regimes: {REGIME_DICT}")
    print(f"Action epoch range: {ACTION_EPOCH_RANGE}")
    print(f"Inits: {N_INITS_PER_SLICE}")
    print(f"{'='*60}\n")

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    # Create results directory
    results_dir = experiment_dir / "output" / "mnist"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    print("Starting training...")
    stats = run_split_correlation_experiment(
        dataset_name="MNIST",
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
        loss_fn=nn.CrossEntropyLoss(),
        n_outputs=10,
        n_neurons_per_init=N_NEURONS_PER_INIT,
        temporal_windows=TEMPORAL_WINDOWS,
    )

    # Convert results to DataFrame and save
    print("\nProcessing results...")
    stats_df = pd.DataFrame(stats)

    # Save results
    output_csv = results_dir / "training_metrics.csv"
    stats_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
