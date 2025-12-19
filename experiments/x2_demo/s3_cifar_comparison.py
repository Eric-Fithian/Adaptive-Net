"""
Script for Part 3 of the demo experiment (x2).
Compares 4 model variations on CIFAR-10.

Variations:
1. Baseline: No splitting.
2. Random Split: Split a random neuron every 4 epochs starting from epoch 11.
3. Greedy Split: Split the best neuron (per LR model) every 4 epochs starting from epoch 11.
4. Anti-Greedy Split: Split the worst neuron (per LR model) every 4 epochs starting from epoch 11.

- Dataset: CIFAR-10
- Regimes: Hidden layer size 30
- Inits: 50 per variation
- Epochs: 50
- Split Schedule: Epochs [11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import random
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm

from anet import (
    split_neuron,
    ExactCopy,
    OrthogonalDecomp,
    WidenableLinear,
    StatsWrapper,
    Trainer,
)
from anet.data_loaders import get_cifar10_loaders


def get_neuron_features(wrapper, neuron_idx, temporal_windows=[8]):
    """Extract features for a specific neuron matching the training data format."""
    # Assuming standard architecture: Linear(0) -> Activation -> Linear(2)
    stats = wrapper.get_neuron_stats(0, 2, neuron_idx)
    for temporal_window in temporal_windows:
        stats_temporal = wrapper.get_neuron_stats_temporal(
            0, 2, neuron_idx, temporal_window
        )
        stats = {**stats, **stats_temporal}
    return stats


def run_variation(
    variation_type: str,
    n_inits: int,
    train_loader,
    test_loader,
    device,
    lr_model=None,
    feature_columns=None,
):
    results = []

    epochs = 50
    split_interval = 4
    start_split_epoch = 11
    # Create list of split epochs: 11, 15, 19, ... < 50
    split_epochs = list(range(start_split_epoch, epochs, split_interval))

    warmup_epochs = 10
    lr = 0.001
    temporal_windows = [8]
    starting_width = 30
    n_outputs = 10  # CIFAR-10
    n_features = 3072  # 32*32*3

    loss_fn = nn.CrossEntropyLoss()

    print(f"\nRunning variation: {variation_type}")
    print(f"Split epochs: {split_epochs}")

    for init_id in tqdm(range(n_inits), desc=f"{variation_type}"):
        # Base model
        model = StatsWrapper(
            nn.Sequential(
                WidenableLinear(n_features, starting_width),
                nn.GELU(),
                WidenableLinear(starting_width, n_outputs),
            ),
            buffer_size=max(temporal_windows),
        )

        # Trainer
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            device=device,
            lr=lr,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            use_cosine=True,
        )

        # Initial training segment before first split
        first_segment_end = split_epochs[0] if variation_type != "baseline" else epochs

        res = trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            start_epoch=0,
            end_epoch=first_segment_end,
            deterministic=True,
        )

        final_test_loss = (
            res["test_losses_epoch"][-1] if res["test_losses_epoch"] else float("inf")
        )

        if variation_type == "baseline":
            # Already trained to completion
            pass

        elif variation_type in ["random", "greedy", "anti-greedy"]:
            wrapper: StatsWrapper = trainer.model

            # Loop through each split epoch
            current_epoch = first_segment_end

            # Note: We iterate through split_epochs.
            # We just finished training up to `current_epoch` (which is split_epochs[0]).
            # So we perform the split NOW, then train to the next target.

            # We need to handle the loop carefully.
            # split_epochs = [11, 15, ...]
            # We just trained 0 -> 11. Now at epoch 11 (completed).
            # Perform split.
            # Train 11 -> 15.
            # ...

            # Let's iterate through the split points
            # We've already reached split_epochs[0]

            # Prepare the schedule of segments
            # [(11, 15), (15, 19), ..., (last_split, 50)]

            schedule = []
            for i in range(len(split_epochs)):
                start = split_epochs[i]
                end = split_epochs[i + 1] if i + 1 < len(split_epochs) else epochs
                schedule.append((start, end))

            # Loop through splits
            for i, (start_seg, end_seg) in enumerate(schedule):
                # We are currently at epoch `start_seg` (completed)
                # Perform split action based on stats captured during previous segment

                current_width = wrapper.layers[0].out_features
                target_neuron_idx = -1

                if variation_type == "random":
                    target_neuron_idx = random.randint(0, current_width - 1)

                elif variation_type in ["greedy", "anti-greedy"]:
                    if lr_model is None:
                        raise ValueError(
                            f"LR model required for {variation_type} split"
                        )

                    # Collect stats for all neurons
                    candidates = []
                    for idx in range(current_width):
                        feats = get_neuron_features(wrapper, idx, temporal_windows)
                        feats["neuron_idx"] = idx
                        candidates.append(feats)

                    df_candidates = pd.DataFrame(candidates)

                    # Align columns
                    if feature_columns is not None:
                        for col in feature_columns:
                            if col not in df_candidates.columns:
                                df_candidates[col] = np.nan
                        X_candidates = df_candidates[feature_columns]
                    else:
                        X_candidates = df_candidates

                    # Predict probability of improvement
                    # Fill NaNs if any remaining (imputer in pipeline should handle, but be safe)
                    # X_candidates = X_candidates.fillna(0)

                    try:
                        probs = lr_model.predict_proba(X_candidates)[:, 1]

                        if variation_type == "greedy":
                            target_neuron_idx = np.argmax(probs)
                        else:  # anti-greedy
                            target_neuron_idx = np.argmin(probs)

                    except Exception as e:
                        print(
                            f"Prediction failed at step {i}, fallback to random. Error: {e}"
                        )
                        target_neuron_idx = random.randint(0, current_width - 1)

                # Split the neuron
                split_neuron(
                    network=wrapper,
                    input_layer_idx=0,
                    output_layer_idx=2,
                    neuron_idx=target_neuron_idx,
                    input_splitter=ExactCopy(),
                    output_splitter=OrthogonalDecomp(),
                )

                # Update optimizer with new params
                existing = {
                    id(p) for g in trainer.optimizer.param_groups for p in g["params"]
                }
                new_params = [p for p in wrapper.parameters() if id(p) not in existing]
                trainer.add_new_params_follow_scheduler(new_params)

                # Train the next segment
                res = trainer.fit(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    start_epoch=start_seg,
                    end_epoch=end_seg,
                    deterministic=True,
                )
                final_test_loss = (
                    res["test_losses_epoch"][-1]
                    if res["test_losses_epoch"]
                    else final_test_loss
                )

        results.append(
            {
                "variation": variation_type,
                "init_id": init_id,
                "final_test_loss": final_test_loss,
            }
        )

    return results


if __name__ == "__main__":
    experiment_dir = Path("experiments/x2_demo")
    model_path = experiment_dir / "output" / "mnist" / "lr_model.joblib"
    results_dir = experiment_dir / "output" / "cifar10"
    results_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device available: {DEVICE}")

    # Load LR Model
    print(f"Loading LR model from {model_path}...")
    try:
        lr_model = joblib.load(model_path)
        # Extract feature names if available (sklearn < 1.0 doesn't have feature_names_in_)
        if hasattr(lr_model["classifier"], "feature_names_in_"):
            # This might be on the classifier step
            feature_columns = lr_model["classifier"].feature_names_in_
        elif hasattr(lr_model, "feature_names_in_"):
            feature_columns = lr_model.feature_names_in_
        else:
            raise ValueError("Feature columns not found in LR model.")
    except FileNotFoundError:
        print(
            "Model file not found. Skipping Greedy/Anti-Greedy variations or failing."
        )
        lr_model = None
        feature_columns = None

    # Load CIFAR-10
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=128,
        device=DEVICE,
    )

    all_results = []

    # 1. Baseline
    res_baseline = run_variation("baseline", 50, train_loader, test_loader, DEVICE)
    all_results.extend(res_baseline)

    # 2. Random
    res_random = run_variation("random", 50, train_loader, test_loader, DEVICE)
    all_results.extend(res_random)

    if lr_model is not None:
        # 3. Greedy
        res_greedy = run_variation(
            "greedy", 50, train_loader, test_loader, DEVICE, lr_model, feature_columns
        )
        all_results.extend(res_greedy)

        # 4. Anti-Greedy
        res_anti = run_variation(
            "anti-greedy",
            50,
            train_loader,
            test_loader,
            DEVICE,
            lr_model,
            feature_columns,
        )
        all_results.extend(res_anti)
    else:
        print("Skipping Greedy/Anti-Greedy variations due to missing model.")

    # Save results
    df_results = pd.DataFrame(all_results)
    output_csv = results_dir / "comparison_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
