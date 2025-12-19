"""
Script for Part 3 of the demo experiment (x2).
Compares 3 model variations on CIFAR-10.

Variations:
1. Baseline: No splitting.
2. Random Split: Split a random neuron at epoch 25.
3. Greedy Split: Split the best neuron (per LR model) at epoch 25.

- Dataset: CIFAR-10
- Regimes: Hidden layer size 10
- Inits: 50 per variation
- Epochs: 50 (Split at 25)
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
    split_epoch = 25
    warmup_epochs = 10
    lr = 0.001
    temporal_windows = [8]
    starting_width = 30
    n_outputs = 10  # CIFAR-10
    n_features = 3072  # 32*32*3

    loss_fn = nn.CrossEntropyLoss()

    print(f"\nRunning variation: {variation_type}")

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

        # Train to split epoch
        trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            start_epoch=0,
            end_epoch=split_epoch,
            deterministic=True,  # Ensure reproducibility if seeded, but here we want different inits?
            # Trainer.fit seeds if deterministic=True.
            # Actually, we want different inits.
            # run_split_correlation_experiment doesn't explicit seed per init,
            # but the model init is random.
            # Note: Trainer.fit with deterministic=True sets seed based on epoch?
            # No, let's look at Trainer.fit. It doesn't seem to force seed unless we tell it.
            # But wait, run_split_correlation_experiment relies on different inits.
            # Let's assume standard initialization is random.
        )

        final_test_loss = 0.0

        if variation_type == "baseline":
            # Continue training to end
            res = trainer.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=split_epoch,
                end_epoch=epochs,
                deterministic=True,
            )
            final_test_loss = res["test_losses_epoch"][-1]

        elif variation_type in ["random", "greedy"]:
            wrapper: StatsWrapper = trainer.model

            target_neuron_idx = -1

            if variation_type == "random":
                target_neuron_idx = random.randint(0, starting_width - 1)

            elif variation_type == "greedy":
                if lr_model is None:
                    raise ValueError("LR model required for greedy split")

                # Collect stats for all neurons
                candidates = []
                for idx in range(starting_width):
                    feats = get_neuron_features(wrapper, idx, temporal_windows)
                    feats["neuron_idx"] = (
                        idx  # Ensure this is present if needed, though likely dropped
                    )
                    candidates.append(feats)

                df_candidates = pd.DataFrame(candidates)

                # Align columns with model features
                # Missing columns filled with 0 (or handled by imputer if it was part of training)
                # The pipeline handles imputation, but we need to ensure all expected columns are present.
                # If feature_columns is provided, we reindex.
                if feature_columns is not None:
                    # Add missing columns as NaN (to be imputed)
                    for col in feature_columns:
                        if col not in df_candidates.columns:
                            df_candidates[col] = np.nan
                    # Select only feature columns in correct order
                    X_candidates = df_candidates[feature_columns]
                else:
                    X_candidates = df_candidates

                # Predict probability of improvement
                probs = lr_model.predict_proba(X_candidates)[:, 1]
                target_neuron_idx = np.argmax(probs)

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

            # Continue training
            res = trainer.fit(
                train_loader=train_loader,
                test_loader=test_loader,
                start_epoch=split_epoch,
                end_epoch=epochs,
                deterministic=True,
            )
            final_test_loss = res["test_losses_epoch"][-1]

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
            # Fallback: Load the training data column names used
            train_csv = experiment_dir / "output" / "mnist" / "training_metrics.csv"
            df_train = pd.read_csv(train_csv)
            # Apply same exclusion logic as s2
            exclude_cols = [
                "regime_name",
                "starting_width",
                "init_id",
                "neuron_idx",
                "action_epoch",
                "delta_test_loss_at_h0",
            ]
            exclude_cols.extend([c for c in df_train.columns if "delta_test_loss" in c])
            feature_columns = [c for c in df_train.columns if c not in exclude_cols]
            print("Loaded feature columns from training CSV.")

    except FileNotFoundError:
        print("Model file not found. Skipping Greedy variation or failing.")
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

    # 3. Greedy
    if lr_model is not None:
        res_greedy = run_variation(
            "greedy", 50, train_loader, test_loader, DEVICE, lr_model, feature_columns
        )
        all_results.extend(res_greedy)
    else:
        print("Skipping Greedy variation due to missing model.")

    # Save results
    df_results = pd.DataFrame(all_results)
    output_csv = results_dir / "comparison_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
