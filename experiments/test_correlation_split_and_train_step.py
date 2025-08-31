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
from trainer import Trainer
from pathlib import Path
import json
from datetime import datetime

# Removed legacy train loop in favor of Trainer class


def get_metrics_to_performances_for_split_action(
    dataset_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    regime_dict: Dict[str, int],
    warmup_epochs: int,
    epochs: int,
    action_epoch: int,
    lr: float,
    device: str | torch.device,
    n_different_model_initializations: int = 50,
    n_neurons_per_init: int = 4,
    temporal_windows: List[int] = [2, 4, 8, 16, 32, 64, 128],
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

    # Use Trainer for pretraining and treatment phases
    for regime_name, starting_width in regime_dict.items():
        for i in tqdm(range(n_different_model_initializations), desc=f"Training models for {dataset_name}->{regime_name} with {n_neurons_per_init} neurons/init"):
            # Initialize model - StatsWrapper now inherits from nn.Sequential
            model = StatsWrapper(
                nn.Sequential(
                    WidenableLinear(n_features, starting_width),
                    nn.GELU(),
                    WidenableLinear(starting_width, n_classes)
                ),
                buffer_size=max(temporal_windows) if temporal_windows else None,
            )

            # First, pretrain to action epoch deterministically once; reuse exact state for all neurons
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

            # Control: continue training without any split from the same snapshot to full epochs
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

            # Then, run treatment per neuron using the same initialization
            neuron_stats_this_init: List[Dict] = []
            raw_feature_rows: List[Dict] = []
            for i in range(n_neurons_per_init):
                # Load exact pre-split state, capture stats at action time, then split and continue deterministically
                # Build a treatment trainer from the same template and load pre-action snapshot
                neuron_idx = i%starting_width
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

                # Capture enriched stats at the action boundary and perform split immediately
                wrapper: StatsWrapper = trainer_treat.model
                stats = wrapper.get_neuron_stats(0, 2, neuron_idx)
                for temporal_window in temporal_windows:
                    stats_temporal = wrapper.get_neuron_stats_temporal(0, 2, neuron_idx, temporal_window)
                    stats = {**stats, **stats_temporal}
                split_neuron(
                    network=wrapper,
                    input_layer_idx=0,
                    output_layer_idx=2,
                    neuron_idx=neuron_idx,
                    input_splitter=ExactCopy(),
                    output_splitter=OrthogonalDecomp(),
                )
                existing = {id(p) for g in trainer_treat.optimizer.param_groups for p in g['params']}
                new_params = [p for p in wrapper.parameters() if id(p) not in existing]
                trainer_treat.add_new_params_follow_scheduler(new_params)

                # Continue deterministically from action_epoch to end
                treat_result = trainer_treat.fit(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    start_epoch=action_epoch,
                    end_epoch=epochs,
                    deterministic=True,
                )
                treat_test_losses_at_horizons = treat_result['test_losses_epoch'] 
                # Take the best across full run (pre + post)
                # Store features captured at action time (already includes within-layer z/pct)
                row = stats.copy()
                row['regime_name'] = regime_name
                row['starting_width'] = starting_width
                row[f'init_id'] = i
                row['neuron_idx'] = neuron_idx
                for i, (control_test_loss, treat_test_loss) in enumerate(zip(control_test_losses_at_horizons, treat_test_losses_at_horizons)):
                    row[f'delta_test_loss_at_h{i}'] = treat_test_loss - control_test_loss
                raw_feature_rows.append(row)
                neuron_stats_this_init.append(stats)
            # Already normalized at action time; append as-is
            all_stats.extend(raw_feature_rows)

    return all_stats

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Device available: {DEVICE}')

    """Main function to run the correlation experiment."""
    BATCH_SIZE = 128
    TEST_SIZE = 0.2
    WARMUP_EPOCHS = 10
    EPOCHS = 100
    ACTION_EPOCH = 50
    LR = 0.005
    N_DIFFERENT_MODEL_INITIALIZATIONS = 1
    N_NEURONS_PER_INIT = 1
    TEMPORAL_WINDOWS = [8]

    
    # Get all available datasets
    all_datasets = get_all_datasets(
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
        device=DEVICE,
    )
    # widths_json_path = Path("experiments/underperforming_models/20250802_231414/underperforming_model_widths.json")
    # widths_json = json.load(widths_json_path.open())['Results']
    regimes_json_path = Path("experiments/underperforming_models/20250802_231414/regimes.json")
    regimes_json = json.load(regimes_json_path.open())
    # Create results directory
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'experiments/correlation/{datetime_str}/')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_results = {}

    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    for dataset_name, (train_loader, test_loader) in all_datasets[:1]:
        regime_dict = regimes_json[dataset_name]
        stats = get_metrics_to_performances_for_split_action(
            dataset_name=dataset_name,
            train_loader=train_loader,
            test_loader=test_loader,
            regime_dict=regime_dict,
            warmup_epochs=WARMUP_EPOCHS,
            epochs=EPOCHS,
            action_epoch=ACTION_EPOCH,
            lr=LR,
            device=DEVICE,
            n_different_model_initializations=N_DIFFERENT_MODEL_INITIALIZATIONS,
            n_neurons_per_init=N_NEURONS_PER_INIT,
            temporal_windows=TEMPORAL_WINDOWS,
        )
        experiment_results[dataset_name] = stats
    
        # Convert stats to dataframe
        stats_df = pd.DataFrame(stats)
        print(stats_df.head())
        stats_df.to_csv(results_dir / f'correlation_experiment_results_{dataset_name}.csv', index=False)

        # Save hyperparameters
        with open(results_dir / 'hyperparameters.txt', 'w') as f:
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"TEST_SIZE: {TEST_SIZE}\n")
            f.write(f"WARMUP_EPOCHS: {WARMUP_EPOCHS}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"ACTION_EPOCH: {ACTION_EPOCH}\n")
            f.write(f"LR: {LR}\n")
            f.write(f"N_DIFFERENT_MODEL_INITIALIZATIONS: {N_DIFFERENT_MODEL_INITIALIZATIONS}\n")
            f.write(f"DATASET_NAME: {dataset_name}\n")
            f.write(f"REGIMES_JSON_PATH: {regimes_json_path}\n")
        
        print(f"\nDetailed results saved to '{results_dir}/correlation_experiment_results_{dataset_name}.csv'")

    profiler.stop()
    profiler.write_html(results_dir / 'profile.html')


        # Calculate correlation for both raw spatial metrics and their within-layer percentiles (should total 26)
        # if len(stats_df) > 1:  # Need at least 2 points for correlation
        #     # Use both raw and _pct metrics; exclude non-metric columns
        #     candidate_cols = [
        #         col for col in stats_df.columns
        #         if col not in ('best_test_loss','delta_best_test_loss','control_best_test_loss','neuron_idx')
        #     ]
        #     table_rows = []
        #     for col in candidate_cols:
        #         # exclude NA rows
        #         pair_df = stats_df[[col, 'delta_best_test_loss']]
        #         pair_df_clean = pair_df.dropna()
        #         # Require at least some variability
        #         if len(pair_df_clean) < 5 or pair_df_clean[col].nunique() < 3:
        #             continue
        #         pearson_corr, pearson_p_val = pearsonr(pair_df_clean[col], pair_df_clean['delta_best_test_loss'])
        #         table_rows.append({
        #             "Metric": col,
        #             "corr": pearson_corr,
        #             "p_value": pearson_p_val,
        #         })
        #     corr_df = pd.DataFrame(table_rows)
        #     print(f"\nCorrelation table for {dataset_name} (target = delta_best_test_loss):")
        #     print(corr_df.to_string(index=False, float_format="%.4f"))
        # else:
        #     print(f"Not enough data points for correlation analysis for {dataset_name}")