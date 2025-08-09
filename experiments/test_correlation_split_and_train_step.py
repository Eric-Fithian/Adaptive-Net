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

def train_model_with_split(
    model: StatsWrapper,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_layer_idx: int,
    output_layer_idx: int,
    neuron_idx: int,
    action_epoch: int,
    take_action: bool = True,
    *,
    epochs: int,
    lr: float,
    device: str | torch.device,
    loss_fn: Callable,   # swap for CE in classification
    graph_loss: bool = True,
    use_cosine_scheduler: bool = True,
    warmup_epochs: int = 0,
    ) -> Tuple[Dict, List[float], List[float]]:
    """Fast, noisy training loop; returns best validation loss and stats at split time.

    If `take_action` is False or `action_epoch` <= 0, the model is trained without splitting (control).
    """
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
        if take_action and action_epoch > 0 and (epoch + 1) == action_epoch:
            # Capture pre-split stats (per-neuron and per-layer for normalization at action time)
            # Collect full layer snapshot so we can compute within-layer normalization correctly
            layer_snapshot = model.get_layer_neuron_stats(input_layer_idx, output_layer_idx)
            # Extract the selected neuron's stats
            stats = next((s for s in layer_snapshot if s.get('neuron_idx') == neuron_idx), {})

            # Compute within-layer percentiles for this neuron's features at action time
            import numpy as np
            keys = [k for k in stats.keys() if k not in ('neuron_idx',)]
            # Build arrays per key from layer_snapshot
            arrays = {}
            for k in keys:
                vals = [row[k] for row in layer_snapshot if row.get(k) is not None]
                arrays[k] = np.asarray(vals, dtype=float) if len(vals) > 0 else None
            for k in keys:
                v = stats.get(k, None)
                arr = arrays[k]
                if v is None or arr is None or arr.size == 0:
                    stats[f"{k}_pct"] = None
                else:
                    stats[f"{k}_pct"] = float((arr < v).mean())

            # Apply split
            split_neuron(
                network=model,
                input_layer_idx=input_layer_idx,
                output_layer_idx=output_layer_idx,
                neuron_idx=neuron_idx,
                input_splitter=ExactCopy(),
                output_splitter=OrthogonalDecomp(),
            )

            # IMPORTANT: Keep optimizer and scheduler state to continue LR schedule uninterrupted.
            # We extend the optimizer's parameter list in-place by adding new params group.
            # This preserves LR scheduler progress/state and avoids restarting warmup/cosine.
            existing_params = {id(p) for g in opt.param_groups for p in g['params']}
            new_params = [p for p in model.parameters() if id(p) not in existing_params]
            if len(new_params) > 0:
                # Use same LR as first param group
                group_lr = opt.param_groups[0].get('lr', lr)
                opt.add_param_group({'params': new_params, 'lr': group_lr})
                # Update scheduler's base_lrs if present so new group follows same schedule
                if scheduler is not None:
                    def _append_base_lrs_recursive(sched):
                        # Append to this scheduler if it tracks base_lrs
                        if hasattr(sched, 'base_lrs') and isinstance(sched.base_lrs, list):
                            sched.base_lrs.append(group_lr)
                        # Handle nested schedulers inside SequentialLR via public or private attribute
                        inner = getattr(sched, 'schedulers', None)
                        if inner is None:
                            inner = getattr(sched, '_schedulers', None)
                        if inner is not None:
                            for sch in inner:
                                _append_base_lrs_recursive(sch)
                    _append_base_lrs_recursive(scheduler)

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
    device: str | torch.device,
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

    # Helper: deterministic loaders (no shuffle) to ensure identical pre-split order
    def make_deterministic_loader(dl: DataLoader) -> DataLoader:
        return DataLoader(dl.dataset, batch_size=dl.batch_size, shuffle=False)

    # Helper: one-time pretraining up to action epoch; returns state dicts and best pre-split test loss
    def pretrain_to_action(
        model_base: StatsWrapper,
        train_dl: DataLoader,
        test_dl: DataLoader,
        *,
        epochs: int,
        action_epoch: int,
        lr: float,
        loss_fn: Callable,
        device: str | torch.device,
        warmup_epochs: int,
    ):
        model_base = deepcopy(model_base).to(device)
        opt = torch.optim.Adam(model_base.parameters(), lr=lr)
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
            sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_pre = float("inf")
        det_train = make_deterministic_loader(train_dl)
        det_test = make_deterministic_loader(test_dl)
        for epoch in range(action_epoch):
            model_base.train()
            for xb, yb in det_train:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model_base(xb), yb)
                loss.backward()
                opt.step()
            if sched is not None:
                sched.step()
            # eval
            model_base.eval()
            acc_loss, acc_samples = 0.0, 0
            with torch.no_grad():
                for xb, yb in det_test:
                    xb, yb = xb.to(device), yb.to(device)
                    l = loss_fn(model_base(xb), yb)
                    acc_loss += l.item() * yb.size(0)
                    acc_samples += yb.size(0)
            best_pre = min(best_pre, acc_loss / acc_samples)

        return {
            'model_state': deepcopy(model_base.state_dict()),
            'opt_state': deepcopy(opt.state_dict()),
            'sched_state': deepcopy(sched.state_dict() if sched is not None else {}),
            'best_pre': best_pre,
        }

    # Helper: continue training from saved state for remaining epochs
    def continue_from_state(
        model_template: StatsWrapper,
        state: Dict,
        train_dl: DataLoader,
        test_dl: DataLoader,
        *,
        start_epoch: int,
        epochs: int,
        lr: float,
        loss_fn: Callable,
        device: str | torch.device,
        warmup_epochs: int,
    ) -> Tuple[StatsWrapper, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, float]:
        model = deepcopy(model_template).to(device)
        model.load_state_dict(state['model_state'])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        opt.load_state_dict(state['opt_state'])
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs)
            sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        if state['sched_state']:
            sched.load_state_dict(state['sched_state'])

        det_train = make_deterministic_loader(train_dl)
        det_test = make_deterministic_loader(test_dl)
        best_post = float("inf")
        # continue epochs start_epoch..epochs-1
        for epoch in range(start_epoch, epochs):
            model.train()
            for xb, yb in det_train:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            if sched is not None:
                sched.step()
            # eval
            model.eval()
            acc_loss, acc_samples = 0.0, 0
            with torch.no_grad():
                for xb, yb in det_test:
                    xb, yb = xb.to(device), yb.to(device)
                    l = loss_fn(model(xb), yb)
                    acc_loss += l.item() * yb.size(0)
                    acc_samples += yb.size(0)
            best_post = min(best_post, acc_loss / acc_samples)
        return model, opt, sched, best_post

    for i in range(n_different_model_initializations):
        # Initialize model - StatsWrapper now inherits from nn.Sequential
        model = StatsWrapper(
            nn.Sequential(
                WidenableLinear(n_features, starting_width),
                nn.GELU(),
                WidenableLinear(starting_width, n_classes)
            )
        )

        # First, pretrain to action epoch deterministically once; reuse exact state for all neurons
        pre_state = pretrain_to_action(
            model_base=model,
            train_dl=train_loader,
            test_dl=test_loader,
            epochs=epochs,
            action_epoch=action_epoch,
            lr=lr,
            loss_fn=loss_fn,
            device=device,
            warmup_epochs=warmup_epochs,
        )
        control_best = pre_state['best_pre']

        # Then, run treatment per neuron using the same initialization
        neuron_stats_this_init: List[Dict] = []
        raw_feature_rows: List[Dict] = []
        for neuron_idx in tqdm(range(starting_width), desc=f"Training model {i+1} of {n_different_model_initializations} for {dataset_name}"):
            # Load exact pre-split state, capture stats at action time, then split and continue deterministically
            model_at_action, opt_at_action, sched_at_action, _ = continue_from_state(
                model_template=model,
                state=pre_state,
                train_dl=train_loader,
                test_dl=test_loader,
                start_epoch=action_epoch,
                epochs=action_epoch,  # no steps taken here; we just need model at action time
                lr=lr,
                loss_fn=loss_fn,
                device=device,
                warmup_epochs=warmup_epochs,
            )
            # Snapshot layer, compute percentiles, then split in-place
            layer_snapshot = model_at_action.get_layer_neuron_stats(0, 2)
            stats = next((s for s in layer_snapshot if s.get('neuron_idx') == neuron_idx), {})
            import numpy as np
            keys = [k for k in stats.keys() if k not in ('neuron_idx',)]
            arrays = {}
            for k in keys:
                vals = [row[k] for row in layer_snapshot if row.get(k) is not None]
                arrays[k] = np.asarray(vals, dtype=float) if len(vals) > 0 else None
            for k in keys:
                v = stats.get(k, None)
                arr = arrays[k]
                if v is None or arr is None or arr.size == 0:
                    stats[f"{k}_pct"] = None
                else:
                    stats[f"{k}_pct"] = float((arr < v).mean())
            # Perform split
            split_neuron(
                network=model_at_action,
                input_layer_idx=0,
                output_layer_idx=2,
                neuron_idx=neuron_idx,
                input_splitter=ExactCopy(),
                output_splitter=OrthogonalDecomp(),
            )
            # Keep optimizer/scheduler state; add new params group
            existing_params = {id(p) for g in opt_at_action.param_groups for p in g['params']}
            new_params = [p for p in model_at_action.parameters() if id(p) not in existing_params]
            if len(new_params) > 0:
                group_lr = opt_at_action.param_groups[0].get('lr', lr)
                opt_at_action.add_param_group({'params': new_params, 'lr': group_lr})
                if sched_at_action is not None:
                    def _append_base_lrs_recursive(sched):
                        if hasattr(sched, 'base_lrs') and isinstance(sched.base_lrs, list):
                            sched.base_lrs.append(group_lr)
                        inner = getattr(sched, 'schedulers', None)
                        if inner is None:
                            inner = getattr(sched, '_schedulers', None)
                        if inner is not None:
                            for sch in inner:
                                _append_base_lrs_recursive(sch)
                    _append_base_lrs_recursive(sched_at_action)

            # Continue deterministically from action_epoch to end
            det_train = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False)
            det_test = DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, shuffle=False)
            best_post = float('inf')
            for epoch in range(action_epoch, epochs):
                model_at_action.train()
                for xb, yb in det_train:
                    xb, yb = xb.to(device), yb.to(device)
                    opt_at_action.zero_grad()
                    loss = loss_fn(model_at_action(xb), yb)
                    loss.backward()
                    opt_at_action.step()
                if sched_at_action is not None:
                    sched_at_action.step()
                model_at_action.eval()
                acc_loss, acc_samples = 0.0, 0
                with torch.no_grad():
                    for xb, yb in det_test:
                        xb, yb = xb.to(device), yb.to(device)
                        l = loss_fn(model_at_action(xb), yb)
                        acc_loss += l.item() * yb.size(0)
                        acc_samples += yb.size(0)
                best_post = min(best_post, acc_loss / acc_samples)
            stats['best_test_loss'] = best_post
            # Store features captured at action time (already includes within-layer z/pct)
            row = {k: v for k, v in stats.items()}
            row['neuron_idx'] = neuron_idx
            row['control_best_test_loss'] = control_best
            row['delta_best_test_loss'] = stats['best_test_loss'] - control_best
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
    N_DIFFERENT_MODEL_INITIALIZATIONS = 25
    
    # Get all available datasets
    all_datasets = get_all_datasets(
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
        device=DEVICE,
    )
    widths_json_path = Path("experiments/underperforming_models/20250802_231414/underperforming_model_widths.json")

    widths_json = json.load(widths_json_path.open())['Results']
    
    # Create results directory
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'experiments/correlation/{datetime_str}/')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_results = {}
    
    for dataset_name, (train_loader, test_loader) in all_datasets:
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
            device=DEVICE,
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

        # Calculate correlation between normalized metrics and delta best test loss
        if len(stats_df) > 1:  # Need at least 2 points for correlation
            # Prefer normalized metrics (_pct); fall back to raw if needed
            candidate_cols = [c for c in stats_df.columns if c.endswith('_pct')]
            if not candidate_cols:
                candidate_cols = [col for col in stats_df.columns if col not in ('best_test_loss','delta_best_test_loss','control_best_test_loss','neuron_idx')]
            table_rows = []
            for col in candidate_cols:
                # exclude NA rows
                pair_df = stats_df[[col, 'delta_best_test_loss']]
                pair_df_clean = pair_df.dropna()
                # Require at least some variability
                if len(pair_df_clean) < 5 or pair_df_clean[col].nunique() < 3:
                    continue
                pearson_corr, pearson_p_val = pearsonr(pair_df_clean[col], pair_df_clean['delta_best_test_loss'])
                table_rows.append({
                    "Metric": col,
                    "corr": pearson_corr,
                    "p_value": pearson_p_val,
                })
            corr_df = pd.DataFrame(table_rows)
            print(f"\nCorrelation table for {dataset_name} (target = delta_best_test_loss):")
            print(corr_df.to_string(index=False, float_format="%.4f"))
        else:
            print(f"Not enough data points for correlation analysis for {dataset_name}")