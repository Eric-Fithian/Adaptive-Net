"""
Step 1: Meta-Data Collection
Run the correlation experiment on the 'Meta-Train' subset of OpenML datasets.
Collects (metrics -> delta_loss) pairs for each split method.
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from anet import run_split_correlation_experiment, OrthogonalDecomp, Half, WithNoise
from experiments.x3_openml.utils import get_dataset_splits, get_openml_dataset


def get_output_splitter(split_method: str):
    """Return the appropriate output splitter for the given method."""
    if split_method == "half_noise":
        return WithNoise(Half(), sigma_ratio=0.01)
    elif split_method == "orthogonal":
        return OrthogonalDecomp()
    else:
        raise ValueError(f"Unknown split method: {split_method}")


if __name__ == "__main__":
    experiment_dir = Path("experiments/x3_openml")
    
    # Experiment Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Two split methods to compare
    SPLIT_METHODS = ["half_noise", "orthogonal"]
    
    # We use a modest config to get through many datasets
    CONFIG = {
        'batch_size': 128,
        'warmup_epochs': 5,
        'epochs': 30,
        # Split between epoch 6 and 22 to ensure 8 epochs horizon (30-22=8)
        'action_epoch_range': (6, 22), 
        'n_action_epoch_slices': 5,
        'n_inits_per_slice': 3,  # Keep low to save time
        'lr': 0.001,
        'n_neurons_per_init': 5, 
        'temporal_windows': [2, 4, 8, 16, 32],
        # We need a regime. We'll stick to a small/medium width for speed.
        'regime_dict': {'standard': 50}, 
    }
    
    train_ids, _ = get_dataset_splits()
    print(f"Meta-Train Datasets: {len(train_ids)}")
    print(f"Split Methods: {SPLIT_METHODS}")
    
    for split_method in SPLIT_METHODS:
        print(f"\n{'='*60}")
        print(f"Running data collection for split method: {split_method}")
        print(f"{'='*60}")
        
        output_dir = experiment_dir / "output_local" / f"data_collection_{split_method}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_splitter = get_output_splitter(split_method)
        
        for d_id in tqdm(train_ids, desc=f"Datasets ({split_method})"):
            dataset_name = f"openml_{d_id}"
            csv_path = output_dir / f"{dataset_name}.csv"
            
            if csv_path.exists():
                print(f"Skipping {dataset_name} (already exists)")
                continue
                
            train_loader, test_loader, input_dim, n_classes = get_openml_dataset(d_id, DEVICE)
            
            if train_loader is None:
                continue
                
            print(f"Running {dataset_name}: {input_dim} features, {n_classes} classes")
            
            try:
                stats = run_split_correlation_experiment(
                    dataset_name=dataset_name,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    regime_dict=CONFIG['regime_dict'],
                    warmup_epochs=CONFIG['warmup_epochs'],
                    epochs=CONFIG['epochs'],
                    action_epoch_range=CONFIG['action_epoch_range'],
                    n_action_epoch_slices=CONFIG['n_action_epoch_slices'],
                    n_inits_per_slice=CONFIG['n_inits_per_slice'],
                    lr=CONFIG['lr'],
                    device=DEVICE,
                    loss_fn=nn.CrossEntropyLoss(),
                    n_outputs=n_classes,
                    n_neurons_per_init=CONFIG['n_neurons_per_init'],
                    temporal_windows=CONFIG['temporal_windows'],
                    output_splitter=output_splitter,
                )
                
                df = pd.DataFrame(stats)
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\nData collection complete for all split methods.")
