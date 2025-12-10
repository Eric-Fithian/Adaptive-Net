"""
Baseline training script for FashionMNIST - no neuron splitting.

This script trains a single network architecture on FashionMNIST:
- One hidden layer with specified width
- Standard training with no adaptive modifications
- Plots training and validation loss curves
- Useful for establishing baseline performance before running splitting experiments
"""

import torch
import torch.nn as nn
from pathlib import Path

from anet import WidenableLinear, StatsWrapper, Trainer
from anet.data_loaders import get_fashionmnist_loaders


if __name__ == "__main__":
    # Configuration
    HIDDEN_WIDTH = 20           # Width of hidden layer (784 -> HIDDEN_WIDTH -> 10)
    BATCH_SIZE = 128
    EPOCHS = 100
    WARMUP_EPOCHS = 10
    LR = 0.001
    TRAIN_SUBSET_SIZE = None
    TEST_SUBSET_SIZE = None
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    print(f"{'='*60}")
    print("FashionMNIST Baseline Training")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Architecture: 784 -> {HIDDEN_WIDTH} -> 10")
    print(f"Epochs: {EPOCHS}")
    print(f"Warmup Epochs: {WARMUP_EPOCHS}")
    print(f"Learning Rate: {LR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading FashionMNIST dataset...")
    train_loader, test_loader = get_fashionmnist_loaders(
        batch_size=BATCH_SIZE,
        device=DEVICE,
        train_subset_size=TRAIN_SUBSET_SIZE,
        test_subset_size=TEST_SUBSET_SIZE,
    )
    n_features = train_loader.dataset.tensors[0].shape[1]
    n_classes = train_loader.dataset.tensors[1].max().item() + 1
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Input features: {n_features}")
    print(f"Output classes: {n_classes}\n")
    
    # Create model
    model = nn.Sequential(
        WidenableLinear(n_features, HIDDEN_WIDTH),
        nn.GELU(),
        WidenableLinear(HIDDEN_WIDTH, n_classes)
    )
    
    # Create trainer
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        device=DEVICE,
        lr=LR,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        use_cosine=True,
    )
    
    # Train
    print("Training...")
    result = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        start_epoch=0,
        end_epoch=EPOCHS,
        deterministic=True,
        progress_bar=True,  # Enable progress bar
    )
    
    # Get losses
    train_losses = trainer.get_train_losses()
    test_losses = trainer.get_test_losses()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Best Test Loss: {min(test_losses):.4f} (epoch {test_losses.index(min(test_losses)) + 1})")
    print(f"{'='*60}\n")
    
    # Create output directory
    experiment_dir_name = Path(__file__).parent.name
    output_dir = Path(f'output_local/{experiment_dir_name}/baseline/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save training curves using built-in method
    plot_path = output_dir / f'baseline_training_h{HIDDEN_WIDTH}.png'
    trainer.graph_train_test_losses(
        title=f'FashionMNIST Training (Architecture: 784-{HIDDEN_WIDTH}-10)',
        save_path=str(plot_path)
    )
    print(f"Plot saved to: {plot_path}")
    
    # Save losses to file
    losses_file = output_dir / f'baseline_losses_h{HIDDEN_WIDTH}.txt'
    with open(losses_file, 'w') as f:
        f.write(f"Architecture: 784-{HIDDEN_WIDTH}-10\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Warmup Epochs: {WARMUP_EPOCHS}\n")
        f.write(f"\nFinal Train Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Test Loss: {test_losses[-1]:.6f}\n")
        f.write(f"Best Test Loss: {min(test_losses):.6f} (epoch {test_losses.index(min(test_losses)) + 1})\n")
        f.write(f"\nEpoch-by-epoch losses:\n")
        f.write("Epoch,Train Loss,Test Loss\n")
        for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses), 1):
            f.write(f"{i},{train_loss:.6f},{test_loss:.6f}\n")
    
    print(f"Losses saved to: {losses_file}")
    print("\nDone!")

