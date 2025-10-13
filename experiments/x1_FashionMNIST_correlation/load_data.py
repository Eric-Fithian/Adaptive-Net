"""
Data loading utilities for FashionMNIST experiments.

This module provides functions to load and preprocess FashionMNIST dataset
in a format compatible with the anet experiment framework.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


def get_fashionmnist_loaders(
    batch_size: int = 128,
    test_size: float = 0.2,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    """
    Load FashionMNIST and return train/test DataLoaders with flattened images.
    
    Returns DataLoaders compatible with the correlation experiment framework,
    where data is in TensorDataset format with features and labels on device.
    
    Args:
        batch_size: Batch size for data loaders.
        test_size: Not used (FashionMNIST has predefined train/test split).
        device: Device to place tensors on ('cuda', 'mps', or 'cpu').
        seed: Random seed (not used, kept for API compatibility).
        
    Returns:
        Tuple of (train_loader, test_loader) with flattened 784-d images.
    """
    # Load FashionMNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Convert to flattened tensors
    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img.flatten())
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img.flatten())
        test_labels.append(label)
    
    # Stack into tensors
    X_train = torch.stack(train_images)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.stack(test_images)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    # Move to device
    X_train = X_train.to(device=device)
    y_train = y_train.to(device=device)
    X_test = X_test.to(device=device)
    y_test = y_test.to(device=device)
    
    # Create TensorDatasets
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, test_loader

