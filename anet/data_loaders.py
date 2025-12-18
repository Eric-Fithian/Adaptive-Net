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
    train_subset_size: int = None,
    test_subset_size: int = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    """
    Load FashionMNIST and return train/test DataLoaders with flattened images.

    Returns DataLoaders compatible with the correlation experiment framework,
    where data is in TensorDataset format with features and labels on device.

    Args:
        batch_size: Batch size for data loaders.
        train_subset_size: Number of training samples to use (None for all).
        test_subset_size: Number of test samples to use (None for all).
        device: Device to place tensors on ('cuda', 'mps', or 'cpu').
        seed: Random seed (not used, kept for API compatibility).

    Returns:
        Tuple of (train_loader, test_loader) with flattened 784-d images.
    """
    # Load FashionMNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Convert to flattened tensors
    train_images = []
    train_labels = []
    for i, (img, label) in enumerate(train_dataset):
        if train_subset_size is not None and i >= train_subset_size:
            break
        train_images.append(img.flatten())
        train_labels.append(label)

    test_images = []
    test_labels = []
    for i, (img, label) in enumerate(test_dataset):
        if test_subset_size is not None and i >= test_subset_size:
            break
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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, test_loader


def get_mnist_loaders(
    batch_size: int = 128,
    train_subset_size: int = None,
    test_subset_size: int = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    """
    Load MNIST and return train/test DataLoaders with flattened images.

    Returns DataLoaders compatible with the correlation experiment framework,
    where data is in TensorDataset format with features and labels on device.

    Args:
        batch_size: Batch size for data loaders.
        train_subset_size: Number of training samples to use (None for all).
        test_subset_size: Number of test samples to use (None for all).
        device: Device to place tensors on ('cuda', 'mps', or 'cpu').
        seed: Random seed (not used, kept for API compatibility).

    Returns:
        Tuple of (train_loader, test_loader) with flattened 784-d images.
    """
    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Convert to flattened tensors
    train_images = []
    train_labels = []
    for i, (img, label) in enumerate(train_dataset):
        if train_subset_size is not None and i >= train_subset_size:
            break
        train_images.append(img.flatten())
        train_labels.append(label)

    test_images = []
    test_labels = []
    for i, (img, label) in enumerate(test_dataset):
        if test_subset_size is not None and i >= test_subset_size:
            break
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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, test_loader


def get_cifar10_loaders(
    batch_size: int = 128,
    train_subset_size: int = None,
    test_subset_size: int = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    """
    Load CIFAR-10 and return train/test DataLoaders with flattened images.

    Returns DataLoaders compatible with the correlation experiment framework,
    where data is in TensorDataset format with features and labels on device.

    Args:
        batch_size: Batch size for data loaders.
        train_subset_size: Number of training samples to use (None for all).
        test_subset_size: Number of test samples to use (None for all).
        device: Device to place tensors on ('cuda', 'mps', or 'cpu').
        seed: Random seed (not used, kept for API compatibility).

    Returns:
        Tuple of (train_loader, test_loader) with flattened 3072-d images.
    """
    # Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Convert to flattened tensors
    train_images = []
    train_labels = []
    for i, (img, label) in enumerate(train_dataset):
        if train_subset_size is not None and i >= train_subset_size:
            break
        train_images.append(img.flatten())
        train_labels.append(label)

    test_images = []
    test_labels = []
    for i, (img, label) in enumerate(test_dataset):
        if test_subset_size is not None and i >= test_subset_size:
            break
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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, test_loader
