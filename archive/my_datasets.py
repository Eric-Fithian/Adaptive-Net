"""
Capacity‑ladder dataset collection for neuron–splitting experiments
===================================================================

This module provides a set of progressively harder tabular datasets – both
synthetic and real – so you can measure how the *minimum‑performing* and
*under‑performing* model widths change with task complexity.

Key features
------------
* **Unified interface** – every dataset creator returns `(train_loader, val_loader)`.
* **Customisable loaders** – `batch_size`, `test_size`, and `seed` can be passed
  from the public helpers rather than being hard‑coded.
* **Registry pattern** – add/remove datasets by editing a single mapping.
* **Automatic standardisation** – all features are mean‑0 / std‑1 on the *train*
  split, then applied to *val*.
* **Consistent targets** – regression targets come back as `(N, 1)` floats;
  classification labels as `LongTensor` suitable for `nn.CrossEntropyLoss`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from sklearn.datasets import (
    fetch_openml,
    make_friedman1,
    make_moons,
    make_regression,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Helper for DataLoader creation                                               #
# --------------------------------------------------------------------------- #

def _to_loaders(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 128,
    test_size: float = 0.2,
    regression: bool = True,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[DataLoader, DataLoader]:
    """Split, standardise, tensorise, return train/val loaders."""

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    x_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(X_val, dtype=torch.float32)

    if regression:
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32).unsqueeze(1)
    else:
        y_train_t = torch.as_tensor(y_train.astype(np.int64))
        y_val_t = torch.as_tensor(y_val.astype(np.int64))

    # Move tensors to device before creating datasets
    x_train_t = x_train_t.to(device=device)
    y_train_t = y_train_t.to(device=device)
    x_val_t = x_val_t.to(device=device)
    y_val_t = y_val_t.to(device=device)
    
    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False),
    )


# --------------------------------------------------------------------------- #
# Individual dataset creators (internal)                                       #
# --------------------------------------------------------------------------- #

def _synthetic_linear(seed: int = 0, **loader_kwargs):
    X, y = make_regression(
        n_samples=5_000,
        n_features=40,
        n_informative=10,
        noise=0.1,
        random_state=seed,
    )
    return _to_loaders(X, y, regression=True, seed=seed, **loader_kwargs)


def _two_moons(seed: int = 0, **loader_kwargs):
    X, y = make_moons(n_samples=6_000, noise=0.3, random_state=seed)
    return _to_loaders(X, y, regression=False, seed=seed, **loader_kwargs)


def _friedman1(seed: int = 0, **loader_kwargs):
    X, y = make_friedman1(n_samples=8_000, noise=1.0, random_state=seed)
    return _to_loaders(X, y, regression=True, seed=seed, **loader_kwargs)


def _uci_concrete(seed: int = 0, **loader_kwargs):
    X, y = fetch_openml("Concrete_Compressive_Strength", as_frame=False, return_X_y=True)
    return _to_loaders(X, y.astype(np.float32), regression=True, seed=seed, **loader_kwargs)


def _uci_protein(seed: int = 0, **loader_kwargs):
    # CASP dataset is no longer available, using a synthetic alternative
    X, y = make_regression(
        n_samples=5_000,
        n_features=50,
        n_informative=20,
        noise=0.5,
        random_state=seed,
    )
    return _to_loaders(X, y, regression=True, seed=seed, **loader_kwargs)


def _uci_covertype(seed: int = 0, n_samples: int = 100_000, **loader_kwargs):
    """Subsample Covertype to keep runtime modest."""
    X, y = fetch_openml("Covertype", as_frame=False, return_X_y=True)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return _to_loaders(X[idx], y[idx], regression=False, seed=seed, **loader_kwargs)


def _synthetic_multimodal(seed: int = 0, n_samples: int = 10_000, d: int = 10, **loader_kwargs):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, size=(n_samples, d))

    def f_piecewise(x: float):
        if x < -1:
            return np.sin(3 * x)
        elif x < 1:
            return 0.5 * x**2
        return np.tanh(2 * x)

    y = np.array([f_piecewise(x) for x in X[:, 0]]) + 0.05 * rng.randn(n_samples)
    return _to_loaders(X, y, regression=True, seed=seed, **loader_kwargs)


# --------------------------------------------------------------------------- #
# Public registry & convenience wrappers                                       #
# --------------------------------------------------------------------------- #

_DATASET_REGISTRY: Dict[str, Callable[..., Tuple[DataLoader, DataLoader]]] = {
    "synthetic_linear": _synthetic_linear,
    "two_moons": _two_moons,
    "friedman1": _friedman1,
    "uci_concrete": _uci_concrete,
    "uci_protein": _uci_protein,
    # "uci_covertype": _uci_covertype,
    "synthetic_multimodal": _synthetic_multimodal,
}

_DATASET_TASK_TYPE: Dict[str, str] = {
    "synthetic_linear": "regression",
    "two_moons": "classification",
    "friedman1": "regression",
    "uci_concrete": "regression",
    "uci_protein": "regression",
    "uci_covertype": "classification",
    "synthetic_multimodal": "regression",
}


def list_datasets() -> List[str]:
    """Return the available dataset keys."""
    return list(_DATASET_REGISTRY)


def get_dataset_by_name(name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Fetch a single dataset with custom loader settings.

    Examples
    --------
    >>> train, val = get_dataset_by_name(
    ...     "uci_concrete", batch_size=64, test_size=0.25, seed=123
    ... )
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return _DATASET_REGISTRY[name](**kwargs)


def get_all_datasets(**loader_kwargs) -> List[Tuple[str, Tuple[DataLoader, DataLoader]]]:
    """Return **all** datasets, passing the same loader kwargs to each creator.

    Notes
    -----
    This is handy when you want to sweep through every dataset while keeping the
    *same* batch size or test split across the board:

    >>> for name, (tr, val) in get_all_datasets(batch_size=256, test_size=0.3):
    ...     ...
    """
    return [(k, creator(**loader_kwargs)) for k, creator in _DATASET_REGISTRY.items()]
