# Adaptive Network (anet) Package

This package provides the core functionality for adaptive neural networks with neuron splitting capabilities.

## Structure

```
anet/
├── __init__.py              # Main package exports
├── actions.py               # Neuron splitting actions and strategies
├── aggregations.py          # Statistical aggregation functions
├── stats_wrapper.py         # Neural network statistics wrapper
├── trainer.py               # Training utilities
├── experiments.py           # High-level experiment utilities
└── layers/
    ├── __init__.py          # Layers subpackage
    └── widenable_layer.py   # WidenableLinear layer implementation
```

## Usage

Import the main components directly from the `anet` package:

```python
from anet import (
    # Neuron splitting
    split_neuron,
    ExactCopy,
    WithNoise,
    Half,
    OrthogonalDecomp,
    
    # Layers
    WidenableLinear,
    
    # Wrappers
    StatsWrapper,
    
    # Training
    Trainer,
    
    # Experiments
    run_split_correlation_experiment,
)
```

## Example

```python
import torch
import torch.nn as nn
from anet import WidenableLinear, StatsWrapper, split_neuron, ExactCopy, OrthogonalDecomp

# Create a model with widenable layers
model = StatsWrapper(
    nn.Sequential(
        WidenableLinear(784, 128),
        nn.GELU(),
        WidenableLinear(128, 10)
    ),
    buffer_size=32
)

# During training, split a neuron
split_neuron(
    network=model,
    input_layer_idx=0,
    output_layer_idx=2,
    neuron_idx=5,
    input_splitter=ExactCopy(),
    output_splitter=OrthogonalDecomp(),
)
```

## Components

### Actions (`actions.py`)
- `split_neuron()`: Split a neuron in a network
- Weight splitting strategies: `ExactCopy`, `WithNoise`, `Half`, `OrthogonalDecomp`

### Layers (`layers/widenable_layer.py`)
- `WidenableLinear`: Linear layer that can dynamically add neurons

### Stats Wrapper (`stats_wrapper.py`)
- `StatsWrapper`: Wraps a model to collect neuron statistics during training

### Trainer (`trainer.py`)
- `Trainer`: Training utilities with support for model snapshots and dynamic architecture changes

### Aggregations (`aggregations.py`)
- Statistical aggregation functions for time series analysis
- Over 40 different aggregation functions for analyzing neuron statistics

### Experiments (`experiments.py`)
- `run_split_correlation_experiment()`: High-level function for running controlled neuron splitting experiments
- Implements treatment/control design for studying the impact of splitting decisions

