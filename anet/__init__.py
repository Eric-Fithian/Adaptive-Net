"""Adaptive Network (anet) package."""

# Core action and splitting functionality
from .actions import (
    split_neuron,
    WeightSplitStrategy,
    ExactCopy,
    WithNoise,
    Half,
    OrthogonalDecomp,
)

# Layers
from .layers import WidenableLinear

# Stats wrapper
from .stats_wrapper import StatsWrapper

# Trainer
from .trainer import Trainer

# Experiments
from .experiments import run_split_correlation_experiment

__all__ = [
    # Actions
    'split_neuron',
    'WeightSplitStrategy',
    'ExactCopy',
    'WithNoise',
    'Half',
    'OrthogonalDecomp',
    
    # Layers
    'WidenableLinear',
    
    # Stats
    'StatsWrapper',
    
    # Training
    'Trainer',
    
    # Experiments
    'run_split_correlation_experiment',
]

