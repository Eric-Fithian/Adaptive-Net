# adaptive_widen_named.py
from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn

from anet.layers.widenable_layer import WidenableLinear


# --------------------------------------------------------------------------- #
# 1.  Strategy interface for *any* 1-D weight or bias vector
# --------------------------------------------------------------------------- #
class WeightSplitStrategy(ABC):
    """
    Split a 1-D tensor v into (v_kept, v_added), both same shape as v.
    """
    @abstractmethod
    def split(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


# --------------------------------------------------------------------------- #
# 2.  Built-in strategies
# --------------------------------------------------------------------------- #
class ExactCopy(WeightSplitStrategy):
    """v_added = copy of v_kept = v."""
    def split(self, v):
        return v, v.clone()


class Half(WeightSplitStrategy):
    """Divide outgoing fan-out evenly: both halves sum back to v."""
    def split(self, v):
        half = v * 0.5
        return half, half.clone()


class OrthogonalDecomp(WeightSplitStrategy):
    """Split into orthogonal halves of equal L2-norm that sum to v."""
    def split(self, v):
        w, r = v.clone(), torch.randn_like(v)
        # Guard: if ||w|| ~ 0, fall back to equal split (function-preserving)
        w_norm_sq = float((w @ w).item()) if w.numel() > 0 else 0.0
        eps = 1e-12
        if w_norm_sq <= eps:
            half = 0.5 * w
            return half, half.clone()
        # Make r orthogonal to w
        r = r - (float((w @ r).item()) / w_norm_sq) * w
        # Guard: if r collapsed to 0, sample another random vector, fall back if needed
        r_norm = float(r.norm().item())
        if r_norm <= eps:
            # try another random draw once
            r = torch.randn_like(w)
            r = r - (float((w @ r).item()) / w_norm_sq) * w
            r_norm = float(r.norm().item())
            if r_norm <= eps:
                half = 0.5 * w
                return half, half.clone()
        # Scale r to match ||w||
        r = r / r_norm * float(w.norm().item())
        v_kept = 0.5 * (w + r)
        v_added = 0.5 * (w - r)
        # Sanity: ensure sum equals original within tolerance
        if not torch.allclose(v_kept + v_added, w, atol=1e-8, rtol=1e-6):
            import warnings
            warnings.warn("OrthogonalDecomp sum check failed; falling back to half split.")
            half = 0.5 * w
            return half, half.clone()
        return v_kept, v_added


# Helpful decorator to add noise to any WeightSplitStrategy
class WithNoise(WeightSplitStrategy):
    """Wrapper that adds noise to any WeightSplitStrategy."""
    def __init__(self, strategy: WeightSplitStrategy, sigma_ratio: float = 0.01):
        self.strategy = strategy
        self.sigma_ratio = sigma_ratio
        
    def split(self, v):
        v_kept, v_added = self.strategy.split(v)
        σ = v.abs().mean() * self.sigma_ratio
        noise = torch.randn_like(v_added) * σ
        return v_kept, v_added + noise

# --------------------------------------------------------------------------- #
# 3.  Widening helper that takes separate strategies for input vs output
# --------------------------------------------------------------------------- #
def split_neuron(
    *,
    network: nn.Module,
    input_layer_idx: int,
    output_layer_idx: int,
    neuron_idx: int,
    input_splitter: WeightSplitStrategy,
    output_splitter: WeightSplitStrategy,
) -> None:
    """
    Duplicate the hidden neuron `neuron_idx` between two adjacent Linear layers:
      - `network[input_layer_idx]` provides the incoming weights & bias
      - `network[output_layer_idx]` provides the outgoing weights

    `input_splitter.split(...)` controls how you split the incoming vector/bias.
    `output_splitter.split(...)` controls how you split the fan-out column.

    WARNING: `network` is modified in place.

    Args:
        network: The network to split the neuron in.
        input_layer_idx: The index of the input layer.
        output_layer_idx: The index of the output layer.
        neuron_idx: The index of the neuron to split.
        input_splitter: The strategy to use for splitting the incoming weights and bias.
        output_splitter: The strategy to use for splitting the outgoing weights.

    Returns:
        None
    """
    # --- validate ---
    if not (
        0 <= input_layer_idx < len(network)
        and 0 < output_layer_idx < len(network)
    ):
        raise ValueError("Input and output layers must be valid indicies.")
    if not (
        0 <= neuron_idx < network[input_layer_idx].out_features
        and 0 <= neuron_idx < network[output_layer_idx].in_features
    ):
        raise ValueError("Selected unit is not a mutable hidden neuron.")
    if not (
        isinstance(network[input_layer_idx], WidenableLinear)
        and isinstance(network[output_layer_idx], WidenableLinear)
    ):
        raise ValueError("Input and output layers must be WidenableLinear.")

    # Check that the dimensions match to indirectly check that the layers are adjacent. Okay if there are activation functions in between.
    if network[input_layer_idx].out_features != network[output_layer_idx].in_features:
        raise ValueError("Input and output layers must have the same number of features. Check that the layers are adjacent.")

    in_l: WidenableLinear = network[input_layer_idx]
    out_l: WidenableLinear = network[output_layer_idx]

    with torch.no_grad():
        # --- split incoming weights & bias ---
        w_old = in_l.weight.data[neuron_idx]
        b_old = in_l.bias.data[neuron_idx]
        w_kept, w_added = input_splitter.split(w_old)
        b_kept, b_added = input_splitter.split(b_old)
        in_l.set_output_feature(neuron_idx, w_kept, b_kept)
        in_l.add_output_feature(w_added, b_added)

        # --- split outgoing weights ---
        v_old = out_l.weight.data[:, neuron_idx]
        v_kept, v_added = output_splitter.split(v_old)
        out_l.set_input_feature(neuron_idx, v_kept)
        out_l.add_input_feature(v_added)


def prune_neuron(
    *,
    network: nn.Module,
    input_layer_idx: int,
    output_layer_idx: int,
    neuron_idx: int,
) -> None:
    """
    Remove the hidden neuron `neuron_idx` between two adjacent Linear layers:
      - `network[input_layer_idx]` provides the incoming weights & bias
      - `network[output_layer_idx]` provides the outgoing weights

    WARNING: `network` is modified in place.

    Args:
        network: The network to prune the neuron from.
        input_layer_idx: The index of the input layer.
        output_layer_idx: The index of the output layer.
        neuron_idx: The index of the neuron to prune.

    Returns:
        None
    """
    # --- validate ---
    if not (
        0 <= input_layer_idx < len(network)
        and 0 < output_layer_idx < len(network)
    ):
        raise ValueError("Input and output layers must be valid indicies.")
    if not (
        0 <= neuron_idx < network[input_layer_idx].out_features
        and 0 <= neuron_idx < network[output_layer_idx].in_features
    ):
        raise ValueError("Selected unit is not a mutable hidden neuron.")
    if not (
        isinstance(network[input_layer_idx], WidenableLinear)
        and isinstance(network[output_layer_idx], WidenableLinear)
    ):
        raise ValueError("Input and output layers must be WidenableLinear.")

    # Check that the dimensions match to indirectly check that the layers are adjacent. Okay if there are activation functions in between.
    if network[input_layer_idx].out_features != network[output_layer_idx].in_features:
        raise ValueError("Input and output layers must have the same number of features. Check that the layers are adjacent.")

    in_l: WidenableLinear = network[input_layer_idx]
    out_l: WidenableLinear = network[output_layer_idx]

    with torch.no_grad():
        # --- remove neuron from input layer (outgoing weights & bias) ---
        # Keep all neurons except the one at neuron_idx
        mask = torch.ones(in_l.out_features, dtype=torch.bool)
        mask[neuron_idx] = False
        
        new_weight = in_l.weight.data[mask]
        new_bias = in_l.bias.data[mask]
        
        # Update input layer dimensions
        in_l.out_features = new_weight.shape[0]
        in_l.weight = nn.Parameter(new_weight)
        in_l.bias = nn.Parameter(new_bias)
        
        # --- remove neuron from output layer (incoming weights) ---
        # Keep all input features except the one at neuron_idx
        new_weight = out_l.weight.data[:, mask]
        
        # Update output layer dimensions
        out_l.in_features = new_weight.shape[1]
        out_l.weight = nn.Parameter(new_weight)

# --------------------------------------------------------------------------- #
# 4.  Example usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    net = nn.Sequential(WidenableLinear(5, 10), nn.ReLU(), WidenableLinear(10, 5))

    # Print original network weights and biases
    print("Original network:")
    for i, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            print(f"Layer {i} weight:\n{layer.weight.data}")
            print(f"Layer {i} bias:\n{layer.bias.data}")

    print("-" * 100)

    # Function-preserving widen (Net2Wider): exact copy incoming + half outgoing + noise
    net1 = deepcopy(net)
    split_neuron(
        network=net1,
        input_layer_idx=0,
        output_layer_idx=2,
        neuron_idx=0,
        input_splitter=ExactCopy(),
        output_splitter=WithNoise(Half(), 0.02),
    )
    print("Function-preserving widen → hidden dim", net1[0].out_features)
    for i, layer in enumerate(net1):
        if isinstance(layer, nn.Linear):
            print(f"Layer {i} weight:\n{layer.weight.data}")
            print(f"Layer {i} bias:\n{layer.bias.data}")

    print("-" * 100)

    # Function-preserving widen (Net2Wider): exact copy incoming + orthogonal outgoing + noise
    net2 = deepcopy(net)
    split_neuron(
        network=net2,
        input_layer_idx=0,
        output_layer_idx=2,
        neuron_idx=0,
        input_splitter=ExactCopy(),
        output_splitter=WithNoise(OrthogonalDecomp(), 0.02),
    )
    print("Orthogonal+Noise widen → hidden dim", net2[0].out_features)
    for i, layer in enumerate(net2):
        if isinstance(layer, nn.Linear):
            print(f"Layer {i} weight:\n{layer.weight.data}")
            print(f"Layer {i} bias:\n{layer.bias.data}")
