import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional


class StatsWrapper(nn.Module):
    """
    Wraps an nn.Module to record neuron-level statistics during forward/backward.
    A neuron is defined as the computational unit spanning from input_layer to output_layer,
    including any activation functions in between.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        # Store ALL layers (Linear + activation layers) in order
        self.layers: List[nn.Module] = []
        self.layer_names: List[str] = []
        self.layer_types: List[str] = []
        
        # Storage for activations and gradients at each layer
        self._layer_outputs: Dict[int, torch.Tensor] = {}  # layer_idx -> output tensor
        self._layer_grads: Dict[int, torch.Tensor] = {}   # layer_idx -> gradient tensor
        
        self._extract_all_layers()
        self._register_hooks()

    def _extract_all_layers(self) -> None:
        """Extract ALL layers (Linear + activations) in sequential order."""
        def extract_from_sequential(module, prefix=""):
            if isinstance(module, nn.Sequential):
                for i, child in enumerate(module.children()):
                    child_name = f"{prefix}.{i}" if prefix else str(i)
                    self.layers.append(child)
                    self.layer_names.append(child_name)
                    self.layer_types.append(type(child).__name__)
            else:
                # For non-sequential models, try to extract in order
                for name, child in module.named_children():
                    child_name = f"{prefix}.{name}" if prefix else name
                    self.layers.append(child)
                    self.layer_names.append(child_name)
                    self.layer_types.append(type(child).__name__)
        
        extract_from_sequential(self.model)

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def _register_hooks(self) -> None:
        """Register hooks to capture outputs and gradients at each layer."""
        for layer_idx, layer in enumerate(self.layers):
            # Forward hook to capture layer output
            layer.register_forward_hook(self._make_forward_hook(layer_idx))
            
            # Backward hook to capture gradients - only use backward hook for consistency
            layer.register_full_backward_hook(self._make_backward_hook(layer_idx))

    def _make_forward_hook(self, layer_idx: int):
        def hook(module, input, output):
            self._layer_outputs[layer_idx] = output.detach().clone()
        return hook

    def _make_backward_hook(self, layer_idx: int):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self._layer_grads[layer_idx] = grad_output[0].detach().clone()
        return hook

    def clear(self) -> None:
        """Clear stored activations and gradients."""
        self._layer_outputs.clear()
        self._layer_grads.clear()

    def get_neuron_stats(self, input_layer_idx: int, output_layer_idx: int, neuron_idx: int) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a specific neuron.
        
        Args:
            input_layer_idx: Index of input layer (should be Linear layer)
            output_layer_idx: Index of output layer (should be Linear layer) 
            neuron_idx: Index of the neuron connecting these layers
            
        Returns:
            Dictionary containing all neuron statistics
        """
        if input_layer_idx >= len(self.layers) or output_layer_idx >= len(self.layers):
            raise ValueError(f"Layer indices out of range. Model has {len(self.layers)} layers.")
        
        input_layer = self.layers[input_layer_idx]
        output_layer = self.layers[output_layer_idx]
        
        if not isinstance(input_layer, nn.Linear):
            raise ValueError(f"Input layer (idx {input_layer_idx}) must be Linear, got {type(input_layer)}")
        if not isinstance(output_layer, nn.Linear):
            raise ValueError(f"Output layer (idx {output_layer_idx}) must be Linear, got {type(output_layer)}")
        
        if neuron_idx >= input_layer.out_features:
            raise ValueError(f"neuron_idx {neuron_idx} exceeds input layer output size {input_layer.out_features}")

        stats = {}
        
        # Input weights: weights from input_layer that connect TO this neuron
        input_weights = input_layer.weight[neuron_idx, :].detach()  # Shape: [in_features]
        stats['input_weights_mean'] = input_weights.mean().item()
        stats['input_weights_var'] = input_weights.var(unbiased=False).item()
        
        # Output weights: weights from output_layer that connect FROM this neuron
        if neuron_idx >= output_layer.in_features:
            stats['output_weights_mean'] = None
            stats['output_weights_var'] = None
        else:
            output_weights = output_layer.weight[:, neuron_idx].detach()  # Shape: [out_features]
            stats['output_weights_mean'] = output_weights.mean().item()
            stats['output_weights_var'] = output_weights.var(unbiased=False).item()
        
        # Bias term from input layer for this neuron
        if input_layer.bias is not None:
            stats['bias'] = input_layer.bias[neuron_idx].detach().item()
        else:
            stats['bias'] = None
        
        # Pre-activation: output of input_layer before any activation function
        if input_layer_idx not in self._layer_outputs:
            stats['pre_activation_mean'] = None
            stats['pre_activation_var'] = None
        else:
            pre_activation = self._layer_outputs[input_layer_idx]
            if len(pre_activation.shape) < 2:  # Not [batch_size, features] or higher dims
                stats['pre_activation_mean'] = None
                stats['pre_activation_var'] = None
            else:
                # Flatten to [batch_size, -1] then select neuron
                pre_activation_flat = pre_activation.view(pre_activation.shape[0], -1)
                if neuron_idx >= pre_activation_flat.shape[1]:
                    stats['pre_activation_mean'] = None
                    stats['pre_activation_var'] = None
                else:
                    neuron_pre_activation = pre_activation_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['pre_activation_mean'] = neuron_pre_activation.mean().item()
                    stats['pre_activation_var'] = neuron_pre_activation.var(unbiased=False).item()
        
        # Post-activation: output after going through activation layers between input and output
        # Find the last activation layer before output_layer_idx
        post_activation_layer_idx = None
        for i in range(input_layer_idx + 1, output_layer_idx):
            if isinstance(self.layers[i], (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.LeakyReLU)):
                post_activation_layer_idx = i
        
        if post_activation_layer_idx is None or post_activation_layer_idx not in self._layer_outputs:
            # No activation function found, post-activation = pre-activation
            stats['post_activation_mean'] = stats['pre_activation_mean']
            stats['post_activation_var'] = stats['pre_activation_var']
        else:
            post_activation = self._layer_outputs[post_activation_layer_idx]
            if len(post_activation.shape) < 2:
                stats['post_activation_mean'] = None
                stats['post_activation_var'] = None
            else:
                post_activation_flat = post_activation.view(post_activation.shape[0], -1)
                if neuron_idx >= post_activation_flat.shape[1]:
                    stats['post_activation_mean'] = None
                    stats['post_activation_var'] = None
                else:
                    neuron_post_activation = post_activation_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['post_activation_mean'] = neuron_post_activation.mean().item()
                    stats['post_activation_var'] = neuron_post_activation.var(unbiased=False).item()
        
        # Gradients with respect to input weights
        if input_layer.weight.grad is None:
            stats['input_weight_grads_mean'] = None
            stats['input_weight_grads_var'] = None
        else:
            input_weight_grads = input_layer.weight.grad[neuron_idx, :].detach()
            stats['input_weight_grads_mean'] = input_weight_grads.mean().item()
            stats['input_weight_grads_var'] = input_weight_grads.var(unbiased=False).item()
        
        # Gradients with respect to output weights  
        if output_layer.weight.grad is None or neuron_idx >= output_layer.in_features:
            stats['output_weight_grads_mean'] = None
            stats['output_weight_grads_var'] = None
        else:
            output_weight_grads = output_layer.weight.grad[:, neuron_idx].detach()
            stats['output_weight_grads_mean'] = output_weight_grads.mean().item()
            stats['output_weight_grads_var'] = output_weight_grads.var(unbiased=False).item()
        
        # Gradient with respect to pre-activation (gradient wrt Linear layer output, before activation)
        if input_layer_idx not in self._layer_grads:
            stats['pre_activation_grad_mean'] = None
            stats['pre_activation_grad_var'] = None
        else:
            pre_activation_grad = self._layer_grads[input_layer_idx]
            if len(pre_activation_grad.shape) < 2:
                stats['pre_activation_grad_mean'] = None
                stats['pre_activation_grad_var'] = None
            else:
                pre_activation_grad_flat = pre_activation_grad.view(pre_activation_grad.shape[0], -1)
                if neuron_idx >= pre_activation_grad_flat.shape[1]:
                    stats['pre_activation_grad_mean'] = None
                    stats['pre_activation_grad_var'] = None
                else:
                    neuron_pre_grad = pre_activation_grad_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['pre_activation_grad_mean'] = neuron_pre_grad.mean().item()
                    stats['pre_activation_grad_var'] = neuron_pre_grad.var(unbiased=False).item()
        
        # Gradient with respect to post-activation (gradient wrt activation layer output, after activation)
        if post_activation_layer_idx is None or post_activation_layer_idx not in self._layer_grads:
            # No separate post-activation gradient - use pre-activation grad
            stats['post_activation_grad_mean'] = stats['pre_activation_grad_mean']
            stats['post_activation_grad_var'] = stats['pre_activation_grad_var']
        else:
            post_activation_grad = self._layer_grads[post_activation_layer_idx]
            if len(post_activation_grad.shape) < 2:
                stats['post_activation_grad_mean'] = None
                stats['post_activation_grad_var'] = None
            else:
                post_activation_grad_flat = post_activation_grad.view(post_activation_grad.shape[0], -1)
                if neuron_idx >= post_activation_grad_flat.shape[1]:
                    stats['post_activation_grad_mean'] = None
                    stats['post_activation_grad_var'] = None
                else:
                    neuron_post_grad = post_activation_grad_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['post_activation_grad_mean'] = neuron_post_grad.mean().item()
                    stats['post_activation_grad_var'] = neuron_post_grad.var(unbiased=False).item()
        
        return stats

    def get_layer_info(self) -> List[Tuple[int, str, str, int]]:
        """Get information about available layers for analysis."""
        info = []
        for i, (name, layer, layer_type) in enumerate(zip(self.layer_names, self.layers, self.layer_types)):
            if isinstance(layer, nn.Linear):
                size = layer.out_features
            else:
                size = None
            info.append((i, name, layer_type, size))
        return info


# Example usage:
if __name__ == '__main__':
    # Define a simple model: Linear->ReLU->Linear->ReLU->Linear
    base = nn.Sequential(
        nn.Linear(5, 10),   # Layer 0
        nn.GELU(),          # Layer 1  
        nn.Linear(10, 8),   # Layer 2
        nn.GELU(),          # Layer 3
        nn.Linear(8, 3)     # Layer 4
    )
    model = StatsWrapper(base)

    # Print layer information
    print("Available layers:")
    for i, name, layer_type, size in model.get_layer_info():
        size_str = f"(size: {size})" if size else ""
        print(f"Layer {i}: {name} - {layer_type} {size_str}")

    # Dummy data + loss
    x = torch.randn(4, 5)
    target = torch.randint(0, 3, (4,))
    criterion = nn.CrossEntropyLoss()

    # Forward + backward
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Get stats for neuron 2 connecting layer 0 (Linear) to layer 2 (Linear)
    # This neuron goes: Layer0 -> Layer1(ReLU) -> connects to Layer2
    neuron_stats = model.get_neuron_stats(input_layer_idx=0, output_layer_idx=2, neuron_idx=2)
    
    print(f"\nStats for neuron 2 connecting layer 0 -> layer 2:")
    for key, value in neuron_stats.items():
        if value is not None:
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: None")