import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from aggregations import AGGREGATION_FUNCS
from pprint import pprint


class StatsWrapper(nn.Module):
    """
    Wraps an nn.Module to record neuron-level statistics during forward/backward.
    A neuron is defined as the computational unit spanning from input_layer to output_layer,
    including any activation functions in between.
    """

    def __init__(self, model: nn.Module, *, buffer_size: Optional[int] = None):
        super().__init__()
        self.model = model
        if buffer_size is not None and buffer_size <= 0:
            raise ValueError("buffer_size must be > 0 when provided")
        self._buffer_size: Optional[int] = buffer_size
        
        # Store ALL layers (Linear + activation layers) in order
        self.layers: List[nn.Module] = []
        self.layer_names: List[str] = []
        self.layer_types: List[str] = []
        
        # Storage for activations and gradients at each layer
        self._layer_outputs: Dict[int, torch.Tensor] = {}  # layer_idx -> output tensor
        self._layer_grads: Dict[int, torch.Tensor] = {}   # layer_idx -> gradient tensor
        
        # Temporal buffers: per (input_linear_idx, next_linear_idx) -> per-neuron -> per-metric deque
        # Structure:
        #   self._temporal_buffers[(in_idx, out_idx)][neuron_idx][metric_name] -> deque(maxlen=buffer_size)
        # We store ONLY atomic (non-percentile) metrics.
        self._temporal_buffers: Dict[Tuple[int, int], Dict[int, Dict[str, deque]]] = {}
        # Map each Linear layer index to the next Linear layer index (if any)
        self._linear_to_next_linear: Dict[int, Optional[int]] = {}
        
        self._extract_all_layers()
        self._build_adjacent_linear_map()
        self._register_hooks()

    def __len__(self) -> int:
        return len(self.model)

    def __iter__(self):
        return iter(self.model)

    def __getitem__(self, idx: int):
        return self.model[idx]

    def __repr__(self) -> str:
        return f"StatsWrapper({repr(self.model)})"

    def __getattr__(self, name: str) -> Any:
        try:
            # Try the wrapper first
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the underlying model
            return getattr(self.model, name)

    def __dir__(self) -> List[str]:
        # Combine attributes so that tab completion works nicely
        return sorted(set(list(super().__dir__()) + dir(self.model)))

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

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

    def _build_adjacent_linear_map(self) -> None:
        """Build a mapping from each Linear layer idx to the next Linear layer idx after it (if any)."""
        linear_indices: List[int] = [i for i, layer in enumerate(self.layers) if isinstance(layer, nn.Linear)]
        for idx_pos, i in enumerate(linear_indices):
            j = linear_indices[idx_pos + 1] if (idx_pos + 1) < len(linear_indices) else None
            if j is not None:
                self._linear_to_next_linear[i] = j

    # def _register_hooks(self) -> None:
    #     """Register only forward tensor hooks on all layers."""
    #     if getattr(self, "_hooks_registered", False):
    #         return
    #     self._hooks_registered = True

    #     for layer_idx, layer in enumerate(self.layers):
    #         layer.register_forward_hook(self._make_forward_hook(layer_idx))


    # def _make_forward_hook(self, layer_idx: int):
    #     def hook(module, inputs, output):
    #         if isinstance(output, torch.Tensor):
    #             # cache forward activations for stats (pre/post activation means)
    #             self._layer_outputs[layer_idx] = output.detach()
    #             # cache gradient wrt this tensor when backprop reaches it
    #             if output.requires_grad:
    #                 def save_grad(g, i=layer_idx):
    #                     self._layer_grads[i] = g.detach()
    #                 output.register_hook(save_grad)
    #     return hook

    def _register_hooks(self) -> None:
        if getattr(self, "_hooks_registered", False): return
        self._hooks_registered = True
        for layer_idx, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_forward_hook(layer_idx))
            layer.register_full_backward_hook(self._make_backward_hook(layer_idx))  # <-- add

    def _make_forward_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                self._layer_outputs[layer_idx] = output.detach()
        return hook

    def _make_backward_hook(self, layer_idx: int):
        def bhook(module, grad_input, grad_output):
            # grad_output[0] is grad wrt module's output
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                self._layer_grads[layer_idx] = grad_output[0].detach()
        return bhook

    def ensure_hooks(self) -> None:
        # remove old handles if you track them; otherwise just re-register
        self._layer_outputs.clear()
        self._layer_grads.clear()
        self.layers.clear(); self.layer_names.clear(); self.layer_types.clear()
        self._extract_all_layers()
        self._build_adjacent_linear_map()
        self._hooks_registered = False
        self._register_hooks()

    def __deepcopy__(self, memo):
        import copy
        # 1) deepcopy the wrapped model modules/params
        new_model = copy.deepcopy(self.model, memo)

        # 2) build a fresh wrapper (this runs __init__, rebuilds layers,
        #    and re-registers forward/backward hooks bound to the *new* self)
        new = StatsWrapper(new_model, buffer_size=self._buffer_size)

        # 3) optionally carry over temporal buffers (NOT per-step caches)
        if self._buffer_size is not None and self._temporal_buffers:
            new.load_temporal_state(self.dump_temporal_state())

        return new



    # def _make_backward_hook(self, layer_idx: int):
    #     def hook(module, grad_input, grad_output):
    #         # Only use the earliest Linear as the global "after-backward" trigger
    #         if getattr(self, "_first_linear_idx", None) != layer_idx:
    #             return

    #         # At this point, parameter .grad tensors should be populated for this backward pass
    #         for in_idx, out_idx in self._linear_to_next_linear.items():
    #             if out_idx is None:
    #                 continue
    #             try:
    #                 snapshot = self.get_layer_neuron_stats_atomic(in_idx, out_idx)
    #             except Exception:
    #                 snapshot = None
    #             if snapshot:
    #                 self._append_temporal_snapshot(in_idx, out_idx, snapshot)

    #         # Clear per-step caches to avoid holding onto tensors/graphs
    #         self.clear()
    #     return hook


    def capture_stats(self) -> None:
        """
        Call immediately after loss.backward() and before optimizer.step().
        Uses parameter .grad (now populated) + cached activations / tensor-grads.
        """
        
        for in_idx, out_idx in self._linear_to_next_linear.items():
            snap = self._get_layer_neuron_stats_atomic(in_idx, out_idx)
            self._append_temporal_snapshot(in_idx, out_idx, snap)
        # release per-step caches
        self.clear()




    def clear(self) -> None:
        """Clear stored activations and gradients."""
        self._layer_outputs.clear()
        self._layer_grads.clear()
        # Do not clear temporal buffers here; they are intended to persist across steps

    def clear_temporal_buffers(self) -> None:
        """Clear all temporal buffers (if allocated)."""
        self._temporal_buffers.clear()

    def dump_temporal_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of temporal buffers and config.

        Structure:
            {
              'buffer_size': int | None,
              'buffers': {
                 'in:out': {
                     'neuron_idx': {
                         'metric_name': [v0, v1, ...]
                     }
                 }
              }
            }
        """
        buffers: Dict[str, Any] = {}
        for (in_idx, out_idx), layer_buf in self._temporal_buffers.items():
            key = f"{in_idx}:{out_idx}"
            neuron_map: Dict[str, Any] = {}
            for neuron_idx, metric_map in layer_buf.items():
                metric_serialized: Dict[str, List[Optional[float]]] = {}
                for metric_name, dq in metric_map.items():
                    metric_serialized[metric_name] = list(dq)
                neuron_map[str(neuron_idx)] = metric_serialized
            buffers[key] = neuron_map
        return {
            'buffer_size': self._buffer_size,
            'buffers': buffers,
        }

    def load_temporal_state(self, state: Dict[str, Any]) -> None:
        """Restore temporal buffers from a snapshot created by dump_temporal_state."""
        if not state:
            return
        # Keep current buffer_size; only set if we had None and state provides a valid size
        snap_buf_size = state.get('buffer_size', None)
        if self._buffer_size is None and snap_buf_size is not None and snap_buf_size > 0:
            self._buffer_size = int(snap_buf_size)
        buffers = state.get('buffers', {}) or {}
        self._temporal_buffers.clear()
        for layer_key, neuron_map in buffers.items():
            try:
                in_str, out_str = layer_key.split(':')
                in_idx, out_idx = int(in_str), int(out_str)
            except Exception:
                continue
            layer_buf: Dict[int, Dict[str, deque]] = {}
            for neuron_str, metric_map in (neuron_map or {}).items():
                try:
                    neuron_idx = int(neuron_str)
                except Exception:
                    continue
                metric_deques: Dict[str, deque] = {}
                for metric_name, seq in (metric_map or {}).items():
                    dq = deque(maxlen=self._buffer_size if self._buffer_size is not None else None)
                    for v in (seq or []):
                        dq.append(None if v is None else float(v))
                    metric_deques[metric_name] = dq
                layer_buf[neuron_idx] = metric_deques
            self._temporal_buffers[(in_idx, out_idx)] = layer_buf

    def state_dict(self) -> Dict[str, Any]:
        return {
            'model_state': self.model.state_dict(),
            'temporal_state': self.dump_temporal_state(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model_state'])
        self.load_temporal_state(state_dict['temporal_state'])

    def _append_temporal_snapshot(self, input_layer_idx: int, output_layer_idx: int, stats_list: List[Dict[str, Any]]) -> None:
        """Append atomic stats into per-neuron, per-metric deques for a given layer pair."""
        if self._buffer_size is None:
            return
        key = (input_layer_idx, output_layer_idx)
        layer_buf = self._temporal_buffers.setdefault(key, {})
        for row in stats_list:
            neuron_idx = int(row.get('neuron_idx')) if 'neuron_idx' in row else None
            # ensure neuron_idx is present; if not, skip (shouldn't happen since caller adds it)
            if neuron_idx is None:
                continue
            neuron_buf = layer_buf.setdefault(neuron_idx, {})
            for metric_name, value in row.items():
                if metric_name == 'neuron_idx' or metric_name.endswith('_pct'):
                    continue
                dq = neuron_buf.get(metric_name)
                if dq is None:
                    dq = deque(maxlen=self._buffer_size)
                    neuron_buf[metric_name] = dq
                # append even if None (to preserve time alignment); temporal mean will skip None values
                dq.append(None if value is None else float(value))

    @torch.no_grad()
    def _get_neuron_stats_atomic(self, input_layer_idx: int, output_layer_idx: int, neuron_idx: int) -> Dict[str, Any]:
        """
        Get local spatial atomic statistics for a specific neuron (no percentiles).
        
        Args:
            input_layer_idx: Index of input layer (should be Linear layer)
            output_layer_idx: Index of output layer (should be Linear layer) 
            neuron_idx: Index of the neuron connecting these layers
            
        Returns:
            Dictionary containing atomic neuron statistics (no relative stats)
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
        
        def _safe_var_1d(vec: torch.Tensor) -> float:
            # Return 0.0 variance when there is only 1 element to avoid NaNs and undefined stats
            if vec.numel() < 2:
                return 0.0
            return vec.var(unbiased=False).item()

        # Input weights: weights from input_layer that connect TO this neuron
        input_weights = input_layer.weight[neuron_idx, :].detach()  # Shape: [in_features]
        stats['input_weights_mean'] = input_weights.mean().item()
        stats['input_weights_var'] = _safe_var_1d(input_weights)
        
        # Output weights: weights from output_layer that connect FROM this neuron
        if neuron_idx >= output_layer.in_features:
            stats['output_weights_mean'] = None
            stats['output_weights_var'] = None
        else:
            output_weights = output_layer.weight[:, neuron_idx].detach()  # Shape: [out_features]
            stats['output_weights_mean'] = output_weights.mean().item()
            stats['output_weights_var'] = _safe_var_1d(output_weights)
        
        # Bias term from input layer for this neuron
        if input_layer.bias is not None:
            stats['bias'] = input_layer.bias[neuron_idx].detach().item()
        else:
            stats['bias'] = None
        
        # Pre-activation: output of input_layer before any activation function
        if input_layer_idx not in self._layer_outputs:
            stats['pre_activation_mean'] = None
        else:
            pre_activation = self._layer_outputs[input_layer_idx]
            if len(pre_activation.shape) < 2:  # Not [batch_size, features] or higher dims
                stats['pre_activation_mean'] = None
            else:
                # Flatten to [batch_size, -1] then select neuron
                pre_activation_flat = pre_activation.view(pre_activation.shape[0], -1)
                if neuron_idx >= pre_activation_flat.shape[1]:
                    stats['pre_activation_mean'] = None
                else:
                    neuron_pre_activation = pre_activation_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['pre_activation_mean'] = neuron_pre_activation.mean().item()
        
        # Post-activation: output after going through activation layers between input and output
        # Find the last activation layer before output_layer_idx
        post_activation_layer_idx = None
        for i in range(input_layer_idx + 1, output_layer_idx):
            if isinstance(self.layers[i], (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.LeakyReLU)):
                post_activation_layer_idx = i
        
        if post_activation_layer_idx is None or post_activation_layer_idx not in self._layer_outputs:
            # No activation function found, post-activation = pre-activation
            stats['post_activation_mean'] = stats['pre_activation_mean']
        else:
            post_activation = self._layer_outputs[post_activation_layer_idx]
            if len(post_activation.shape) < 2:
                stats['post_activation_mean'] = None
            else:
                post_activation_flat = post_activation.view(post_activation.shape[0], -1)
                if neuron_idx >= post_activation_flat.shape[1]:
                    stats['post_activation_mean'] = None
                else:
                    neuron_post_activation = post_activation_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['post_activation_mean'] = neuron_post_activation.mean().item()
        
        # Gradients with respect to input weights
        if input_layer.weight.grad is None:
            stats['input_weight_grads_mean'] = None
            stats['input_weight_grads_var'] = None
        else:
            input_weight_grads = input_layer.weight.grad[neuron_idx, :].detach()
            stats['input_weight_grads_mean'] = input_weight_grads.mean().item()
            stats['input_weight_grads_var'] = _safe_var_1d(input_weight_grads)
        
        # Gradients with respect to output weights  
        if output_layer.weight.grad is None or neuron_idx >= output_layer.in_features:
            stats['output_weight_grads_mean'] = None
            stats['output_weight_grads_var'] = None
        else:
            output_weight_grads = output_layer.weight.grad[:, neuron_idx].detach()
            stats['output_weight_grads_mean'] = output_weight_grads.mean().item()
            stats['output_weight_grads_var'] = _safe_var_1d(output_weight_grads)
        
        # Gradient with respect to pre-activation (gradient wrt Linear layer output, before activation)
        if input_layer_idx not in self._layer_grads:
            stats['pre_activation_grad_mean'] = None
        else:
            pre_activation_grad = self._layer_grads[input_layer_idx]
            if len(pre_activation_grad.shape) < 2:
                stats['pre_activation_grad_mean'] = None
            else:
                pre_activation_grad_flat = pre_activation_grad.view(pre_activation_grad.shape[0], -1)
                if neuron_idx >= pre_activation_grad_flat.shape[1]:
                    stats['pre_activation_grad_mean'] = None
                else:
                    neuron_pre_grad = pre_activation_grad_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['pre_activation_grad_mean'] = neuron_pre_grad.mean().item()
        
        # Gradient with respect to post-activation (gradient wrt activation layer output, after activation)
        if post_activation_layer_idx is None or post_activation_layer_idx not in self._layer_grads:
            # No separate post-activation gradient - use pre-activation grad
            stats['post_activation_grad_mean'] = stats['pre_activation_grad_mean']
        else:
            post_activation_grad = self._layer_grads[post_activation_layer_idx]
            if len(post_activation_grad.shape) < 2:
                stats['post_activation_grad_mean'] = None
            else:
                post_activation_grad_flat = post_activation_grad.view(post_activation_grad.shape[0], -1)
                if neuron_idx >= post_activation_grad_flat.shape[1]:
                    stats['post_activation_grad_mean'] = None
                else:
                    neuron_post_grad = post_activation_grad_flat[:, neuron_idx]  # Shape: [batch_size]
                    # Aggregate across batch dimension
                    stats['post_activation_grad_mean'] = neuron_post_grad.mean().item()
        
        return stats

    @torch.no_grad()
    def get_neuron_stats(self, input_layer_idx: int, output_layer_idx: int, neuron_idx: int) -> Dict[str, Any]:
        """Get the most recent non-temporal stats for a specific neuron from the temporal buffer.

        Returns the most recent atomic stats (no percentiles) for the specified neuron.
        """
        if self._buffer_size is None:
            raise RuntimeError("Temporal buffers are not enabled; construct StatsWrapper with buffer_size > 0")
        
        key = (input_layer_idx, output_layer_idx)
        layer_buf = self._temporal_buffers.get(key)
        if not layer_buf:
            raise ValueError(f"No temporal buffer found for layer pair {input_layer_idx} -> {output_layer_idx}")
        
        neuron_buf = layer_buf.get(neuron_idx)
        if not neuron_buf:
            raise ValueError(f"No temporal buffer found for neuron {neuron_idx} in layer pair {input_layer_idx} -> {output_layer_idx}")
        
        # Get the most recent value for each metric
        stats = {'neuron_idx': neuron_idx}
        for metric_name, dq in neuron_buf.items():
            if dq:  # If deque is not empty
                # Get the most recent value (last in deque)
                most_recent = dq[-1]
                stats[metric_name] = most_recent
            else:
                stats[metric_name] = None
        
        return stats

    @torch.no_grad()
    def _get_layer_neuron_stats_atomic(self, input_layer_idx: int, output_layer_idx: int) -> List[Dict[str, Any]]:
        """
        Get atomic statistics for ALL neurons in the layer connecting `input_layer_idx` -> `output_layer_idx`.

        This captures a snapshot of per-neuron atomic metrics at the current time (based on the
        most recent forward/backward hooks).
        """
        if input_layer_idx >= len(self.layers) or output_layer_idx >= len(self.layers):
            raise ValueError(f"Layer indices out of range. Model has {len(self.layers)} layers.")

        input_layer = self.layers[input_layer_idx]
        output_layer = self.layers[output_layer_idx]

        if not isinstance(input_layer, nn.Linear):
            raise ValueError(f"Input layer (idx {input_layer_idx}) must be Linear, got {type(input_layer)}")
        if not isinstance(output_layer, nn.Linear):
            raise ValueError(f"Output layer (idx {output_layer_idx}) must be Linear, got {type(output_layer)}")

        num_neurons = input_layer.out_features
        stats_list: List[Dict[str, Any]] = []
        for neuron_idx in range(num_neurons):
            s = self._get_neuron_stats_atomic(input_layer_idx, output_layer_idx, neuron_idx)
            s['neuron_idx'] = neuron_idx
            stats_list.append(s)

        return stats_list

    # @torch.no_grad()
    # def _get_layer_neuron_stats(self, input_layer_idx: int, output_layer_idx: int) -> List[Dict[str, Any]]:
    #     """
    #     Get statistics for ALL neurons in the layer connecting `input_layer_idx` -> `output_layer_idx`,
    #     including within-layer percentiles.

    #     This captures a snapshot of per-neuron metrics at the current time (based on the
    #     most recent forward/backward hooks) and computes percentiles across the layer.
    #     """
    #     stats_list = self._get_layer_neuron_stats_atomic(input_layer_idx, output_layer_idx)

    #     if not stats_list:
    #         return stats_list

    #     # Compute within-layer percentiles for each metric and attach as {metric}_pct
    #     # Build arrays per key from layer_snapshot
    #     keys = [k for k in stats_list[0].keys() if k not in ('neuron_idx',)]
    #     arrays: Dict[str, List[float]] = {}
    #     for k in keys:
    #         vals: List[float] = []
    #         for row in stats_list:
    #             v = row.get(k, None)
    #             if v is not None:
    #                 vals.append(float(v))
    #         arrays[k] = vals
    #     # Attach percentiles
    #     for row in stats_list:
    #         for k in keys:
    #             v = row.get(k, None)
    #             arr = arrays[k]
    #             if v is None or len(arr) == 0:
    #                 row[f"{k}_pct"] = None
    #             else:
    #                 # Percentile as fraction of layer neurons with strictly lower value
    #                 try:
    #                     row[f"{k}_pct"] = float((sum(1 for x in arr if x < float(v))) / len(arr))
    #                 except Exception:
    #                     row[f"{k}_pct"] = None

    #     return stats_list

    @torch.no_grad()
    def get_neuron_stats_temporal(
        self,
        input_layer_idx: int,
        output_layer_idx: int,
        neuron_idx: int,
        temporal_window: int,
    ) -> Dict[str, Any]:
        """Temporal aggregation of per-neuron metrics over the last W steps for all registered aggregations.

        Returns keys formatted as tmp{W}__{agg}__{metric_name} for all available metrics
        recorded in the buffer.
        """
        if self._buffer_size is None:
            raise RuntimeError("Temporal buffers are not enabled; construct StatsWrapper with buffer_size > 0")
        key = (input_layer_idx, output_layer_idx)
        layer_buf = self._temporal_buffers.get(key)
        if not layer_buf:
            return {}
        neuron_buf = layer_buf.get(neuron_idx)
        if not neuron_buf:
            return {}
        W = temporal_window
        if W <= 0 or W > self._buffer_size:
            raise ValueError(f"temporal_window must be > 0 and <= buffer_size ({self._buffer_size})")
        out: Dict[str, Any] = {}
        for metric_name, dq in neuron_buf.items():
            recent = list(dq)[-W:] if dq else []
            # Most aggregation functions expect real numbers only; filter Nones here
            recent_filtered = [v for v in recent if v is not None]
            for agg_name, agg_fn in AGGREGATION_FUNCS.items():
                key = f"tmp{W}__{agg_name}__{metric_name}"
                out[key] = agg_fn(recent_filtered)
        return out

    @torch.no_grad()
    def _get_layer_neuron_stats_temporal(
        self,
        input_layer_idx: int,
        output_layer_idx: int,
        temporal_window: int,
    ) -> List[Dict[str, Any]]:
        """Temporal aggregation (mean) for ALL neurons in the layer over the last W steps.

        Returns one dict per neuron with keys tmp{W}__mean__{metric_name}.
        """
        if self._buffer_size is None:
            raise RuntimeError("Temporal buffers are not enabled; construct StatsWrapper with buffer_size > 0")
        key = (input_layer_idx, output_layer_idx)
        if key not in self._temporal_buffers:
            return []
        # Determine current number of neurons from the live layer (can change over time)
        input_layer = self.layers[input_layer_idx]
        if not isinstance(input_layer, nn.Linear):
            raise ValueError(f"Input layer (idx {input_layer_idx}) must be Linear, got {type(input_layer)}")
        num_neurons = input_layer.out_features
        results: List[Dict[str, Any]] = []
        for neuron_idx in range(num_neurons):
            agg = self.get_neuron_stats_temporal(input_layer_idx, output_layer_idx, neuron_idx, temporal_window)
            agg['neuron_idx'] = neuron_idx
            results.append(agg)
        return results

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
    # base = nn.Sequential(
    #     nn.Linear(5, 10),   # Layer 0
    #     nn.GELU(),          # Layer 1  
    #     nn.Linear(10, 8),   # Layer 2
    #     nn.GELU(),          # Layer 3
    #     nn.Linear(8, 3)     # Layer 4
    # )
    # model = StatsWrapper(base)

    # # Print layer information
    # print("Available layers:")
    # for i, name, layer_type, size in model.get_layer_info():
    #     size_str = f"(size: {size})" if size else ""
    #     print(f"Layer {i}: {name} - {layer_type} {size_str}")

    # # Dummy data + loss
    # x = torch.randn(4, 5)
    # target = torch.randint(0, 3, (4,))
    # criterion = nn.CrossEntropyLoss()

    # # Forward + backward
    # model.train()
    # out = model(x)
    # loss = criterion(out, target)
    # loss.backward()

    # # Get stats for neuron 2 connecting layer 0 (Linear) to layer 2 (Linear)
    # # This neuron goes: Layer0 -> Layer1(ReLU) -> connects to Layer2
    # neuron_stats = model.get_neuron_stats(input_layer_idx=0, output_layer_idx=2, neuron_idx=2)
    
    # print(f"\nStats for neuron 2 connecting layer 0 -> layer 2:")
    # for key, value in neuron_stats.items():
    #     if value is not None:
    #         print(f"{key}: {value:.6f}")
    #     else:
    #         print(f"{key}: None")

    # Device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tiny toy model: Linear -> GELU -> Linear (so layer indices 0 and 2 are Linear)
    base = nn.Sequential(
        nn.Linear(5, 6),  # 5 inputs -> 6 hidden
        nn.GELU(),
        nn.Linear(6, 3),  # 6 hidden -> 3 outputs (pretend 3-class classification)
    )
    model = StatsWrapper(base, buffer_size=8).to(device)
    print("\nModel:")
    print(model)

    # Show layer info
    print("\nLayer info (idx, name, type, size):")
    for i, name, layer_type, size in model.get_layer_info():
        print(f"  {i:>2}  {name:<10}  {layer_type:<12}  size={size}")

    # Show available temporal aggregations
    agg_names = sorted(AGGREGATION_FUNCS.keys())
    print(f"\nTemporal aggregations available ({len(agg_names)}):")
    print(", ".join(agg_names))

    # Fake training loop to populate temporal buffers (window=8)
    batch_size = 32
    steps = 20  # > 8 to fill the window fully
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    in_features = 5
    num_classes = 3
    torch.manual_seed(42)

    print("\nTraining (fake data) ...")
    for step in range(1, steps + 1):
        x = torch.randn(batch_size, in_features, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        print("========================================================")
        print("STEP")
        print("========================================================")
        model.capture_stats()
        optimizer.step()

        print(f"  step {step:02d}  loss={loss.item():.4f}")

    print("Done training.\n")

    # Compute temporal stats (window=8) for the connection 0 -> 2
    W = 8
    input_layer_idx = 0
    output_layer_idx = 2
    print(f"Collecting temporal stats with window={W} for layer pair {input_layer_idx}->{output_layer_idx} ...")
    layer_temporal = model._get_layer_neuron_stats_temporal(input_layer_idx, output_layer_idx, W)
    
    if not layer_temporal:
        print("No temporal stats available (buffer may be empty).")
    else:
        print(f"Temporal stats returned for {len(layer_temporal)} neurons.")
        # Print keys count summary
        key_counts = [len([k for k in row.keys() if k != 'neuron_idx']) for row in layer_temporal]
        print("Per-neuron metric key counts:")
        for row, cnt in zip(layer_temporal, key_counts):
            print(f"  neuron {row['neuron_idx']}: {cnt} keys")

        # Verbose: show all keys for neuron 0
        sample = next((row for row in layer_temporal if row.get('neuron_idx') == 0), layer_temporal[0])
        print("\nDetailed temporal metrics for neuron 0:")
        for k in sorted(k for k in sample.keys() if k != 'neuron_idx'):
            print(f"  {k}: {sample[k]}")

        # Also show a compact preview for remaining neurons
        print("\nCompact previews for remaining neurons:")
        for row in layer_temporal:
            if row.get('neuron_idx') == sample['neuron_idx']:
                continue
            keys = sorted(k for k in row.keys() if k != 'neuron_idx')
            preview_keys = keys[:10]
            print(f"  neuron {row['neuron_idx']}: showing {len(preview_keys)} of {len(keys)} keys")
            for k in preview_keys:
                print(f"    {k}: {row[k]}")

    # Optional: show a tiny snapshot of raw atomic stats at the end for context
    # print("\nRaw spatial stats (non-temporal) for neuron 0 at end-of-run:")
    # try:
    #     spatial = model.get_neuron_stats(input_layer_idx, output_layer_idx, 0)
    #     for k in sorted(k for k in spatial.keys() if not k.endswith('_pct'))[:20]:
    #         print(f"  {k}: {spatial[k]}")
    # except Exception as e:
    #     print(f"  Failed to get spatial stats: {e}")
