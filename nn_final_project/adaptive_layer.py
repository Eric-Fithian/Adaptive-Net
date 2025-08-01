import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 adapt_interval: int = 100,
                 k_split: float = 2.0,
                 k_prune: float = 0.1,
                 activation: nn.Module = nn.GELU()):
        super().__init__()
        
        # Core learnable parameters for this layer
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Adaptation settings
        self.adapt_interval = adapt_interval
        self.k_split = k_split
        self.k_prune = k_prune
        self.activation_function = activation
        self.train_steps = 0
        
        # Buffers for activation & gradient stats
        self.register_buffer("act_sum", torch.zeros(out_features))
        self.register_buffer("act_sq_sum", torch.zeros(out_features))
        self.register_buffer("grad_sum", torch.zeros(out_features))
        self.register_buffer("count", torch.zeros(1))
        
        self._hook = None
        self.next_layer = None  # we'll set this after creating the next layer

    def set_next_layer(self, layer):
        """
        Store a reference to the downstream layer.
        The layer should have 'weight' and 'in_features' attributes.
        """
        # Check if layer has the necessary attributes
        if hasattr(layer, 'weight') and hasattr(layer, 'in_features'):
            self.next_layer = layer
        else:
            raise ValueError("Next layer must have 'weight' and 'in_features' attributes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.activation_function(out)
        
        # Accumulate activation stats
        with torch.no_grad():
            batch_size = out.shape[0]
            self.count += batch_size
            self.act_sum += out.sum(dim=0)
            self.act_sq_sum += (out ** 2).sum(dim=0)
        
        # Register a backward hook once
        if out.requires_grad and self._hook is None:
            self._hook = out.register_hook(self._capture_grad_hook)
        
        return out
    
    def _capture_grad_hook(self, grad_output: torch.Tensor):
        # grad_output is shape (batch_size, out_features)
        with torch.no_grad():
            self.grad_sum += grad_output.abs().sum(dim=0)
    
    def backward_step(self):
        """
        Call after each backward() to increment step count and maybe adapt structure.
        """
        self.train_steps += 1
        if self.train_steps % self.adapt_interval == 0:
            self.adapt_structure()
            self._reset_stats()
            self._hook = None

    def adapt_structure(self):
        """
        Adjust neurons by either splitting or pruning, and also update the next layer's input dimension accordingly.
        """
        if self.count.item() < 1:
            return  # no data yet

        # Compute average metrics
        act_mean = self.act_sum / self.count
        act_sq_mean = self.act_sq_sum / self.count
        var_activation = act_sq_mean - act_mean ** 2
        
        grad_mean = self.grad_sum / self.count
        
        avg_grad = grad_mean.mean().item()
        avg_var = var_activation.mean().item()

        # Only process neurons up to the size of our stats buffer
        max_neurons = min(self.out_features, grad_mean.size(0))
        
        neurons_to_split = []
        neurons_to_prune = []
        
        i = 0
        while i < max_neurons:
            g_i = grad_mean[i].item()
            v_i = var_activation[i].item()
            
            # Split
            if g_i > self.k_split * avg_grad and v_i > self.k_split * avg_var:
                neurons_to_split.append(i)
            
            # Prune
            if g_i < self.k_prune * avg_grad and v_i < self.k_prune * avg_var:
                neurons_to_prune.append(i)
            
            i += 1
            
        # Split neurons
        for idx in neurons_to_split:
            self._split_neuron(idx)
        
        # Prune neurons
        for idx in reversed(neurons_to_prune):
            self._prune_neuron(idx)
        
    
    def _reset_stats(self):
        out_size = self.out_features
        device = self.weight.device
        self.act_sum = torch.zeros(out_size, device=device)
        self.act_sq_sum = torch.zeros(out_size, device=device)
        self.grad_sum = torch.zeros(out_size, device=device)
        self.count = torch.zeros(1, device=device)
    
    def _split_neuron(self, idx: int):
        """
        Duplicate neuron idx (scaling the original and new by 1/2 to preserve function),
        and add small noise to the new neuron's weights and bias to break symmetry.
        Also update next_layer if present.
        """
        with torch.no_grad():
            old_w = self.weight.data[idx].clone()
            old_b = self.bias.data[idx].clone()

            epsilon = 1e-2  # small noise factor
            
            # Original and new neuron: half the value, but new gets noise
            self.weight.data[idx] = 0.5 * old_w
            self.bias.data[idx] = 0.5 * old_b

            noise_w = torch.randn_like(old_w) * epsilon
            noise_b = torch.randn_like(old_b) * epsilon

            new_w = 0.5 * old_w + noise_w
            new_b = 0.5 * old_b + noise_b

            new_weight = torch.cat([self.weight.data, new_w.unsqueeze(0)], dim=0)
            new_bias = torch.cat([self.bias.data, new_b.unsqueeze(0)], dim=0)

            self.weight = nn.Parameter(new_weight)
            self.bias = nn.Parameter(new_bias)
            self.out_features += 1

            # Expand statistics buffers to match the new layer size
            device = self.weight.device
            self.act_sum = torch.cat([self.act_sum, torch.zeros(1, device=device)], dim=0)
            self.act_sq_sum = torch.cat([self.act_sq_sum, torch.zeros(1, device=device)], dim=0)
            self.grad_sum = torch.cat([self.grad_sum, torch.zeros(1, device=device)], dim=0)

            if self.next_layer is not None:
                self._expand_downstream_layer(idx)

    def _prune_neuron(self, idx: int):
        """
        Remove neuron idx from this layer and from the next layer if present.
        """
        with torch.no_grad():
            # Slice out row from self
            mask = torch.ones(self.out_features, dtype=torch.bool, device=self.weight.device)
            mask[idx] = False
            
            self.weight = nn.Parameter(self.weight.data[mask])
            self.bias = nn.Parameter(self.bias.data[mask])
            self.out_features -= 1

            # Also prune statistics buffers to match the new layer size
            self.act_sum = self.act_sum[mask]
            self.act_sq_sum = self.act_sq_sum[mask]
            self.grad_sum = self.grad_sum[mask]

            # Also prune from next layer if it exists
            if self.next_layer is not None:
                self._prune_downstream_layer(idx)
                
    def _expand_downstream_layer(self, idx: int):
        """
        Insert a new column in next_layer.weight corresponding to the new neuron.
        Handles both nn.Linear and AdaptiveLayer downstream layers.
        """
        layer = self.next_layer
        if layer is None:
            return
            
        # Check if layer has weight and in_features attributes
        if not hasattr(layer, 'weight') or not hasattr(layer, 'in_features'):
            return
            
        with torch.no_grad():
            w_next = layer.weight.data  # shape: (out_features, in_features)
            col_to_duplicate = w_next[:, idx].clone()
            
            epsilon = 1e-2
            noise_col = torch.randn_like(col_to_duplicate) * epsilon

            # Adjust original
            w_next[:, idx] = 0.5 * col_to_duplicate

            # Create noisy copy
            new_col = 0.5 * col_to_duplicate + noise_col
            new_col = new_col.unsqueeze(1)  # shape: (out_features, 1)

            updated_weight = torch.cat([w_next, new_col], dim=1)
            layer.weight = nn.Parameter(updated_weight)
            layer.in_features += 1

    def _prune_downstream_layer(self, idx: int):
        """
        Remove the column at 'idx' from next_layer.weight.
        Handles both nn.Linear and AdaptiveLayer downstream layers.
        """
        layer = self.next_layer
        if layer is None:
            return
        
        # Check if layer has weight and in_features attributes
        if not hasattr(layer, 'weight') or not hasattr(layer, 'in_features'):
            return
        
        with torch.no_grad():
            w_next = layer.weight.data  # shape: (out_features, in_features)
            mask = torch.ones(w_next.shape[1], dtype=torch.bool, device=w_next.device)
            mask[idx] = False
            # keep all columns except idx
            updated_weight = w_next[:, mask]
            layer.weight = nn.Parameter(updated_weight)
            layer.in_features -= 1