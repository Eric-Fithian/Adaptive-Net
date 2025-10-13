import torch
import torch.nn as nn
from typing import Tuple, Union

class WidenableLinear(nn.Linear):
    def add_output_feature(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor
    ) -> None:
        # weight: (in_features,), bias: tensor
        if weight.shape != (self.in_features,):
            raise ValueError(f"Expected weight shape ({self.in_features},), got {weight.shape}")
        
        bias_val = bias

        # new parameters
        W_new = torch.cat([self.weight, weight.unsqueeze(0)], dim=0)
        b_new = torch.cat([self.bias, bias_val.unsqueeze(0)])

        self.out_features += 1
        self.weight = nn.Parameter(W_new)
        self.bias   = nn.Parameter(b_new)

    def add_input_feature(
        self,
        weight: torch.Tensor
    ) -> None:
        # weight: (out_features,)
        if weight.shape != (self.out_features,):
            raise ValueError(f"Expected weight shape ({self.out_features},), got {weight.shape}")
        W_new = torch.cat([self.weight, weight.unsqueeze(1)], dim=1)

        self.in_features += 1
        self.weight = nn.Parameter(W_new)

    def set_output_feature(
        self,
        neuron_idx: int,
        weight: torch.Tensor,
        bias: torch.Tensor
    ) -> None:
        # weight: (in_features,), bias: tensor
        if weight.shape != (self.in_features,):
            raise ValueError(f"Expected weight shape ({self.in_features},), got {weight.shape}")
        
        bias_val = bias

        self.weight.data[neuron_idx, :] = weight
        self.bias.data[neuron_idx]      = bias_val

    def set_input_feature(
        self,
        neuron_idx: int,
        weight: torch.Tensor
    ) -> None:
        # weight: (out_features,)
        if weight.shape != (self.out_features,):
            raise ValueError(f"Expected weight shape ({self.out_features},), got {weight.shape}")
        self.weight.data[:, neuron_idx] = weight

