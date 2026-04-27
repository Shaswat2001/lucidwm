"""Common network architectures for world models."""

from __future__ import annotations

import torch
import torch.nn as nn

class MLP(nn.Module):
    """General-purpose MLP with configurable depth and normalization.

    Args:
        in_dim: Input dimension.
        out_dim: Output dimension.
        hidden_dim: Hidden layer dimension. Default: 400.
        num_layers: Number of hidden layers. Default: 2.
        activation: Activation class. Default: nn.ELU.
        norm: Whether to use LayerNorm. Default: True.
        output_activation: Optional activation on output. Default: None.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 400,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.ELU,
        norm: bool = True,
        output_activation: type[nn.Module] | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # hidden layers
                if norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(activation())
            elif output_activation is not None:  # output layer
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
