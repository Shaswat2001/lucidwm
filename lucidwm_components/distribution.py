"""
Distribution utilities for world models.

Concrete implementations of mathematical transforms and distributions
used across multiple algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# SimNorm
class SimNorm(nn.Module):
    """Simplicial normalization. Groups of V elements -> softmax each group."""

    def __init__(self, dim: int):
        super(SimNorm, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(*x.shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


# Two Hot Distribution (Discrete regression via two-hot encoding)
class TwoHotDist:
    """Two-hot encoded discrete regression distribution.

    Represents a scalar value as a distribution over a fixed set of bins.
    The value is encoded by placing weight on the two nearest bins
    proportional to the distance.

    Args:
        num_bins: Number of bins. Default: 255.
        low: Lower bound of bin range. Default: -20.0.
        high: Upper bound of bin range. Default: 20.0.
    """
    def __init__(self, num_bins: int, low: int = -20, high: int = 20):
        self.num_bins = num_bins
        self.low = low
        self.high = high
        self.bins = torch.linspace(low, high, num_bins)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar values as two-hot vectors.

        Args:
            x: (...) scalar values

        Returns:
            two_hot: (..., num_bins) two-hot encoded targets
        """
        bins = self.bins.to(x.device)
        x = x.clamp(self.low, self.high)

        # Find the two nearest bins
        below = torch.bucketize(x, bins) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1

        # Compute weights (linear interpolation)
        below_val = bins[below]
        above_val = bins[above]
        weight_above = (x - below_val) / (above_val - below_val + 1e-8)
        weight_below = 1.0 - weight_above

        # Build two-hot
        two_hot = torch.zeros(*x.shape, self.num_bins, device=x.device)
        two_hot.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        two_hot.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))

        return two_hot

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to scalar values via expected value.

        Args:
            logits: (..., num_bins) raw logits from prediction head

        Returns:
            values: (...) decoded scalar values
        """
        bins = self.bins.to(logits.device)
        probs = F.softmax(logits, dim=-1)
        return (probs * bins).sum(dim=-1)