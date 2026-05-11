"""
Distribution utilities for world models.

Concrete implementations of mathematical transforms and distributions
used across multiple algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# SimNorm
class SimNorm(nn.Module):
    """Simplicial normalization. Groups of V elements -> softmax each group."""

    def __init__(self, dim: int, temp: float = 1.0):
        super(SimNorm, self).__init__()
        self.dim = dim
        self.temp = temp
    
    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(*x.shape[:-1], -1, self.dim)
        x = F.softmax(x / self.temp, dim=-1)
        return x.view(*shape)


# Two Hot Distribution (Discrete regression via two-hot encoding)
class TwoHotDist:
    """Two-hot encoded discrete regression distribution.

    Represents a scalar value as a distribution over a fixed set of bins.
    The value is encoded by placing weight on the two nearest bins
    proportional to the distance.

    Adapted from https://github.com/nicklashansen/tdmpc2.

    Args:
        num_bins: Number of bins. Default: 255.
        low: Lower bound of bin range. Default: -20.0.
        high: Upper bound of bin range. Default: 20.0.
    """
    def __init__(self, num_bins: int, low: int = -20, high: int = 20):
        self.num_bins = num_bins
        self.low = low
        self.high = high

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode scalar values as two-hot vectors.

        Args:
            x: (...) scalar values

        Returns:
            two_hot: (..., num_bins) two-hot encoded targets
        """

        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return symlog(x)

        x = torch.clamp(symlog(x), self.low, self.high)

        # bin_width = (high - low) / (num_bins - 1): spacing between adjacent bin centres
        bin_width = (self.high - self.low) / (self.num_bins - 1)
        below = torch.floor((x - self.low) / bin_width).long().clamp(0, self.num_bins - 2)
        above = below + 1
        bin_offset = ((x - self.low) / bin_width - below.float()).unsqueeze(-1)

        # Build two-hot
        two_hot = torch.zeros(*x.shape, self.num_bins, device=x.device)
        two_hot.scatter_(-1, below.unsqueeze(-1), 1 - bin_offset)
        two_hot.scatter_(-1, above.unsqueeze(-1), bin_offset)

        return two_hot

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to scalar values via expected value with symexp.

        Args:
            logits: (..., num_bins) raw logits from prediction head

        Returns:
            values: (..., 1) decoded scalar values
        """
        if self.num_bins == 0:
            return logits
        elif self.num_bins == 1:
            return symexp(logits)
        dreg_bins = torch.linspace(self.low, self.high, self.num_bins, device=logits.device, dtype=logits.dtype)
        logits = F.softmax(logits, dim=-1)
        logits = torch.sum(logits * dreg_bins, dim=-1, keepdim=True)
        return symexp(logits)

    def decode_no_symexp(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits without the final symexp transform.

        Used in first-order gradient (FoG) policy training. symexp has an
        unbounded derivative near large values; omitting it prevents gradient
        explosion when backpropagating through multi-step imagined rollouts.

        Args:
            logits: (..., num_bins) raw logits from prediction head

        Returns:
            values: (...) decoded scalar values (no symexp, no keepdim)
        """
        if self.num_bins == 0:
            return logits
        elif self.num_bins == 1:
            return logits
        dreg_bins = torch.linspace(self.low, self.high, self.num_bins, device=logits.device, dtype=logits.dtype)
        probs = F.softmax(logits, dim=-1)
        return (probs * dreg_bins).sum(dim=-1)
