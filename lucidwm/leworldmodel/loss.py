"""LeWorldModel: Loss functions"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer from the official LeWM release."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D)."""
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a / (a.norm(p=2, dim=0, keepdim=True) + 1e-8)

        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
