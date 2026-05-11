"""PWM: Model components"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucidwm_components.networks import MLP
from lucidwm_components.distribution import SimNorm

class PWMModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, args):
        super(PWMModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            MLP(obs_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp),
        )
        self.dynamics = nn.Sequential(
            MLP(args.latent_dim + action_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp),
        )
        # Scalar reward head — MSE loss, matches official PWM
        self.rewards = MLP(args.latent_dim + action_dim, 1, args.hidden_dim, activation=nn.Mish)

        self.apply(self.init_weights)
        # Zero-init reward head final layer: prevents large reward predictions early in training
        self.zero([self.rewards.net[-1].weight])

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.02, 0.02)

    @staticmethod
    def zero(params):
        for p in params:
            p.data.fill_(0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_state(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict scalar reward. Returns shape (B,)."""
        return self.rewards(torch.cat([z, a], dim=-1)).squeeze(-1)


class PWMPolicy(nn.Module):
    def __init__(self, args, action_dim: int):
        super(PWMPolicy, self).__init__()

        self.net = MLP(args.latent_dim, action_dim, activation=nn.Mish)
        # Learnable scalar log_std (not state-dependent) — matches official PWM actor
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, z: torch.Tensor, deterministic: bool = False):
        mean = self.net(z)
        log_std = self.log_std.clamp(-10, 2)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean), torch.zeros(z.shape[0], device=z.device)

        noise = torch.randn_like(mean)
        raw = mean + std * noise
        action = torch.tanh(raw)
        log_prob = (-0.5 * noise.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob -= (2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(-1)
        return action, log_prob


class PWMCriticEnsemble(nn.Module):
    def __init__(self, args, num_critics: int = 3):
        super(PWMCriticEnsemble, self).__init__()

        self.critics = nn.ModuleList([
            MLP(args.latent_dim, 1, activation=nn.Mish) for _ in range(num_critics)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns (num_critics, B) tensor of per-head values."""
        return torch.stack([critic(z).squeeze(-1) for critic in self.critics])

    def min_value(self, z: torch.Tensor) -> torch.Tensor:
        """Pessimistic min across ensemble — use for actor/target computation."""
        return self.forward(z).min(dim=0).values

    def mean_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z).mean(dim=0)
