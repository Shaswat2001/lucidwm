"""PWM: Model components"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lucidwm_components.networks import MLP
from lucidwm_components.distribution import TwoHotDist, SimNorm

class PWMModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, args):
        super(PWMModel, self).__init__()

        self.encoder = nn.Sequential(
            MLP(obs_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp),
        )
        self.dynamics = nn.Sequential(
            MLP(args.latent_dim + action_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp),
        )
        self.rewards = MLP(args.latent_dim + action_dim, args.num_bins, args.hidden_dim, activation=nn.Mish)
        self.two_hot_distribution = TwoHotDist(num_bins=args.num_bins)

        self.apply(self.init_weights)
        self.zero([self.rewards[-1].weight])

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.02, 0.02)
        elif isinstance(m, nn.ParameterList):
            for i, p in enumerate(m):
                if p.dim() == 3:  # Linear
                    nn.init.trunc_normal_(p, std=0.02)  # Weight
                    nn.init.constant_(m[i + 1], 0)  # Bias
    
    @staticmethod
    def zero(params):
        for p in params:
            p.data.fill_(0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_state(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.rewards(torch.cat([z, a], dim=-1))

    def reward_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.two_hot_distribution.encode(x)

    def reward_scalar(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.two_hot_distribution.decode(self.reward(z, a))

class PWMPolicy(nn.Module):
    def __init__(self, args, action_dim: int):
        super(PWMPolicy, self).__init__()

        self.net = MLP(args.latent_dim, 2 * action_dim, activation=nn.Mish)

    def forward(self, z: torch.Tensor, deterministic: bool = False):
        out = self.net(z)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
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

    def forward(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [critic(z).squeeze(-1) for critic in self.critics]

    def mean_value(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack(self.forward(z)).mean(dim=0)
