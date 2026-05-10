"""DIAMOND: Model components (diffusion world model + reward/termination + actor-critic)"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

def one_hot_action(action: int, action_dim: int) -> np.ndarray:
    vec = np.zeros(action_dim, dtype=np.float32)
    vec[action] = 1.0
    return vec

def sigma_schedule(num_steps: int, sigma_min: float, sigma_max: float, rho: float, device: torch.device) -> torch.Tensor:
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)

def edm_preconditioning(sigma: torch.Tensor, sigma_data: float):
    sigma2 = sigma.square()
    data2 = sigma_data ** 2
    c_in = 1.0 / torch.sqrt(sigma2 + data2)
    c_out = sigma * sigma_data / torch.sqrt(sigma2 + data2)
    c_skip = data2 / (sigma2 + data2)
    c_noise = 0.25 * torch.log(sigma)
    return c_in, c_out, c_skip, c_noise

def reward_class(reward: torch.Tensor) -> torch.Tensor:
    return torch.clamp(reward.sign().long() + 1, 0, 2)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10_000.0), half, device=x.device) * -1)
        args = x[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb

class AdaGNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, 2 * (in_channels + out_channels))
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.cond_proj(cond)
        s1, b1, s2, b2 = torch.split(mod, [self.in_channels, self.in_channels, self.out_channels, self.out_channels], dim=-1)
        h = self.norm1(x)
        h = h * (1 + s1[:, :, None, None]) + b1[:, :, None, None]
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = h * (1 + s2[:, :, None, None]) + b2[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int, cond_dim: int):
        super().__init__()
        C = base_channels
        self.in_conv = nn.Conv2d(in_channels, C, 3, padding=1)
        self.down1 = nn.ModuleList([AdaGNResidualBlock(C, C, cond_dim) for _ in range(2)])
        self.down2_in = Downsample(C)
        self.down2 = nn.ModuleList([AdaGNResidualBlock(C, C, cond_dim) for _ in range(2)])
        self.down3_in = Downsample(C)
        self.down3 = nn.ModuleList([AdaGNResidualBlock(C, C, cond_dim) for _ in range(2)])
        self.down4_in = Downsample(C)
        self.down4 = nn.ModuleList([AdaGNResidualBlock(C, C, cond_dim) for _ in range(2)])
        self.mid = nn.ModuleList([AdaGNResidualBlock(C, C, cond_dim) for _ in range(2)])
        self.up4 = Upsample(C)
        self.up4_blocks = nn.ModuleList([AdaGNResidualBlock(2 * C, C, cond_dim) for _ in range(2)])
        self.up3 = Upsample(C)
        self.up3_blocks = nn.ModuleList([AdaGNResidualBlock(2 * C, C, cond_dim) for _ in range(2)])
        self.up2 = Upsample(C)
        self.up2_blocks = nn.ModuleList([AdaGNResidualBlock(2 * C, C, cond_dim) for _ in range(2)])
        self.up1 = Upsample(C)
        self.up1_blocks = nn.ModuleList([AdaGNResidualBlock(2 * C, C, cond_dim) for _ in range(2)])
        self.out_norm = nn.GroupNorm(8, C)
        self.out_conv = nn.Conv2d(C, out_channels, 3, padding=1)

    def run_blocks(self, blocks: nn.ModuleList, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, cond)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        s1 = self.run_blocks(self.down1, x, cond)
        s2 = self.run_blocks(self.down2, self.down2_in(s1), cond)
        s3 = self.run_blocks(self.down3, self.down3_in(s2), cond)
        s4 = self.run_blocks(self.down4, self.down4_in(s3), cond)
        h = self.run_blocks(self.mid, s4, cond)
        h = self.up4(h)
        h = self.run_blocks(self.up4_blocks, torch.cat([h, s3], dim=1), cond)
        h = self.up3(h)
        h = self.run_blocks(self.up3_blocks, torch.cat([h, s2], dim=1), cond)
        h = self.up2(h)
        h = self.run_blocks(self.up2_blocks, torch.cat([h, s1], dim=1), cond)
        h = self.up1(h)
        h = self.run_blocks(self.up1_blocks, torch.cat([h, x], dim=1), cond)
        return self.out_conv(F.silu(self.out_norm(h)))

class DiffusionWorldModel(nn.Module):
    def __init__(self, action_dim: int, context_len: int, base_channels: int, cond_dim: int, sigma_data: float):
        super().__init__()
        self.action_dim = action_dim
        self.context_len = context_len
        self.sigma_data = sigma_data
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(context_len * action_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.unet = ConditionalUNet(
            in_channels=3 * (context_len + 1),
            out_channels=3,
            base_channels=base_channels,
            cond_dim=cond_dim,
        )

    def forward(self, noisy_next: torch.Tensor, context_obs: torch.Tensor, context_actions: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        c_in, c_out, c_skip, c_noise = edm_preconditioning(sigma, self.sigma_data)
        c_in = c_in[:, None, None, None]
        c_out = c_out[:, None, None, None]
        c_skip = c_skip[:, None, None, None]
        cond = self.time_embed(c_noise) + self.action_embed(context_actions.reshape(context_actions.shape[0], -1))
        inp = torch.cat([c_in * noisy_next, context_obs.reshape(context_obs.shape[0], -1, context_obs.shape[-2], context_obs.shape[-1])], dim=1)
        pred = self.unet(inp, cond)
        return c_skip * noisy_next + c_out * pred

class ResidualCNNBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], blocks_per_stage: list[int]):
        super().__init__()
        layers = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        current = channels[0]
        for stage, out_channels in enumerate(channels):
            if stage > 0:
                layers.append(nn.Conv2d(current, out_channels, 3, padding=1))
                current = out_channels
            for _ in range(blocks_per_stage[stage]):
                layers.append(ResidualCNNBlock(current))
            layers.append(nn.MaxPool2d(2, 2))
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1] * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(x.shape[0], -1)

@dataclass
class RecurrentState:
    h: torch.Tensor
    c: torch.Tensor

class RewardTerminationModel(nn.Module):
    def __init__(self, action_dim: int, channels: int, cond_dim: int, lstm_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(3, [channels, channels, channels, channels], [2, 2, 2, 2])
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.fc = nn.Linear(self.encoder.out_dim + cond_dim, lstm_dim)
        self.lstm = nn.LSTMCell(lstm_dim, lstm_dim)
        self.reward_head = nn.Linear(lstm_dim, 3)
        self.done_head = nn.Linear(lstm_dim, 1)

    def init_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        return RecurrentState(
            h=torch.zeros(batch_size, self.lstm.hidden_size, device=device),
            c=torch.zeros(batch_size, self.lstm.hidden_size, device=device),
        )

    def step(self, obs: torch.Tensor, action: torch.Tensor, state: RecurrentState) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        feat = self.encoder(obs)
        act = self.action_proj(action)
        x = F.silu(self.fc(torch.cat([feat, act], dim=-1)))
        h, c = self.lstm(x, (state.h, state.c))
        return self.reward_head(h), self.done_head(h), RecurrentState(h=h, c=c)

class ActorCritic(nn.Module):
    def __init__(self, action_dim: int, lstm_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = ConvEncoder(3, [32, 32, 64, 64], [1, 1, 1, 1])
        self.fc = nn.Linear(self.encoder.out_dim, lstm_dim)
        self.lstm = nn.LSTMCell(lstm_dim, lstm_dim)
        self.policy_head = nn.Linear(lstm_dim, action_dim)
        self.value_head = nn.Linear(lstm_dim, 1)

    def init_state(self, batch_size: int, device: torch.device) -> RecurrentState:
        return RecurrentState(
            h=torch.zeros(batch_size, self.lstm.hidden_size, device=device),
            c=torch.zeros(batch_size, self.lstm.hidden_size, device=device),
        )

    def step(self, obs: torch.Tensor, state: RecurrentState) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        feat = F.silu(self.fc(self.encoder(obs)))
        h, c = self.lstm(feat, (state.h, state.c))
        return self.policy_head(h), self.value_head(h).squeeze(-1), RecurrentState(h=h, c=c)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: RecurrentState, deterministic: bool, eps: float = 0.0) -> tuple[int, np.ndarray, RecurrentState]:
        logits, _, next_state = self.step(obs, state)
        if not deterministic and np.random.rand() < eps:
            action = int(np.random.randint(self.action_dim))
        else:
            dist = td.Categorical(logits=logits)
            action = int(logits.argmax(dim=-1).item()) if deterministic else int(dist.sample().item())
        return action, one_hot_action(action, self.action_dim), next_state
