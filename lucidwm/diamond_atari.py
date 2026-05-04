"""LucidWM: DIAMOND on Atari.

Paper:     "Diffusion for World Modeling: Visual Details Matter in Atari" (Alonso et al., 2024)
Reference: https://github.com/eloialonso/diamond

This implementation follows the main DIAMOND structure:
  - diffusion world model trained directly in pixel space
  - separate reward / termination CNN-LSTM model
  - separate actor-critic CNN-LSTM trained in imagination with REINFORCE + value baseline
  - Atari RGB observations with frame skip 4, reward clipping, and terminal-on-life-loss

As with the rest of LucidWM, this is a compact educational implementation rather than a line-by-line
port of the official codebase.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: DIAMOND on Atari")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--prefill-steps", type=int, default=5_000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--updates-per-epoch", type=int, default=400)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)

    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--burnin-len", type=int, default=4)
    parser.add_argument("--imagine-horizon", type=int, default=15)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay-diffusion", type=float, default=1e-2)
    parser.add_argument("--weight-decay-rt", type=float, default=1e-2)
    parser.add_argument("--weight-decay-ac", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    parser.add_argument("--diffusion-channels", type=int, default=64)
    parser.add_argument("--diffusion-cond-dim", type=int, default=256)
    parser.add_argument("--rt-channels", type=int, default=32)
    parser.add_argument("--rt-cond-dim", type=int, default=128)
    parser.add_argument("--lstm-dim", type=int, default=512)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=2e-3)
    parser.add_argument("--sigma-max", type=float, default=5.0)
    parser.add_argument("--sigma-rho", type=float, default=7.0)
    parser.add_argument("--p-mean", type=float, default=-0.4)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--sample-steps", type=int, default=3)

    parser.add_argument("--discount", type=float, default=0.985)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--collect-eps", type=float, default=0.01)
    return parser.parse_args()


class CHWObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

    def observation(self, obs):
        return obs.transpose(2, 0, 1).astype(np.float32)


def make_diamond_atari_env(env_id: str, seed: int, img_size: int) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=img_size,
        terminal_on_life_loss=True,
        grayscale_obs=False,
        scale_obs=True,
    )
    env = CHWObservation(env)
    env.reset(seed=seed)
    return env


def clip_reward(reward: float) -> float:
    return float(np.sign(reward))


def reward_class(reward: torch.Tensor) -> torch.Tensor:
    return torch.clamp(reward.sign().long() + 1, 0, 2)


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
    data2 = sigma_data**2
    c_in = 1.0 / torch.sqrt(sigma2 + data2)
    c_out = sigma * sigma_data / torch.sqrt(sigma2 + data2)
    c_skip = data2 / (sigma2 + data2)
    c_noise = 0.25 * torch.log(sigma)
    return c_in, c_out, c_skip, c_noise


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(10_000.0),
                half,
                device=x.device,
            )
            * -1
        )
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

    def _modulate(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod = self.cond_proj(cond)
        s1, b1, s2, b2 = torch.split(mod, [self.in_channels, self.in_channels, self.out_channels, self.out_channels], dim=-1)
        h = self.norm1(x)
        h = self._modulate(h, s1, b1)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self._modulate(h, s2, b2)
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
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int, cond_dim: int):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = nn.ModuleList([AdaGNResidualBlock(base_channels, base_channels, cond_dim) for _ in range(2)])
        self.down2_in = Downsample(base_channels)
        self.down2 = nn.ModuleList([AdaGNResidualBlock(base_channels, base_channels, cond_dim) for _ in range(2)])
        self.down3_in = Downsample(base_channels)
        self.down3 = nn.ModuleList([AdaGNResidualBlock(base_channels, base_channels, cond_dim) for _ in range(2)])
        self.down4_in = Downsample(base_channels)
        self.down4 = nn.ModuleList([AdaGNResidualBlock(base_channels, base_channels, cond_dim) for _ in range(2)])
        self.mid = nn.ModuleList([AdaGNResidualBlock(base_channels, base_channels, cond_dim) for _ in range(2)])
        self.up4 = Upsample(base_channels)
        self.up4_blocks = nn.ModuleList([AdaGNResidualBlock(2 * base_channels, base_channels, cond_dim) for _ in range(2)])
        self.up3 = Upsample(base_channels)
        self.up3_blocks = nn.ModuleList([AdaGNResidualBlock(2 * base_channels, base_channels, cond_dim) for _ in range(2)])
        self.up2 = Upsample(base_channels)
        self.up2_blocks = nn.ModuleList([AdaGNResidualBlock(2 * base_channels, base_channels, cond_dim) for _ in range(2)])
        self.up1 = Upsample(base_channels)
        self.up1_blocks = nn.ModuleList([AdaGNResidualBlock(2 * base_channels, base_channels, cond_dim) for _ in range(2)])
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def _run_blocks(self, blocks: nn.ModuleList, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, cond)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        s1 = self._run_blocks(self.down1, x, cond)
        s2 = self._run_blocks(self.down2, self.down2_in(s1), cond)
        s3 = self._run_blocks(self.down3, self.down3_in(s2), cond)
        s4 = self._run_blocks(self.down4, self.down4_in(s3), cond)
        h = self._run_blocks(self.mid, s4, cond)

        h = self.up4(h)
        h = torch.cat([h, s3], dim=1)
        h = self._run_blocks(self.up4_blocks, h, cond)

        h = self.up3(h)
        h = torch.cat([h, s2], dim=1)
        h = self._run_blocks(self.up3_blocks, h, cond)

        h = self.up2(h)
        h = torch.cat([h, s1], dim=1)
        h = self._run_blocks(self.up2_blocks, h, cond)

        h = self.up1(h)
        h = torch.cat([h, x], dim=1)
        h = self._run_blocks(self.up1_blocks, h, cond)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


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

    def forward(
        self,
        noisy_next: torch.Tensor,
        context_obs: torch.Tensor,
        context_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
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


def sample_log_sigma(batch_size: int, p_mean: float, p_std: float, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    log_sigma = torch.randn(batch_size, device=device) * p_std + p_mean
    return log_sigma.exp().clamp(sigma_min, sigma_max)


def diffusion_loss(model: DiffusionWorldModel, batch: dict[str, torch.Tensor], args) -> torch.Tensor:
    context_obs = batch["obs"][:, : args.context_len]
    context_actions = batch["action"][:, : args.context_len]
    target_obs = batch["obs"][:, args.context_len]
    sigma = sample_log_sigma(target_obs.shape[0], args.p_mean, args.p_std, args.sigma_min, args.sigma_max, target_obs.device)
    noisy_next = target_obs + torch.randn_like(target_obs) * sigma[:, None, None, None]
    pred = model(noisy_next, context_obs, context_actions, sigma)
    return F.mse_loss(pred, target_obs)


def reward_termination_loss(model: RewardTerminationModel, batch: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, float]]:
    obs = batch["obs"][:, : args.burnin_len + args.imagine_horizon]
    actions = batch["action"][:, : args.burnin_len + args.imagine_horizon]
    rewards = batch["reward"][:, : args.burnin_len + args.imagine_horizon]
    done = batch["done"][:, : args.burnin_len + args.imagine_horizon]
    state = model.init_state(obs.shape[0], obs.device)
    reward_losses = []
    done_losses = []
    for t in range(obs.shape[1]):
        reward_logits, done_logit, state = model.step(obs[:, t], actions[:, t], state)
        reward_losses.append(F.cross_entropy(reward_logits, reward_class(rewards[:, t]), reduction="none"))
        done_losses.append(F.binary_cross_entropy_with_logits(done_logit.squeeze(-1), done[:, t], reduction="none"))
    reward_loss = torch.stack(reward_losses, dim=1).mean()
    done_loss = torch.stack(done_losses, dim=1).mean()
    total = reward_loss + done_loss
    return total, {
        "losses/reward_model": reward_loss.item(),
        "losses/done_model": done_loss.item(),
    }


def lambda_return(reward: torch.Tensor, value: torch.Tensor, discount: torch.Tensor, bootstrap: torch.Tensor, lambda_: float) -> torch.Tensor:
    returns = torch.zeros_like(reward)
    next_value = bootstrap
    for t in reversed(range(reward.shape[0])):
        next_value = reward[t] + discount[t] * ((1 - lambda_) * value[t] + lambda_ * next_value)
        returns[t] = next_value
    return returns


def sample_next_obs(
    model: DiffusionWorldModel,
    context_obs: torch.Tensor,
    context_actions: torch.Tensor,
    args,
) -> torch.Tensor:
    batch_size = context_obs.shape[0]
    device = context_obs.device
    x = torch.randn(batch_size, 3, context_obs.shape[-2], context_obs.shape[-1], device=device) * args.sigma_max
    sigmas = sigma_schedule(args.sample_steps, args.sigma_min, args.sigma_max, args.sigma_rho, device)
    for idx in range(args.sample_steps):
        sigma = sigmas[idx].expand(batch_size)
        denoised = model(x, context_obs, context_actions, sigma)
        d = (x - denoised) / sigma[:, None, None, None]
        x = x + (sigmas[idx + 1] - sigmas[idx]) * d
    return x.clamp(0.0, 1.0)


def actor_critic_loss(
    diffusion_model: DiffusionWorldModel,
    rt_model: RewardTerminationModel,
    actor_critic: ActorCritic,
    batch: dict[str, torch.Tensor],
    args,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    obs = batch["obs"]
    actions = batch["action"]
    batch_size = obs.shape[0]
    device = obs.device

    actor_state = actor_critic.init_state(batch_size, device)
    for t in range(args.burnin_len):
        _, _, actor_state = actor_critic.step(obs[:, t], actor_state)

    rt_state = rt_model.init_state(batch_size, device)
    for t in range(max(0, args.burnin_len - 1)):
        _, _, rt_state = rt_model.step(obs[:, t], actions[:, t], rt_state)

    obs_hist = obs[:, : args.context_len].clone()
    if args.context_len > 1:
        act_hist = actions[:, : args.context_len - 1].clone()
    else:
        act_hist = torch.zeros(batch_size, 0, actions.shape[-1], device=device)
    current_obs = obs[:, args.context_len - 1]

    log_probs = []
    entropies = []
    values = []
    rewards = []
    discounts = []

    for _ in range(args.imagine_horizon):
        logits, value, actor_state = actor_critic.step(current_obs, actor_state)
        dist = td.Categorical(logits=logits)
        action_idx = dist.sample()
        action = F.one_hot(action_idx, actor_critic.action_dim).float()
        with torch.no_grad():
            reward_logits, done_logit, rt_state = rt_model.step(current_obs, action, rt_state)
            reward_pred = (reward_logits.softmax(dim=-1) * reward_logits.new_tensor([-1.0, 0.0, 1.0])).sum(dim=-1)
            done_prob = torch.sigmoid(done_logit.squeeze(-1))

            full_actions = torch.cat([act_hist, action.unsqueeze(1)], dim=1) if args.context_len > 1 else action.unsqueeze(1)
            next_obs = sample_next_obs(diffusion_model, obs_hist, full_actions, args)

        log_probs.append(dist.log_prob(action_idx))
        entropies.append(dist.entropy())
        values.append(value)
        rewards.append(reward_pred)
        discounts.append(args.discount * (1.0 - done_prob))

        if args.context_len > 1:
            act_hist = full_actions[:, 1:]
        obs_hist = torch.cat([obs_hist[:, 1:], next_obs.unsqueeze(1)], dim=1)
        current_obs = next_obs

    with torch.no_grad():
        _, bootstrap, _ = actor_critic.step(current_obs, actor_state)

    reward_t = torch.stack(rewards, dim=0)
    value_t = torch.stack(values, dim=0)
    discount_t = torch.stack(discounts, dim=0)
    log_prob_t = torch.stack(log_probs, dim=0)
    entropy_t = torch.stack(entropies, dim=0)

    returns = lambda_return(reward_t[:-1], value_t[:-1], discount_t[:-1], bootstrap, args.lambda_return)
    advantages = returns - value_t[:-1]
    policy_loss = -(log_prob_t[:-1] * advantages.detach() + args.entropy_weight * entropy_t[:-1]).mean()
    value_loss = F.mse_loss(value_t[:-1], returns.detach())
    metrics = {
        "losses/policy": policy_loss.item(),
        "losses/value": value_loss.item(),
        "algo/imagined_reward": reward_t.mean().item(),
        "algo/imagined_value": value_t.mean().item(),
    }
    return policy_loss, value_loss, metrics


def tensor_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.tensor(batch["obs"], dtype=torch.float32, device=device),
        "action": torch.tensor(batch["action"], dtype=torch.float32, device=device),
        "reward": torch.tensor(batch["reward"], dtype=torch.float32, device=device),
        "done": torch.tensor(batch["done"], dtype=torch.float32, device=device),
    }


def evaluate_agent(actor_critic: ActorCritic, env_id: str, num_episodes: int, img_size: int, seed: int, device: torch.device) -> dict[str, float]:
    returns = []
    lengths = []
    for ep in range(num_episodes):
        env = make_diamond_atari_env(env_id, seed + 100 + ep, img_size)
        obs, _ = env.reset(seed=seed + 100 + ep)
        state = actor_critic.init_state(1, device)
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, state = actor_critic.act(obs_t, state, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(ep_length)
        env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def collect_steps(
    env: gym.Env,
    actor_critic: ActorCritic,
    buffer: ReplayBuffer,
    num_steps: int,
    action_dim: int,
    device: torch.device,
    deterministic: bool,
    epsilon: float,
    seed: int | None = None,
):
    obs, _ = env.reset(seed=seed)
    state = actor_critic.init_state(1, device)
    episodic = []
    episode_return = 0.0
    episode_length = 0

    for _ in range(num_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, action_one_hot, state = actor_critic.act(obs_t, state, deterministic=deterministic, eps=epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        clipped = clip_reward(reward)
        buffer.add_step(obs, action_one_hot, clipped, done)
        episode_return += reward
        episode_length += 1
        obs = next_obs
        if done:
            episodic.append((episode_return, episode_length))
            obs, _ = env.reset()
            state = actor_critic.init_state(1, device)
            episode_return = 0.0
            episode_length = 0

    return episodic


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    env = make_diamond_atari_env(args.env_id, args.seed, args.img_size)
    obs_shape = tuple(env.observation_space.shape)
    action_dim = int(env.action_space.n)

    diffusion_model = DiffusionWorldModel(
        action_dim=action_dim,
        context_len=args.context_len,
        base_channels=args.diffusion_channels,
        cond_dim=args.diffusion_cond_dim,
        sigma_data=args.sigma_data,
    ).to(device)
    rt_model = RewardTerminationModel(
        action_dim=action_dim,
        channels=args.rt_channels,
        cond_dim=args.rt_cond_dim,
        lstm_dim=args.lstm_dim,
    ).to(device)
    actor_critic = ActorCritic(action_dim=action_dim, lstm_dim=args.lstm_dim).to(device)

    diffusion_opt = optim.AdamW(
        diffusion_model.parameters(),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay_diffusion,
    )
    rt_opt = optim.AdamW(
        rt_model.parameters(),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay_rt,
    )
    ac_opt = optim.AdamW(
        actor_critic.parameters(),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay_ac,
    )

    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
        seq_len=args.context_len + args.imagine_horizon,
    )
    logger = Logger(
        project=args.wandb_project,
        name=f"diamond_{args.env_id}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    print(f"DIAMOND Atari | env={args.env_id} | obs={obs_shape} | act={action_dim} | device={device}")
    print("Collecting prefill experience...")
    prefill_metrics = collect_steps(
        env,
        actor_critic,
        buffer,
        args.prefill_steps,
        action_dim,
        device,
        deterministic=False,
        epsilon=1.0,
        seed=args.seed,
    )
    total_steps = args.prefill_steps
    for ret, length in prefill_metrics:
        logger.log({"charts/episodic_return": ret, "charts/episodic_length": length}, step=total_steps)

    seq_len = args.context_len + args.imagine_horizon
    epoch = 0
    while total_steps < args.total_steps:
        epoch += 1
        episodic = collect_steps(
            env,
            actor_critic,
            buffer,
            args.steps_per_epoch,
            action_dim,
            device,
            deterministic=False,
            epsilon=args.collect_eps,
        )
        total_steps += args.steps_per_epoch
        for ret, length in episodic:
            logger.log({"charts/episodic_return": ret, "charts/episodic_length": length}, step=total_steps)

        if buffer.num_episodes < 1:
            continue

        diff_metrics = {}
        rt_metrics = {}
        ac_metrics = {}
        for _ in range(args.updates_per_epoch):
            try:
                batch_np = buffer.sample(args.batch_size, seq_len=seq_len)
            except ValueError:
                break
            batch = tensor_batch(batch_np, device)

            diffusion_opt.zero_grad(set_to_none=True)
            diff_loss = diffusion_loss(diffusion_model, batch, args)
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), args.grad_clip)
            diffusion_opt.step()
            diff_metrics = {"losses/diffusion": diff_loss.item()}

            rt_opt.zero_grad(set_to_none=True)
            rt_loss, rt_metrics = reward_termination_loss(rt_model, batch, args)
            rt_loss.backward()
            torch.nn.utils.clip_grad_norm_(rt_model.parameters(), args.grad_clip)
            rt_opt.step()

            ac_opt.zero_grad(set_to_none=True)
            policy_loss, value_loss, ac_metrics = actor_critic_loss(diffusion_model, rt_model, actor_critic, batch, args)
            ac_total = policy_loss + value_loss
            ac_total.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.grad_clip)
            ac_opt.step()

        metrics = {}
        metrics.update(diff_metrics)
        metrics.update(rt_metrics)
        metrics.update(ac_metrics)
        if metrics:
            logger.log(metrics, step=total_steps)
            print(
                f"epoch={epoch:04d} steps={total_steps:06d} "
                f"diff={metrics.get('losses/diffusion', 0.0):.4f} "
                f"reward={metrics.get('losses/reward_model', 0.0):.4f} "
                f"policy={metrics.get('losses/policy', 0.0):.4f} "
                f"value={metrics.get('losses/value', 0.0):.4f}"
            )

        if total_steps % args.eval_freq == 0:
            actor_critic.eval()
            eval_result = evaluate_agent(actor_critic, args.env_id, args.eval_episodes, args.img_size, args.seed, device)
            logger.log(
                {
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                    "charts/eval_length": eval_result["mean_length"],
                },
                step=total_steps,
            )
            print(
                f"eval steps={total_steps:06d} return={eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}"
            )
            actor_critic.train()

    torch.save(
        {
            "diffusion_model": diffusion_model.state_dict(),
            "reward_termination_model": rt_model.state_dict(),
            "actor_critic": actor_critic.state_dict(),
            "args": vars(args),
        },
        f"diamond_{args.env_id.replace('/', '_')}_s{args.seed}.pt",
    )
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
