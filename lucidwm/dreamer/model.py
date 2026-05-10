"""Dreamer V1: Model components"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from lucidwm_components.networks import MLP

@dataclass
class RSSMState:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

def stack_states(states: list[RSSMState]) -> RSSMState:
    return RSSMState(
        mean=torch.stack([s.mean for s in states], dim=1),
        std=torch.stack([s.std for s in states], dim=1),
        stoch=torch.stack([s.stoch for s in states], dim=1),
        deter=torch.stack([s.deter for s in states], dim=1),
    )

def flatten_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        mean=state.mean.reshape(-1, state.mean.shape[-1]),
        std=state.std.reshape(-1, state.std.shape[-1]),
        stoch=state.stoch.reshape(-1, state.stoch.shape[-1]),
        deter=state.deter.reshape(-1, state.deter.shape[-1]),
    )

def detach_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        mean=state.mean.detach(),
        std=state.std.detach(),
        stoch=state.stoch.detach(),
        deter=state.deter.detach(),
    )

class DenseDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int, num_layers: int = 2, distribution: str = "normal"):
        super().__init__()
        self.distribution = distribution
        self.out_dim = out_dim
        self.net = MLP(in_dim, out_dim, hidden_dim=hidden_size, num_layers=num_layers, activation=nn.ELU, norm=False)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        if self.distribution == "normal":
            return td.independent.Independent(td.Normal(x, 1), 1)
        if self.distribution == "binary":
            return td.independent.Independent(td.Bernoulli(logits=x), 1)
        raise NotImplementedError(self.distribution)

class ConvEncoder(nn.Module):
    """Dreamer V1 conv encoder: 64x64x3 -> 32*depth embedding."""

    def __init__(self, depth: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(depth, 2 * depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2), nn.ReLU(),
        )
        self.out_dim = 32 * depth

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return x.reshape(obs.shape[0], -1)

class ConvDecoder(nn.Module):
    """Dreamer V1 conv decoder: feature -> image mean."""

    def __init__(self, feat_dim: int, depth: int = 32):
        super().__init__()
        self.out_channels = 32 * depth
        self.fc = nn.Linear(feat_dim, self.out_channels)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(2 * depth, depth, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(depth, 3, 6, stride=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).reshape(-1, self.out_channels, 1, 1)
        return self.net(x)

class RSSM(nn.Module):
    """Gaussian recurrent state-space model from Dreamer V1."""

    def __init__(self, action_dim: int, stoch_size: int, deter_size: int, hidden_size: int, embed_dim: int):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.img_in = nn.Linear(stoch_size + action_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size, deter_size)
        self.img_hidden = nn.Linear(deter_size, hidden_size)
        self.img_out = nn.Linear(hidden_size, 2 * stoch_size)
        self.obs_hidden = nn.Linear(deter_size + embed_dim, hidden_size)
        self.obs_out = nn.Linear(hidden_size, 2 * stoch_size)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            mean=torch.zeros(batch_size, self.stoch_size, device=device),
            std=torch.zeros(batch_size, self.stoch_size, device=device),
            stoch=torch.zeros(batch_size, self.stoch_size, device=device),
            deter=torch.zeros(batch_size, self.deter_size, device=device),
        )

    def calculate_prior(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        x = torch.cat([prev_state.stoch, prev_action], dim=-1)
        x = F.elu(self.img_in(x))
        deter = self.gru(x, prev_state.deter)
        x = F.elu(self.img_hidden(deter))
        out = self.img_out(x)
        return self.calculate_stat(out, deter)

    def calculate_posterior(self, prior: RSSMState, embed: torch.Tensor) -> RSSMState:
        x = torch.cat([prior.deter, embed], dim=-1)
        x = F.elu(self.obs_hidden(x))
        stats = self.obs_out(x)
        return self.calculate_stat(stats, prior.deter)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.stoch, state.deter], dim=-1)

    def get_dist(self, state: RSSMState) -> td.Normal:
        return td.Normal(state.mean, state.std)

    def calculate_stat(self, stats: torch.Tensor, deter: torch.Tensor) -> RSSMState:
        mean, std = stats.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        stoch = td.Normal(mean, std).rsample()
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)

    def obs_step(self, prev_state: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        prior = self.calculate_prior(prev_state, prev_action)
        post = self.calculate_posterior(prior, embed)
        return post, prior

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, state: RSSMState | None = None) -> tuple[RSSMState, RSSMState]:
        batch_size, horizon = actions.shape[:2]
        if state is None:
            state = self.init_state(batch_size, actions.device)
        priors = []
        posts = []
        prev_state = state
        for t in range(horizon):
            post, prior = self.obs_step(prev_state, actions[:, t], embeds[:, t])
            posts.append(post)
            priors.append(prior)
            prev_state = post
        return stack_states(posts), stack_states(priors)

class TanhNormalActor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, units: int, init_std: float, min_std: float, mean_scale: float):
        super().__init__()
        self.net = MLP(feat_dim, 2 * action_dim, hidden_dim=units, num_layers=4, activation=nn.ELU, norm=False)
        self.raw_init_std = math.log(math.exp(init_std) - 1.0)
        self.min_std = min_std
        self.mean_scale = mean_scale

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(feat)
        mean, std = out.chunk(2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        return mean, std, torch.tanh(mean)

    def sample(self, feat: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std, mode = self.forward(feat)
        if deterministic:
            return mode, torch.zeros(feat.shape[0], device=feat.device)
        dist = td.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

class DreamerModel(nn.Module):
    def __init__(self, action_dim: int, args):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = ConvEncoder(depth=args.cnn_depth)
        self.rssm = RSSM(
            action_dim,
            stoch_size=args.stoch_size,
            deter_size=args.deter_size,
            hidden_size=args.rssm_hidden_size,
            embed_dim=self.encoder.out_dim,
        )
        feat_dim = args.stoch_size + args.deter_size
        self.decoder = ConvDecoder(feat_dim, depth=args.cnn_depth)
        self.reward = DenseDecoder(feat_dim, 1, hidden_size=args.num_units, num_layers=2)
        self.value = DenseDecoder(feat_dim, 1, hidden_size=args.num_units, num_layers=3)
        self.actor = TanhNormalActor(
            feat_dim, action_dim, units=args.num_units,
            init_std=args.action_init_std, min_std=args.min_std, mean_scale=args.mean_scale,
        )
        self.cont = (
            DenseDecoder(feat_dim, 1, hidden_size=args.num_units, num_layers=3, distribution="binary")
            if args.learn_cont else None
        )
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear | nn.Conv2d | nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
