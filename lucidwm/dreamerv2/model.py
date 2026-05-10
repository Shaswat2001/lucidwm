"""Dreamer V2: Model components"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from lucidwm_components.networks import MLP

@dataclass
class RSSMState:
    logits: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor

def stack_states(states: list[RSSMState]) -> RSSMState:
    return RSSMState(
        logits=torch.stack([s.logits for s in states], dim=1),
        stoch=torch.stack([s.stoch for s in states], dim=1),
        deter=torch.stack([s.deter for s in states], dim=1),
    )

def flatten_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        logits=state.logits.reshape(-1, *state.logits.shape[-2:]),
        stoch=state.stoch.reshape(-1, *state.stoch.shape[-2:]),
        deter=state.deter.reshape(-1, state.deter.shape[-1]),
    )

def detach_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        logits=state.logits.detach(),
        stoch=state.stoch.detach(),
        deter=state.deter.detach(),
    )

def set_requires_grad(modules: list[nn.Module], requires_grad: bool):
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(requires_grad)

def sample_straight_through(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    sample = td.OneHotCategorical(logits=logits).sample()
    return sample + probs - probs.detach()

def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    post = td.OneHotCategorical(logits=post_logits)
    prior = td.OneHotCategorical(logits=prior_logits)
    return td.kl_divergence(post, prior).sum(dim=-1)

class DenseDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int, num_layers: int = 2, distribution: str = "normal"):
        super().__init__()
        self.distribution = distribution
        self.net = MLP(in_dim, out_dim, hidden_dim=hidden_size, num_layers=num_layers, activation=nn.ELU, norm=False)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        if self.distribution == "normal":
            return td.Independent(td.Normal(x, 1), 1)
        if self.distribution == "binary":
            return td.Independent(td.Bernoulli(logits=x), 1)
        raise NotImplementedError(self.distribution)

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, depth: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(depth, 2 * depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2), nn.ReLU(),
        )
        self.out_dim = 32 * depth

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).reshape(obs.shape[0], -1)

class ConvDecoder(nn.Module):
    def __init__(self, feat_dim: int, out_channels: int, depth: int = 48):
        super().__init__()
        self.out_channels = 32 * depth
        self.fc = nn.Linear(feat_dim, self.out_channels)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(2 * depth, depth, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(depth, out_channels, 6, stride=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).reshape(-1, self.out_channels, 1, 1)
        return self.net(x)

class CategoricalRSSM(nn.Module):
    def __init__(self, action_dim: int, stoch_size: int, stoch_classes: int, deter_size: int, hidden_size: int, embed_dim: int):
        super().__init__()
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.deter_size = deter_size
        stoch_dim = stoch_size * stoch_classes
        self.img_in = nn.Linear(stoch_dim + action_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size, deter_size)
        self.img_hidden = nn.Linear(deter_size, hidden_size)
        self.img_out = nn.Linear(hidden_size, stoch_dim)
        self.obs_hidden = nn.Linear(deter_size + embed_dim, hidden_size)
        self.obs_out = nn.Linear(hidden_size, stoch_dim)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            logits=torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device),
            stoch=torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device),
            deter=torch.zeros(batch_size, self.deter_size, device=device),
        )

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        stoch = state.stoch.reshape(state.stoch.shape[0], -1)
        return torch.cat([stoch, state.deter], dim=-1)

    def calculate_state(self, logits: torch.Tensor, deter: torch.Tensor) -> RSSMState:
        logits = logits.reshape(-1, self.stoch_size, self.stoch_classes)
        stoch = sample_straight_through(logits)
        return RSSMState(logits=logits, stoch=stoch, deter=deter)

    def calculate_prior(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        stoch = prev_state.stoch.reshape(prev_state.stoch.shape[0], -1)
        x = torch.cat([stoch, prev_action], dim=-1)
        x = F.elu(self.img_in(x))
        deter = self.gru(x, prev_state.deter)
        x = F.elu(self.img_hidden(deter))
        return self.calculate_state(self.img_out(x), deter)

    def calculate_posterior(self, prior: RSSMState, embed: torch.Tensor) -> RSSMState:
        x = torch.cat([prior.deter, embed], dim=-1)
        x = F.elu(self.obs_hidden(x))
        return self.calculate_state(self.obs_out(x), prior.deter)

    def obs_step(self, prev_state: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor) -> tuple[RSSMState, RSSMState]:
        prior = self.calculate_prior(prev_state, prev_action)
        post = self.calculate_posterior(prior, embed)
        return post, prior

    def observe(self, embeds: torch.Tensor, actions: torch.Tensor, state: RSSMState | None = None) -> tuple[RSSMState, RSSMState]:
        batch_size, horizon = actions.shape[:2]
        if state is None:
            state = self.init_state(batch_size, actions.device)
        priors, posts = [], []
        prev_state = state
        for t in range(horizon):
            post, prior = self.obs_step(prev_state, actions[:, t], embeds[:, t])
            posts.append(post)
            priors.append(prior)
            prev_state = post
        return stack_states(posts), stack_states(priors)

class DiscreteActor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = MLP(feat_dim, action_dim, hidden_dim=hidden_size, num_layers=4, activation=nn.ELU, norm=False)

    def forward(self, feat: torch.Tensor) -> td.OneHotCategorical:
        return td.OneHotCategorical(logits=self.net(feat))

class DreamerV2Model(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_dim: int, args):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = ConvEncoder(obs_shape[0], depth=args.cnn_depth)
        self.rssm = CategoricalRSSM(
            action_dim=action_dim,
            stoch_size=args.stoch_size,
            stoch_classes=args.stoch_classes,
            deter_size=args.deter_size,
            hidden_size=args.rssm_hidden_size,
            embed_dim=self.encoder.out_dim,
        )
        feat_dim = args.stoch_size * args.stoch_classes + args.deter_size
        self.decoder = ConvDecoder(feat_dim, obs_shape[0], depth=args.cnn_depth)
        self.reward = DenseDecoder(feat_dim, 1, hidden_size=args.num_units, num_layers=2)
        self.value = DenseDecoder(feat_dim, 1, hidden_size=args.num_units, num_layers=3)
        self.actor = DiscreteActor(feat_dim, action_dim, args.num_units)
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
