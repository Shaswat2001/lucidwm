"""LucidWM: DreamerV2 on Atari

Paper:     "Mastering Atari with Discrete World Models" (Hafner et al., 2021)
Reference: https://danijar.com/project/dreamerv2/

Key DreamerV2 changes relative to Dreamer V1:
  - Categorical latent states with straight-through gradients
  - KL balancing instead of free nats
  - Discrete-action actor trained with reinforce-style gradients
  - Policy entropy regularization for exploration
  - Larger model defaults for Atari

This file keeps the single-file LucidWM style while adapting the core
algorithmic pieces to Atari.
"""

from __future__ import annotations

import argparse
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as td

from lucidwm_components.networks import MLP
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.envs import make_env
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: DreamerV2 on Atari")

    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--prefill-steps", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")

    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-length", type=int, default=50)
    parser.add_argument("--train-every", type=int, default=1_000)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--model-lr", type=float, default=2e-4)
    parser.add_argument("--actor-lr", type=float, default=4e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    parser.add_argument("--stoch-size", type=int, default=32)
    parser.add_argument("--stoch-classes", type=int, default=32)
    parser.add_argument("--deter-size", type=int, default=600)
    parser.add_argument("--rssm-hidden-size", type=int, default=600)
    parser.add_argument("--num-units", type=int, default=600)
    parser.add_argument("--cnn-depth", type=int, default=48)

    parser.add_argument("--discount", type=float, default=0.999)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--kl-scale", type=float, default=1.0)
    parser.add_argument("--kl-balance", type=float, default=0.8)
    parser.add_argument("--pcont-scale", type=float, default=10.0)
    parser.add_argument("--actor-entropy-scale", type=float, default=1e-3)
    parser.add_argument("--explore-entropy-scale", type=float, default=3e-4)
    parser.add_argument("--learn-cont", action="store_true")
    return parser.parse_args()

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

def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    return obs - 0.5

class DenseDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        distribution: str = "normal",
    ):
        super().__init__()
        self.distribution = distribution
        self.net = MLP(
            in_dim,
            out_dim,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            activation=nn.ELU,
            norm=False,
        )

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
            nn.Conv2d(in_channels, depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(depth, 2 * depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride=2),
            nn.ReLU(),
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
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * depth, depth, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, out_channels, 6, stride=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).reshape(-1, self.out_channels, 1, 1)
        return self.net(x)

def sample_straight_through(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    sample = td.OneHotCategorical(logits=logits).sample()
    return sample + probs - probs.detach()

def categorical_kl(post_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    post = td.OneHotCategorical(logits=post_logits)
    prior = td.OneHotCategorical(logits=prior_logits)
    return td.kl_divergence(post, prior).sum(dim=-1)

class CategoricalRSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        stoch_size: int,
        stoch_classes: int,
        deter_size: int,
        hidden_size: int,
        embed_dim: int,
    ):
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
        logits = torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device)
        stoch = torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device)
        deter = torch.zeros(batch_size, self.deter_size, device=device)
        return RSSMState(logits=logits, stoch=stoch, deter=deter)

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
        logits = self.img_out(x)
        return self.calculate_state(logits, deter)

    def calculate_posterior(self, prior: RSSMState, embed: torch.Tensor) -> RSSMState:
        x = torch.cat([prior.deter, embed], dim=-1)
        x = F.elu(self.obs_hidden(x))
        logits = self.obs_out(x)
        return self.calculate_state(logits, prior.deter)

    def obs_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.calculate_prior(prev_state, prev_action)
        post = self.calculate_posterior(prior, embed)
        return post, prior

    def observe(
        self,
        embeds: torch.Tensor,
        actions: torch.Tensor,
        state: RSSMState | None = None,
    ) -> tuple[RSSMState, RSSMState]:
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

class DiscreteActor(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.net = MLP(
            feat_dim,
            action_dim,
            hidden_dim=hidden_size,
            num_layers=4,
            activation=nn.ELU,
            norm=False,
        )

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
            DenseDecoder(
                feat_dim,
                1,
                hidden_size=args.num_units,
                num_layers=3,
                distribution="binary",
            )
            if args.learn_cont
            else None
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear | nn.Conv2d | nn.ConvTranspose2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def preprocess_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    obs = torch.tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=device)
    action = torch.tensor(np.asarray(batch["action"]), dtype=torch.float32, device=device)
    reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)
    return {"obs": preprocess_obs(obs), "action": action, "reward": reward, "done": done}

def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    returns = torch.zeros_like(reward)
    next_value = bootstrap
    for t in reversed(range(reward.shape[0])):
        next_value = reward[t] + discount[t] * ((1 - lambda_) * value[t] + lambda_ * next_value)
        returns[t] = next_value
    return returns


def world_model_loss(
    model: DreamerV2Model,
    batch: dict[str, torch.Tensor],
    args,
) -> tuple[torch.Tensor, RSSMState, dict[str, float]]:
    obs = batch["obs"]
    action = batch["action"][:, :-1]
    reward = batch["reward"][:, :-1]
    done = batch["done"][:, :-1]
    target_obs = obs[:, 1:]

    embed = model.encoder(target_obs.reshape(-1, *target_obs.shape[-3:]))
    embed = embed.reshape(target_obs.shape[0], target_obs.shape[1], -1)
    post, prior = model.rssm.observe(embed, action)
    feat = model.rssm.get_feat(flatten_state(post))
    feat = feat.reshape(target_obs.shape[0], target_obs.shape[1], -1)

    recon = model.decoder(feat.reshape(-1, feat.shape[-1]))
    recon = recon.reshape(target_obs.shape[0], target_obs.shape[1], *target_obs.shape[-3:])
    image_dist = td.Normal(recon, torch.ones_like(recon))
    obs_loss = -image_dist.log_prob(target_obs).sum(dim=(2, 3, 4)).mean()

    reward_dist = model.reward(feat.reshape(-1, feat.shape[-1]))
    reward_loss = -reward_dist.log_prob(reward.reshape(-1, 1)).mean()

    if model.cont is not None:
        cont_dist = model.cont(feat.reshape(-1, feat.shape[-1]))
        cont_target = args.discount * (1.0 - done)
        cont_loss = -cont_dist.log_prob(cont_target.reshape(-1, 1)).mean()
    else:
        cont_loss = torch.tensor(0.0, device=obs.device)

    prior_logits = prior.logits
    post_logits = post.logits
    kl_lhs = categorical_kl(post_logits.detach(), prior_logits).mean()
    kl_rhs = categorical_kl(post_logits, prior_logits.detach()).mean()
    kl_loss = args.kl_scale * (args.kl_balance * kl_lhs + (1 - args.kl_balance) * kl_rhs)

    total = obs_loss + reward_loss + kl_loss
    if model.cont is not None:
        total = total + args.pcont_scale * cont_loss

    metrics = {
        "losses/model": total.item(),
        "losses/recon": obs_loss.item(),
        "losses/reward": reward_loss.item(),
        "losses/kl": (kl_lhs + kl_rhs).item() * 0.5,
        "losses/kl_lhs": kl_lhs.item(),
        "losses/kl_rhs": kl_rhs.item(),
    }
    if model.cont is not None:
        metrics["losses/cont"] = cont_loss.item()
    return total, post, metrics

def imagine_rollout(
    model: DreamerV2Model,
    start: RSSMState,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = start
    feats = []
    rewards = []
    discounts = []
    log_probs = []
    entropies = []

    for _ in range(args.imagine_horizon):
        feat = model.rssm.get_feat(state)
        action_dist = model.actor(feat.detach())
        action_idx = action_dist.sample()
        action = action_idx + action_dist.probs - action_dist.probs.detach()
        log_prob = action_dist.log_prob(action_idx)
        entropy = action_dist.entropy()
        state = model.rssm.calculate_prior(state, action)
        feat = model.rssm.get_feat(state)

        feats.append(feat)
        rewards.append(model.reward(feat).mean.squeeze(-1))
        log_probs.append(log_prob)
        entropies.append(entropy)
        if model.cont is not None:
            discounts.append(model.cont(feat).mean.squeeze(-1) * args.discount)
        else:
            discounts.append(torch.full(feat.shape[:1], args.discount, device=feat.device))

    return (
        torch.stack(feats, dim=0),
        torch.stack(rewards, dim=0),
        torch.stack(discounts, dim=0),
        torch.stack(log_probs, dim=0),
        torch.stack(entropies, dim=0),
    )


def behavior_losses(
    model: DreamerV2Model,
    post: RSSMState,
    args,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if model.cont is not None:
        post = RSSMState(
            logits=post.logits[:, :-1],
            stoch=post.stoch[:, :-1],
            deter=post.deter[:, :-1],
        )

    start = detach_state(flatten_state(post))
    model_modules = [model.encoder, model.decoder, model.reward, model.rssm]
    if model.cont is not None:
        model_modules.append(model.cont)

    set_requires_grad(model_modules + [model.value], False)
    imag_feat, reward, discount, log_prob, entropy = imagine_rollout(model, start, args)
    value = model.value(imag_feat.reshape(-1, imag_feat.shape[-1])).mean.squeeze(-1)
    value = value.reshape(args.imagine_horizon, -1)
    returns = lambda_return(
        reward[:-1],
        value[:-1],
        discount[:-1],
        bootstrap=value[-1],
        lambda_=args.lambda_return,
    )
    set_requires_grad(model_modules + [model.value], True)

    baseline = value[:-1].detach()
    advantage = returns.detach() - baseline
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-2]], dim=0),
        dim=0,
    )
    actor_objective = log_prob[:-1] * advantage + args.actor_entropy_scale * entropy[:-1]
    actor_loss = -(weights.detach() * actor_objective).mean()

    with torch.no_grad():
        value_feat = imag_feat[:-1].detach()
        value_target = returns.detach()
        value_weight = weights.detach()

    value_pred = model.value(value_feat.reshape(-1, value_feat.shape[-1]))
    value_loss = -value_pred.log_prob(value_target.reshape(-1, 1))
    value_loss = value_loss.reshape(args.imagine_horizon - 1, -1)
    value_loss = (value_weight * value_loss).mean()

    metrics = {
        "losses/actor": actor_loss.item(),
        "losses/value": value_loss.item(),
        "algo/imagined_reward": reward.mean().item(),
        "algo/imagined_value": value.mean().item(),
        "algo/policy_entropy": entropy.mean().item(),
    }
    return actor_loss, value_loss, metrics


def train_step(
    model: DreamerV2Model,
    model_opt: optim.Optimizer,
    actor_opt: optim.Optimizer,
    value_opt: optim.Optimizer,
    batch: dict[str, np.ndarray],
    args,
    device: torch.device,
) -> dict[str, float]:
    batch_t = preprocess_batch(batch, device)

    model_opt.zero_grad(set_to_none=True)
    actor_opt.zero_grad(set_to_none=True)
    value_opt.zero_grad(set_to_none=True)

    model_loss, post, wm_metrics = world_model_loss(model, batch_t, args)
    actor_loss, value_loss, behavior_metrics = behavior_losses(model, post, args)

    model_loss.backward()
    actor_loss.backward()
    value_loss.backward()

    model_grad = torch.nn.utils.clip_grad_norm_(
        list(model.encoder.parameters())
        + list(model.rssm.parameters())
        + list(model.decoder.parameters())
        + list(model.reward.parameters())
        + ([] if model.cont is None else list(model.cont.parameters())),
        args.grad_clip,
    )
    actor_grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), args.grad_clip)
    value_grad = torch.nn.utils.clip_grad_norm_(model.value.parameters(), args.grad_clip)

    model_opt.step()
    actor_opt.step()
    value_opt.step()

    metrics = dict(wm_metrics)
    metrics.update(behavior_metrics)
    metrics["grads/model"] = float(model_grad)
    metrics["grads/actor"] = float(actor_grad)
    metrics["grads/value"] = float(value_grad)
    return metrics


@torch.no_grad()
def act(
    model: DreamerV2Model,
    obs: np.ndarray,
    prev_state: RSSMState | None,
    prev_action: torch.Tensor | None,
    device: torch.device,
    deterministic: bool,
    entropy_scale: float,
) -> tuple[int, RSSMState, torch.Tensor]:
    obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
    embed = model.encoder(preprocess_obs(obs_t))
    if prev_state is None:
        prev_state = model.rssm.init_state(1, device)
    if prev_action is None:
        prev_action = torch.zeros(1, model.action_dim, device=device)

    post, _ = model.rssm.obs_step(prev_state, prev_action, embed)
    feat = model.rssm.get_feat(post)
    action_dist = model.actor(feat)
    if deterministic:
        action_idx = action_dist.probs.argmax(dim=-1)
    else:
        action_idx = action_dist.sample()
    action = F.one_hot(action_idx, model.action_dim).float()
    return int(action_idx.item()), post, action


@torch.no_grad()
def evaluate_agent(
    model: DreamerV2Model,
    env_fn,
    num_episodes: int,
    device: torch.device,
) -> dict[str, float]:
    returns = []
    lengths = []
    for _ in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        latent_state = None
        prev_action = None
        ep_return = 0.0
        ep_length = 0
        while not done:
            action, latent_state, prev_action = act(
                model,
                obs,
                latent_state,
                prev_action,
                device,
                deterministic=True,
                entropy_scale=0.0,
            )
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
        "returns": returns,
    }


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    env = make_env(
        args.env_id,
        seed=args.seed,
        img_size=args.img_size,
        frame_stack=args.frame_stack,
    )

    def eval_env_fn():
        return make_env(
            args.env_id,
            seed=args.seed + 100,
            img_size=args.img_size,
            frame_stack=args.frame_stack,
        )

    obs_shape = tuple(env.observation_space.shape)
    action_dim = int(env.action_space.n)

    model = DreamerV2Model(obs_shape, action_dim, args).to(device)
    model_opt = optim.Adam(
        list(model.encoder.parameters())
        + list(model.rssm.parameters())
        + list(model.decoder.parameters())
        + list(model.reward.parameters())
        + ([] if model.cont is None else list(model.cont.parameters())),
        lr=args.model_lr,
    )
    actor_opt = optim.Adam(model.actor.parameters(), lr=args.actor_lr)
    value_opt = optim.Adam(model.value.parameters(), lr=args.value_lr)

    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        action_dim=action_dim,
        seq_len=args.batch_length + 1,
    )

    logger = Logger(
        project=args.wandb_project,
        name=f"dreamerv2_{args.env_id.replace('/', '_')}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    print(
        f"DreamerV2 Atari | env={args.env_id} | obs={obs_shape} | act={action_dim} | "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    latent_state = None
    prev_action = None

    for step in range(args.total_steps):
        if step < args.prefill_steps:
            env_action = int(env.action_space.sample())
            action_vec = F.one_hot(torch.tensor(env_action), action_dim).float().cpu().numpy()
        else:
            env_action, latent_state, prev_action = act(
                model,
                obs,
                latent_state,
                prev_action,
                device,
                deterministic=False,
                entropy_scale=args.explore_entropy_scale,
            )
            action_vec = prev_action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        buffer.add_step(obs, action_vec.astype(np.float32), float(reward), done)

        episode_return += reward
        episode_length += 1
        obs = next_obs

        if done:
            logger.log(
                {
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                },
                step=step,
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            latent_state = None
            prev_action = None

        if step >= args.prefill_steps and buffer.num_episodes >= 1 and step % args.train_every == 0:
            metrics = None
            for _ in range(args.train_steps):
                try:
                    batch = buffer.sample(args.batch_size, seq_len=args.batch_length + 1)
                except ValueError:
                    break
                metrics = train_step(model, model_opt, actor_opt, value_opt, batch, args, device)
            if metrics is not None:
                logger.log(metrics, step=step)

        if step > 0 and step % args.eval_freq == 0:
            model.eval()
            eval_result = evaluate_agent(
                model,
                eval_env_fn,
                num_episodes=args.eval_episodes,
                device=device,
            )
            print(
                f"Step {step:>7d} | eval={eval_result['mean_return']:.1f} "
                f"+/- {eval_result['std_return']:.1f}"
            )
            logger.log(
                {
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                },
                step=step,
            )
            model.train()

    torch.save(
        {
            "model": model.state_dict(),
            "action_dim": action_dim,
            "obs_shape": obs_shape,
            "args": vars(args),
        },
        f"dreamerv2_{args.env_id.replace('/', '_')}_s{args.seed}.pt",
    )
    logger.close()
    env.close()
