"""LucidWM: Dreamer V1 on DMControl

Paper:     "Dream to Control: Learning Behaviors by Latent Imagination"
           (Hafner et al., ICLR 2020)
Reference: https://github.com/danijar/dreamer

Faithful ingredients from the paper and official implementation:
  - Pixel observations from DeepMind Control Suite (64x64 RGB)
  - Gaussian RSSM with stochastic + deterministic latent state
  - Conv encoder / deconv decoder world model
  - Reconstruction, reward, KL, and optional continuation losses
  - Imagination actor-critic trained with lambda-returns in latent space

Repo adaptation:
  - Single-file PyTorch implementation
  - Reuses LucidWM replay buffer, logger, and evaluation helpers
  - Uses Gymnasium + shimmy wrappers for DMControl compatibility

Example:
  python -m lucidwm.dreamer_dmcontrol --env-id walker-walk --seed 1 --track
"""

import math
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
    parser = argparse.ArgumentParser(description="LucidWM: Dreamer V1 on DMControl")

    parser.add_argument("--env-id", type=str, default="walker-walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--prefill-steps", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")

    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=1_000)

    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-length", type=int, default=50)
    parser.add_argument("--train-every", type=int, default=1_000)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--model-lr", type=float, default=6e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--value-lr", type=float, default=8e-5)
    parser.add_argument("--grad-clip", type=float, default=100.0)

    parser.add_argument("--stoch-size", type=int, default=30)
    parser.add_argument("--deter-size", type=int, default=200)
    parser.add_argument("--rssm-hidden-size", type=int, default=200)
    parser.add_argument("--num-units", type=int, default=400)
    parser.add_argument("--cnn-depth", type=int, default=32)
    parser.add_argument("--free-nats", type=float, default=3.0)
    parser.add_argument("--kl-scale", type=float, default=1.0)
    parser.add_argument("--pcont-scale", type=float, default=10.0)

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--action-init-std", type=float, default=5.0)
    parser.add_argument("--min-std", type=float, default=1e-4)
    parser.add_argument("--mean-scale", type=float, default=5.0)
    parser.add_argument("--expl-amount", type=float, default=0.3)
    parser.add_argument("--eval-noise", type=float, default=0.0)

    parser.add_argument("--learn-cont", action="store_true")
    return parser.parse_args()

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

def set_requires_grad(modules: list[nn.Module], requires_grad: bool):
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(requires_grad)

class DenseDecoder(nn.Module):
    """State decoder for vector-observation Dreamer."""

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
        self.out_dim = out_dim
        self.net = MLP(
            in_dim,
            out_dim,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            activation=nn.ELU,
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)

        if self.distribution == "normal":
            return td.independent.Independent(
                td.Normal(x, 1), 1
            )
        
        if self.distribution == "binary":
            return td.independent.Independent(
                td.Bernoulli(logits=x), 1
            )

        raise NotImplementedError(self.distribution)

class ConvEncoder(nn.Module):
    """Dreamer V1 conv encoder: 64x64x3 -> 32*depth embedding."""

    def __init__(self, depth: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth, 4, stride=2),
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
        x = self.net(obs)
        return x.reshape(obs.shape[0], -1)

class ConvDecoder(nn.Module):
    """Dreamer V1 conv decoder: feature -> image mean."""

    def __init__(self, feat_dim: int, depth: int = 32):
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
            nn.ConvTranspose2d(depth, 3, 6, stride=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).reshape(-1, self.out_channels, 1, 1)
        return self.net(x)
    
class RSSM(nn.Module):
    """Gaussian recurrent state-space model from Dreamer V1."""

    def __init__(
        self,
        action_dim: int,
        stoch_size: int,
        deter_size: int,
        hidden_size: int,
        embed_dim: int,
    ):
        super(RSSM, self).__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size

        self.img_in = nn.Linear(stoch_size+action_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size, deter_size)
        self.img_hidden = nn.Linear(deter_size, hidden_size)
        self.img_out = nn.Linear(hidden_size, 2 * stoch_size)

        self.obs_hidden = nn.Linear(deter_size + embed_dim, hidden_size)
        self.obs_out = nn.Linear(hidden_size, 2 * stoch_size)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        mean = torch.zeros(batch_size, self.stoch_size, device=device)
        std = torch.zeros(batch_size, self.stoch_size, device=device)
        stoch = torch.zeros(batch_size, self.stoch_size, device=device)
        deter = torch.zeros(batch_size, self.deter_size, device=device)
        return RSSMState(mean=mean, std=std, stoch=stoch, deter=deter)
    
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

class TanhNormalActor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        units: int,
        init_std: float,
        min_std: float,
        mean_scale: float,
    ):
        super().__init__()
        self.net = MLP(
            feat_dim,
            2 * action_dim,
            hidden_dim=units,
            num_layers=4,
            activation=nn.ELU,
            norm=False,
        )
        self.raw_init_std = math.log(math.exp(init_std) - 1.0)
        self.min_std = min_std
        self.mean_scale = mean_scale

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(feat)
        mean, std = out.chunk(2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        return mean, std, torch.tanh(mean)

    def sample(
        self,
        feat: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            feat_dim,
            action_dim,
            units=args.num_units,
            init_std=args.action_init_std,
            min_std=args.min_std,
            mean_scale=args.mean_scale,
        )
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
    obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
    reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)
    return {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
    }

def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    horizon = reward.shape[0]
    returns = torch.zeros_like(reward)
    next_value = bootstrap
    for t in reversed(range(horizon)):
        next_value = reward[t] + discount[t] * ((1 - lambda_) * value[t] + lambda_ * next_value)
        returns[t] = next_value
    return returns

def imagine_rollout(
    model: DreamerModel,
    start: RSSMState,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = start
    feats = []
    discounts = []

    for _ in range(args.imagine_horizon):
        feat = model.rssm.get_feat(state)
        action, _ = model.actor.sample(feat.detach(), deterministic=False)
        state = model.rssm.calculate_prior(state, action)
        feat = model.rssm.get_feat(state)
        feats.append(feat)
        if model.cont is not None:
            discounts.append(model.cont(feat).mean.squeeze(-1) * args.discount)
        else:
            discounts.append(torch.full(feat.shape[:1], args.discount, device=feat.device))

    imag_feat = torch.stack(feats, dim=0)
    discount = torch.stack(discounts, dim=0)
    reward = model.reward(imag_feat.reshape(-1, imag_feat.shape[-1])).mean.squeeze(-1)
    reward = reward.reshape(args.imagine_horizon, -1)
    return imag_feat, reward, discount

def world_model_loss(
    model: DreamerModel,
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
    feat = model.rssm.get_feat(post)

    recon = model.decoder(feat.reshape(-1, feat.shape[-1]))
    recon = recon.reshape(target_obs.shape[0], target_obs.shape[1], *target_obs.shape[-3:])

    image_dist = td.Normal(recon, torch.ones_like(recon))
    obs_loss = -image_dist.log_prob(target_obs).sum(dim=(2, 3, 4)).mean()

    reward_dist = model.reward(feat.reshape(-1, feat.shape[-1]))
    reward_loss = -reward_dist.log_prob(reward.reshape(-1, 1)).mean()

    kl = torch.distributions.kl_divergence(model.rssm.get_dist(post), model.rssm.get_dist(prior))
    kl = kl.sum(dim=-1).mean()
    kl_loss = torch.maximum(kl, torch.tensor(args.free_nats, device=kl.device))

    total = obs_loss + reward_loss + args.kl_scale * kl_loss
    cont_loss = torch.tensor(0.0, device=obs.device)
    if model.cont is not None:
        cont_dist = model.cont(feat.reshape(-1, feat.shape[-1]))
        cont_target = args.discount * (1.0 - done)
        cont_loss = -cont_dist.log_prob(cont_target.reshape(-1, 1)).mean()
        total = total + args.pcont_scale * cont_loss

    metrics = {
        "losses/model": total.item(),
        "losses/recon": obs_loss.item(),
        "losses/reward": reward_loss.item(),
        "losses/kl": kl.item(),
        "losses/kl_clamped": kl_loss.item(),
    }
    if model.cont is not None:
        metrics["losses/cont"] = cont_loss.item()
    return total, post, metrics

def behavior_losses(
    model: DreamerModel,
    post: RSSMState,
    args,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if model.cont is not None:
        post = RSSMState(
            mean=post.mean[:, :-1],
            std=post.std[:, :-1],
            stoch=post.stoch[:, :-1],
            deter=post.deter[:, :-1],
        )

    start = detach_state(flatten_state(post))
    model_modules = [model.encoder, model.decoder, model.reward, model.rssm]
    if model.cont is not None:
        model_modules.append(model.cont)

    set_requires_grad(model_modules + [model.value], False)
    imag_feat, reward, discount = imagine_rollout(model, start, args)
    value_dist = model.value(imag_feat.reshape(-1, imag_feat.shape[-1]))
    value = value_dist.mean.squeeze(-1).reshape(args.imagine_horizon, -1)

    returns = lambda_return(
        reward[:-1],
        value[:-1],
        discount[:-1],
        bootstrap=value[-1],
        lambda_=args.lambda_return,
    )
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-2]], dim=0),
        dim=0,
    )

    actor_loss = -(weights * returns).mean()
    set_requires_grad(model_modules + [model.value], True)

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
    }
    return actor_loss, value_loss, metrics

def train_step(
    model: DreamerModel,
    model_opt: optim.Optimizer,
    actor_opt: optim.Optimizer,
    value_opt: optim.Optimizer,
    batch: dict[str, np.ndarray],
    args,
    device: torch.device,
) -> dict[str, float]:
    batch_t = preprocess_batch(batch, device)

    model_opt.zero_grad(set_to_none=True)
    model_loss, post, wm_metrics = world_model_loss(model, batch_t, args)
    model_loss.backward()
    model_grad = torch.nn.utils.clip_grad_norm_(
        list(model.encoder.parameters())
        + list(model.rssm.parameters())
        + list(model.decoder.parameters())
        + list(model.reward.parameters())
        + ([] if model.cont is None else list(model.cont.parameters())),
        args.grad_clip,
    )
    model_opt.step()

    actor_opt.zero_grad(set_to_none=True)
    actor_loss, value_loss, behavior_metrics = behavior_losses(model, post, args)
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), args.grad_clip)
    actor_opt.step()

    value_opt.zero_grad(set_to_none=True)
    _, value_loss, _ = behavior_losses(model, post, args)
    value_loss.backward()
    value_grad = torch.nn.utils.clip_grad_norm_(model.value.parameters(), args.grad_clip)
    value_opt.step()

    metrics = dict(wm_metrics)
    metrics.update(behavior_metrics)
    metrics["grads/model"] = float(model_grad)
    metrics["grads/actor"] = float(actor_grad)
    metrics["grads/value"] = float(value_grad)
    return metrics

@torch.no_grad()
def act(
    model: DreamerModel,
    obs: np.ndarray,
    prev_state: RSSMState | None,
    prev_action: torch.Tensor | None,
    device: torch.device,
    deterministic: bool,
    expl_amount: float,
) -> tuple[np.ndarray, RSSMState, torch.Tensor]:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    embed = model.encoder(obs_t)
    if prev_state is None:
        prev_state = model.rssm.init_state(1, device)
    if prev_action is None:
        prev_action = torch.zeros(1, model.action_dim, device=device)

    post, _ = model.rssm.obs_step(prev_state, prev_action, embed)
    feat = model.rssm.get_feat(post)
    action_t, _ = model.actor.sample(feat, deterministic=deterministic)
    if not deterministic and expl_amount > 0:
        action_t = torch.clamp(action_t + expl_amount * torch.randn_like(action_t), -1.0, 1.0)
    return action_t.squeeze(0).cpu().numpy(), post, action_t

@torch.no_grad()
def evaluate_agent(
    model: DreamerModel,
    env_fn,
    num_episodes: int,
    device: torch.device,
    eval_noise: float,
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
                expl_amount=eval_noise,
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
        obs="rgb",
        img_size=args.img_size,
        action_repeat=args.action_repeat,
        time_limit=args.time_limit,
    )

    def eval_env_fn():
        return make_env(
            args.env_id,
            seed=args.seed + 100,
            obs="rgb",
            img_size=args.img_size,
            action_repeat=args.action_repeat,
            time_limit=args.time_limit,
        )

    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))

    model = DreamerModel(action_dim, args).to(device)
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
        name=f"dreamer_{args.env_id}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    print(
        f"Dreamer V1 DMControl | env={args.env_id} | obs={obs_shape} | act={action_dim} | "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    latent_state = None
    prev_action = None

    for step in range(args.total_steps):
        if step < args.prefill_steps:
            action = env.action_space.sample()
        else:
            action, latent_state, prev_action = act(
                model,
                obs,
                latent_state,
                prev_action,
                device,
                deterministic=False,
                expl_amount=args.expl_amount,
            )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add_step(obs, action.astype(np.float32), float(reward), done)

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
                eval_noise=args.eval_noise,
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
            "args": vars(args),
        },
        f"dreamer_{args.env_id}_s{args.seed}.pt",
    )
    logger.close()
    env.close()
