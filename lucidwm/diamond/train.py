"""DIAMOND: Training loss functions"""
from __future__ import annotations
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from .model import DiffusionWorldModel, RewardTerminationModel, ActorCritic, reward_class, sigma_schedule, edm_preconditioning

def sample_log_sigma(batch_size: int, p_mean: float, p_std: float, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    log_sigma = torch.randn(batch_size, device=device) * p_std + p_mean
    return log_sigma.exp().clamp(sigma_min, sigma_max)

def sample_next_obs(model: DiffusionWorldModel, context_obs: torch.Tensor, context_actions: torch.Tensor, args) -> torch.Tensor:
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

def diffusion_loss(model: DiffusionWorldModel, batch: dict[str, torch.Tensor], args) -> torch.Tensor:
    context_obs = batch["obs"][:, :args.context_len]
    context_actions = batch["action"][:, :args.context_len]
    target_obs = batch["obs"][:, args.context_len]
    sigma = sample_log_sigma(target_obs.shape[0], args.p_mean, args.p_std, args.sigma_min, args.sigma_max, target_obs.device)
    noisy_next = target_obs + torch.randn_like(target_obs) * sigma[:, None, None, None]
    pred = model(noisy_next, context_obs, context_actions, sigma)
    return F.mse_loss(pred, target_obs)

def reward_termination_loss(model: RewardTerminationModel, batch: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, float]]:
    obs = batch["obs"][:, :args.burnin_len + args.imagine_horizon]
    actions = batch["action"][:, :args.burnin_len + args.imagine_horizon]
    rewards = batch["reward"][:, :args.burnin_len + args.imagine_horizon]
    done = batch["done"][:, :args.burnin_len + args.imagine_horizon]
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
    obs_hist = obs[:, :args.context_len].clone()
    act_hist = actions[:, :args.context_len - 1].clone() if args.context_len > 1 else torch.zeros(batch_size, 0, actions.shape[-1], device=device)
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
    return policy_loss, value_loss, {
        "losses/policy": policy_loss.item(),
        "losses/value": value_loss.item(),
        "algo/imagined_reward": reward_t.mean().item(),
        "algo/imagined_value": value_t.mean().item(),
    }

def tensor_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.tensor(batch["obs"], dtype=torch.float32, device=device),
        "action": torch.tensor(batch["action"], dtype=torch.float32, device=device),
        "reward": torch.tensor(batch["reward"], dtype=torch.float32, device=device),
        "done": torch.tensor(batch["done"], dtype=torch.float32, device=device),
    }
