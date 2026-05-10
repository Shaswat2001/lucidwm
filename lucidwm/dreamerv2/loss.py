"""Dreamer V2: Loss functions"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td
from .model import DreamerV2Model, RSSMState, flatten_state, detach_state, categorical_kl, set_requires_grad

def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    return obs - 0.5

def preprocess_batch(batch: dict, device: torch.device) -> dict:
    obs = torch.tensor(np.asarray(batch["obs"]), dtype=torch.float32, device=device)
    action = torch.tensor(np.asarray(batch["action"]), dtype=torch.float32, device=device)
    reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)
    return {"obs": preprocess_obs(obs), "action": action, "reward": reward, "done": done}

def lambda_return(reward: torch.Tensor, value: torch.Tensor, discount: torch.Tensor, bootstrap: torch.Tensor, lambda_: float) -> torch.Tensor:
    returns = torch.zeros_like(reward)
    next_value = bootstrap
    for t in reversed(range(reward.shape[0])):
        next_value = reward[t] + discount[t] * ((1 - lambda_) * value[t] + lambda_ * next_value)
        returns[t] = next_value
    return returns

def imagine_rollout(model: DreamerV2Model, start: RSSMState, args):
    state = start
    feats, rewards, discounts, log_probs, entropies = [], [], [], [], []
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

def world_model_loss(model: DreamerV2Model, batch: dict, args) -> tuple[torch.Tensor, RSSMState, dict]:
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
    kl_lhs = categorical_kl(post.logits.detach(), prior.logits).mean()
    kl_rhs = categorical_kl(post.logits, prior.logits.detach()).mean()
    kl_loss = args.kl_scale * (args.kl_balance * kl_lhs + (1 - args.kl_balance) * kl_rhs)
    total = obs_loss + reward_loss + kl_loss
    if model.cont is not None:
        total = total + args.pcont_scale * cont_loss
    metrics = {
        "losses/model": total.item(),
        "losses/recon": obs_loss.item(),
        "losses/reward": reward_loss.item(),
        "losses/kl": (kl_lhs + kl_rhs).item() * 0.5,
    }
    if model.cont is not None:
        metrics["losses/cont"] = cont_loss.item()
    return total, post, metrics

def behavior_losses(model: DreamerV2Model, post: RSSMState, args) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if model.cont is not None:
        post = RSSMState(logits=post.logits[:, :-1], stoch=post.stoch[:, :-1], deter=post.deter[:, :-1])
    start = detach_state(flatten_state(post))
    model_modules = [model.encoder, model.decoder, model.reward, model.rssm]
    if model.cont is not None:
        model_modules.append(model.cont)
    set_requires_grad(model_modules + [model.value], False)
    imag_feat, reward, discount, log_prob, entropy = imagine_rollout(model, start, args)
    value = model.value(imag_feat.reshape(-1, imag_feat.shape[-1])).mean.squeeze(-1)
    value = value.reshape(args.imagine_horizon, -1)
    returns = lambda_return(reward[:-1], value[:-1], discount[:-1], bootstrap=value[-1], lambda_=args.lambda_return)
    set_requires_grad(model_modules + [model.value], True)
    baseline = value[:-1].detach()
    advantage = returns.detach() - baseline
    weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-2]], dim=0), dim=0)
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
