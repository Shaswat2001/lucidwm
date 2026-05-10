"""Dreamer V1: Loss functions"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from lucidwm_utils.misc import set_requires_grad
from .model import DreamerModel, RSSMState, flatten_state, detach_state

def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    return obs - 0.5

def preprocess_batch(batch: dict, device: torch.device) -> dict:
    obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
    reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)
    return {"obs": preprocess_obs(obs), "action": action, "reward": reward, "done": done}

def lambda_return(reward: torch.Tensor, value: torch.Tensor, discount: torch.Tensor, bootstrap: torch.Tensor, lambda_: float) -> torch.Tensor:
    horizon = reward.shape[0]
    returns = torch.zeros_like(reward)
    next_value = bootstrap
    for t in reversed(range(horizon)):
        next_value = reward[t] + discount[t] * ((1 - lambda_) * value[t] + lambda_ * next_value)
        returns[t] = next_value
    return returns

def imagine_rollout(model: DreamerModel, start: RSSMState, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

def world_model_loss(model: DreamerModel, batch: dict, args) -> tuple[torch.Tensor, RSSMState, dict]:
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
    image_dist = td.Independent(td.Normal(recon, 1), 3)
    obs_loss = -image_dist.log_prob(target_obs).mean()
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

def behavior_losses(model: DreamerModel, post: RSSMState, args) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if model.cont is not None:
        post = RSSMState(
            mean=post.mean[:, :-1], std=post.std[:, :-1],
            stoch=post.stoch[:, :-1], deter=post.deter[:, :-1],
        )
    start = detach_state(flatten_state(post))
    model_modules = [model.encoder, model.decoder, model.reward, model.rssm]
    if model.cont is not None:
        model_modules.append(model.cont)
    set_requires_grad(model_modules + [model.value], False)
    imag_feat, reward, discount = imagine_rollout(model, start, args)
    value_dist = model.value(imag_feat.reshape(-1, imag_feat.shape[-1]))
    value = value_dist.mean.squeeze(-1).reshape(args.imagine_horizon, -1)
    returns = lambda_return(reward[:-1], value[:-1], discount[:-1], bootstrap=value[-1], lambda_=args.lambda_return)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-2]], dim=0), dim=0
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
