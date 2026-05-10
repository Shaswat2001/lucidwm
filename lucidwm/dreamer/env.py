"""Dreamer V1: Environment helpers"""
from __future__ import annotations
import numpy as np
import torch
from lucidwm_utils.envs import make_env
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.logger import Logger
from .model import DreamerModel, RSSMState
from .loss import preprocess_obs

def make_env_fn(args, seed_offset: int = 0):
    def thunk():
        return make_env(
            args.env_id,
            seed=args.seed + seed_offset,
            obs="rgb",
            img_size=args.img_size,
            action_repeat=args.action_repeat,
            time_limit=args.time_limit,
        )
    return thunk

@torch.no_grad()
def act(model: DreamerModel, obs: np.ndarray, prev_state: RSSMState | None, prev_action: torch.Tensor | None, device: torch.device, deterministic: bool, expl_amount: float) -> tuple[np.ndarray, RSSMState, torch.Tensor]:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    embed = model.encoder(preprocess_obs(obs_t))
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
def evaluate_agent(model: DreamerModel, env_fn, num_episodes: int, device: torch.device, eval_noise: float) -> dict:
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
                model, obs, latent_state, prev_action, device, deterministic=True, expl_amount=eval_noise,
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
    }

def reset_env_batch(envs: list, base_seed: int | None = None) -> np.ndarray:
    observations = []
    for idx, env in enumerate(envs):
        seed = None if base_seed is None else base_seed + idx
        obs, _ = env.reset(seed=seed)
        observations.append(obs)
    return np.stack(observations, axis=0)

def step_env_batch(envs: list, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    next_obs, rewards, dones = [], [], []
    for idx, env in enumerate(envs):
        obs, reward, terminated, truncated, _ = env.step(actions[idx])
        done = terminated or truncated
        if done:
            obs, _ = env.reset()
        next_obs.append(obs)
        rewards.append(reward)
        dones.append(done)
    return (
        np.stack(next_obs, axis=0),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(dones, dtype=np.bool_),
    )

def collect_batched_steps(
    envs, observations, model, buffer, args, device, total_env_steps,
    latent_states, prev_actions, episode_storage, episode_returns, episode_lengths, logger,
):
    if total_env_steps < args.prefill_steps:
        actions = np.stack([env.action_space.sample() for env in envs], axis=0)
    else:
        action_list = []
        for idx in range(args.num_envs):
            action, latent_states[idx], prev_actions[idx] = act(
                model, observations[idx], latent_states[idx], prev_actions[idx],
                device, deterministic=False, expl_amount=args.expl_amount,
            )
            action_list.append(action.astype(np.float32))
        actions = np.stack(action_list, axis=0)
    next_obs, rewards, done = step_env_batch(envs, actions)
    for idx in range(args.num_envs):
        episode_storage[idx]["obs"].append(observations[idx])
        episode_storage[idx]["action"].append(actions[idx].astype(np.float32))
        episode_storage[idx]["reward"].append(float(rewards[idx]))
        episode_storage[idx]["done"].append(bool(done[idx]))
        episode_returns[idx] += float(rewards[idx])
        episode_lengths[idx] += 1
        if done[idx]:
            buffer.add_episode(
                np.array(episode_storage[idx]["obs"]),
                np.array(episode_storage[idx]["action"]),
                np.array(episode_storage[idx]["reward"], dtype=np.float32),
                np.array(episode_storage[idx]["done"], dtype=np.float32),
            )
            logger.log(
                {"charts/episodic_return": float(episode_returns[idx]),
                 "charts/episodic_length": float(episode_lengths[idx])},
                step=total_env_steps + idx,
            )
            episode_storage[idx] = {"obs": [], "action": [], "reward": [], "done": []}
            episode_returns[idx] = 0.0
            episode_lengths[idx] = 0
            latent_states[idx] = None
            prev_actions[idx] = None
    return envs, next_obs, latent_states, prev_actions, episode_storage, episode_returns, episode_lengths
