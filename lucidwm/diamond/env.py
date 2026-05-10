"""DIAMOND: Environment utilities"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
from lucidwm_utils.buffers import ReplayBuffer
from .model import ActorCritic, one_hot_action

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
) -> list[tuple[float, int]]:
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
        buffer.add_step(obs, action_one_hot, clip_reward(reward), done)
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
