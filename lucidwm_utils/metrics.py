"""Evaluation and metrics for world model algorithms.

Standardized evaluation protocol shared across all implementations.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Callable

def evaluate(
    env_fn: Callable[[], gym.Env],
    agent_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int = 10,
    record_video: bool = False,
) -> dict:
    """Run evaluation episodes with a trained agent.

    Args:
        env_fn: Callable that creates a fresh environment instance.
        agent_fn: Callable mapping observation -> action (deterministic).
                  obs: np.ndarray of shape obs_shape
                  returns: np.ndarray of shape (action_dim,)
        num_episodes: Number of evaluation episodes.
        record_video: Whether to record frames for video logging.

    Returns:
        dict with:
            mean_return: float
            std_return: float
            mean_length: float
            returns: list[float] -- per-episode returns
            video_frames: list[np.ndarray] (only if record_video, first episode only)
    """
    returns = []
    lengths = []
    video_frames = []

    for ep in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0

        while not done:
            action = agent_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated

            if record_video and ep == 0:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

        returns.append(ep_return)
        lengths.append(ep_length)
        env.close()

    result = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
    }
    if record_video:
        result["video_frames"] = video_frames

    return result
