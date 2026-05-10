"""World Models: Data collection and agent rollout"""
from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import gymnasium as gym
from .model import ASIZE, IMG_SIZE

def collect_data(env_id: str, num_rollouts: int, max_steps: int, data_dir: str, seed: int):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    for i in range(num_rollouts):
        env = gym.make(env_id, render_mode=None)
        obs, _ = env.reset(seed=seed + i)
        frames, actions, rewards, dones = [], [], [], []
        action = np.zeros(ASIZE, dtype=np.float32)
        for _ in range(max_steps):
            frame = cv2.resize(obs, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            action = np.clip(action + 0.1 * np.random.randn(ASIZE).astype(np.float32), -1, 1)
            env_action = action.copy()
            env_action[1] = np.clip((env_action[1] + 1) / 2, 0, 1)
            env_action[2] = np.clip((env_action[2] + 1) / 2, 0, 1)
            obs, reward, terminated, truncated, _ = env.step(env_action)
            actions.append(action.copy())
            rewards.append(reward)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break
        env.close()
        np.savez_compressed(
            os.path.join(data_dir, f"episode_{i:05d}.npz"),
            obs=np.array(frames, dtype=np.uint8),
            action=np.array(actions, dtype=np.float32),
            reward=np.array(rewards, dtype=np.float32),
            done=np.array(dones, dtype=np.bool_),
        )
        if (i + 1) % 100 == 0:
            print(f"  Collected {i+1}/{num_rollouts} rollouts")

def rollout_agent(env_id: str, vae, rnn, controller, device: torch.device, max_steps: int = 1000, render: bool = False) -> float:
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset()
    vae.eval()
    rnn.eval()
    controller.eval()
    hidden = rnn.initial_hidden(1, device)
    total_reward = 0.0
    prev_action = torch.zeros(1, ASIZE, device=device)
    with torch.no_grad():
        for _ in range(max_steps):
            frame = cv2.resize(obs, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            frame_t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            mu, _ = vae.encode(frame_t)
            h = hidden[0].squeeze(0)
            action = controller(mu, h)
            hidden, _ = rnn.forward_single(mu, prev_action, hidden)
            prev_action = action
            env_action = action.squeeze(0).cpu().numpy()
            env_action[1] = np.clip((env_action[1] + 1) / 2, 0, 1)
            env_action[2] = np.clip((env_action[2] + 1) / 2, 0, 1)
            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            if terminated or truncated:
                break
    env.close()
    return total_reward
