"""World Models: Frame and sequence datasets for VAE/RNN training"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, data_dir: str):
        frames = []
        for ep_file in sorted(Path(data_dir).glob("episode_*.npz")):
            frames.append(np.load(ep_file)["obs"])
        self.frames = np.concatenate(frames, axis=0)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        f = self.frames[idx].astype(np.float32) / 255.0
        return torch.from_numpy(f.transpose(2, 0, 1))

class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, vae, device: torch.device, seq_len: int):
        self.seq_len = seq_len
        self.episodes = []
        vae.eval()
        with torch.no_grad():
            for ep_file in sorted(Path(data_dir).glob("episode_*.npz")):
                data = np.load(ep_file)
                obs = torch.from_numpy(
                    data["obs"].astype(np.float32) / 255.0
                ).permute(0, 3, 1, 2).to(device)
                zs = []
                for i in range(0, len(obs), 256):
                    mu, _ = vae.encode(obs[i:i+256])
                    zs.append(mu.cpu().numpy())
                self.episodes.append({
                    "z": np.concatenate(zs, axis=0),
                    "action": data["action"].astype(np.float32),
                    "reward": data["reward"].astype(np.float32),
                    "done": data["done"].astype(np.float32),
                })
        self.index = []
        for ep_idx, ep in enumerate(self.episodes):
            for start in range(len(ep["z"]) - seq_len - 1):
                self.index.append((ep_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        s = slice(start, start + self.seq_len)
        s1 = slice(start + 1, start + self.seq_len + 1)
        return {
            "z": torch.from_numpy(ep["z"][s]),
            "action": torch.from_numpy(ep["action"][s]),
            "z_next": torch.from_numpy(ep["z"][s1]),
            "reward": torch.from_numpy(ep["reward"][s]),
            "done": torch.from_numpy(ep["done"][s]),
        }
