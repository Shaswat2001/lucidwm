"""DINO-WM: PushT trajectory dataset"""
from __future__ import annotations
import pickle
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, n_rollout: int | None = None, normalize_action: bool = True, relative: bool = True, action_scale: float = 100.0, with_velocity: bool = True, img_size: int = 224):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.pusht_action_mean = torch.tensor([-0.0087, 0.0068])
        self.pusht_action_std = torch.tensor([0.2019, 0.2002])
        self.states = torch.load(self.data_path / "states.pth", weights_only=True).float()
        action_file = "rel_actions.pth" if relative else "abs_actions.pth"
        self.actions = torch.load(self.data_path / action_file, weights_only=True).float() / action_scale
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)
        n = n_rollout if n_rollout else len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states[..., :2].clone()
        if with_velocity:
            vel_path = self.data_path / "velocities.pth"
            if vel_path.exists():
                velocities = torch.load(vel_path, weights_only=True)[:n].float()
                self.states = torch.cat([self.states, velocities], dim=-1)
                self.proprios = torch.cat([self.proprios, velocities], dim=-1)
        if normalize_action:
            self.actions = (self.actions - self.pusht_action_mean) / self.pusht_action_std
        self.action_dim = self.actions.shape[-1]
        print(f"Loaded {n} rollouts (action_dim={self.action_dim})")

    def get_seq_length(self, idx: int) -> int:
        return self.seq_lengths[idx]

    def get_frames(self, idx: int, frames):
        try:
            from decord import VideoReader
            import decord
            decord.bridge.set_bridge("torch")
        except ImportError:
            raise ImportError("decord required: pip install decord")
        vid_path = self.data_path / "obses" / f"episode_{idx:03d}.mp4"
        reader = VideoReader(str(vid_path), num_threads=1)
        image = reader.get_batch(list(frames)).float() / 255.0
        image = image.permute(0, 3, 1, 2)
        if image.shape[-1] != self.img_size:
            image = F.interpolate(image, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        obs = {"visual": image, "proprio": self.proprios[idx, list(frames)]}
        return obs, self.actions[idx, list(frames)], self.states[idx, list(frames)]

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        T = self.get_seq_length(idx)
        return self.get_frames(idx, range(T))

class TrajSlicerDataset(Dataset):
    def __init__(self, traj_dataset: TrajectoryDataset, num_frames: int, frameskip: int = 0):
        self.traj_dataset = traj_dataset
        self.num_frames = num_frames
        self.effective_skip = frameskip + 1
        self.slices = []
        for ep_idx in range(len(traj_dataset)):
            T = traj_dataset.get_seq_length(ep_idx)
            window_size = num_frames * self.effective_skip
            for start in range(T - window_size + 1):
                self.slices.append((ep_idx, start))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        ep_idx, start = self.slices[idx]
        frames = list(range(start, start + self.num_frames * self.effective_skip, self.effective_skip))
        obs, actions, states = self.traj_dataset.get_frames(ep_idx, frames)
        return obs, actions, states
