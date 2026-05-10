"""TD-MPC2: Model components and offline dataset"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from lucidwm_components.networks import MLP
from lucidwm_components.distribution import SimNorm, TwoHotDist

class TDMPC2Model(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            MLP(obs_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp),
        )
        self.dynamics = nn.Sequential(
            MLP(args.latent_dim + action_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp),
        )
        self.rewards = MLP(args.latent_dim + action_dim, args.num_bins, args.hidden_dim, activation=nn.Mish)
        self.termination = MLP(args.latent_dim + action_dim, 1, args.hidden_dim, activation=nn.Mish)
        self.policy_prior = MLP(args.latent_dim, 2 * action_dim, args.hidden_dim, activation=nn.Mish)
        self.q = nn.ModuleList([
            MLP(args.latent_dim + action_dim, args.num_bins, args.hidden_dim, activation=nn.Mish)
            for _ in range(args.num_q)
        ])
        self.two_hot = TwoHotDist(num_bins=args.num_bins)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_state(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.rewards(torch.cat([z, a], dim=-1))

    def cont(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.termination(torch.cat([z, a], dim=-1))

    def q_values(self, z: torch.Tensor, a: torch.Tensor) -> list[torch.Tensor]:
        za = torch.cat([z, a], dim=-1)
        return [head(za) for head in self.q]

    def policy(self, z: torch.Tensor, deterministic: bool = False):
        out = self.policy_prior(z)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        if deterministic:
            return torch.tanh(mean), torch.zeros(z.shape[0], device=z.device)
        noise = torch.randn_like(mean)
        raw_action = mean + std * noise
        action = torch.tanh(raw_action)
        log_prob = (-0.5 * noise.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)
        return action, log_prob

class OfflineDataset:
    def __init__(self, dataset: str = "mt30", data_dir: str = "data/tdmpc2", task_filter: str | None = None):
        self.data_dir = Path(data_dir) / dataset
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.task_filter = task_filter
        self.download_chunks()
        self.obs_data = []
        self.action_data = []
        self.reward_data = []
        self.task_data = []
        self.done_data = []
        self.load_chunks()

    def download_chunks(self):
        try:
            from huggingface_hub import hf_hub_download, list_repo_tree
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")
        repo_id = "nicklashansen/tdmpc2"
        print(f"Checking HuggingFace repo for {self.dataset} dataset...")
        try:
            files = list_repo_tree(repo_id, path_in_repo=self.dataset, repo_type="dataset")
            chunk_files = [f.rfilename for f in files if f.rfilename.endswith(".pt")]
        except Exception:
            chunk_files = [f"{self.dataset}/chunk_{i}.pt" for i in range(20)]
        for chunk_file in chunk_files:
            local_path = self.data_dir / chunk_file.split("/")[-1]
            if not local_path.exists():
                print(f"  Downloading {chunk_file}...")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=chunk_file,
                        repo_type="dataset",
                        local_dir=str(self.data_dir.parent.parent),
                    )
                except Exception as e:
                    print(f"  Warning: could not download {chunk_file}: {e}")
            else:
                print(f"  Found cached {chunk_file}")

    def load_chunks(self):
        chunk_files = sorted(self.data_dir.glob("chunk_*.pt"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.data_dir}.")
        print(f"Loading {len(chunk_files)} chunks from {self.data_dir}...")
        for chunk_path in chunk_files:
            print(f"  Loading {chunk_path.name}...")
            try:
                chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
                if hasattr(chunk, "get") or isinstance(chunk, dict):
                    obs = chunk.get("obs", chunk.get("observation", None))
                    act = chunk.get("action", None)
                    rew = chunk.get("reward", None)
                    done = chunk.get("done", chunk.get("terminated", chunk.get("termination", None)))
                    task = chunk.get("task", None)
                else:
                    print(f"  Warning: unknown format in {chunk_path.name}, skipping")
                    continue
                if obs is None or act is None:
                    continue
                for name, val in [("obs", obs), ("act", act), ("rew", rew), ("done", done), ("task", task)]:
                    if isinstance(val, torch.Tensor):
                        val = val.numpy()
                    if name == "obs": self.obs_data.append(val)
                    elif name == "act": self.action_data.append(val)
                    elif name == "rew" and val is not None: self.reward_data.append(val)
                    elif name == "done" and val is not None: self.done_data.append(val.astype(np.float32))
                    elif name == "task" and val is not None: self.task_data.append(val)
            except Exception as e:
                print(f"  Warning: failed to load {chunk_path.name}: {e}")
        if not self.obs_data:
            raise RuntimeError("No data loaded from chunks")
        self.obs_all = np.concatenate(self.obs_data, axis=0)
        self.action_all = np.concatenate(self.action_data, axis=0)
        self.reward_all = np.concatenate(self.reward_data, axis=0) if self.reward_data else None
        self.done_all = np.concatenate(self.done_data, axis=0) if self.done_data else None
        self.task_all = np.concatenate(self.task_data, axis=0) if self.task_data else None
        self.total_transitions = len(self.obs_all)
        if self.task_filter is not None and self.task_all is not None:
            mask = self.task_mask(self.task_filter)
            if mask.any():
                self.obs_all = self.obs_all[mask]
                self.action_all = self.action_all[mask]
                if self.reward_all is not None: self.reward_all = self.reward_all[mask]
                if self.done_all is not None: self.done_all = self.done_all[mask]
                self.task_all = self.task_all[mask]
        self.valid_starts = self.build_valid_starts()
        print(f"Loaded {self.total_transitions:,} transitions "
              f"(obs: {self.obs_all.shape}, action: {self.action_all.shape})")
        del self.obs_data, self.action_data, self.reward_data, self.done_data, self.task_data

    def task_mask(self, task_filter: str) -> np.ndarray:
        if self.task_all.dtype.kind in {"U", "S", "O"}:
            return np.array([str(t) for t in self.task_all]) == task_filter
        return np.ones(len(self.task_all), dtype=bool)

    def build_valid_starts(self) -> np.ndarray:
        if self.done_all is None:
            return np.arange(max(0, self.total_transitions - 1), dtype=np.int64)
        done = self.done_all.astype(bool).reshape(-1)
        valid = np.ones_like(done, dtype=bool)
        valid[-1] = False
        valid[:-1] &= ~done[:-1]
        return np.flatnonzero(valid)

    def sample(self, batch_size: int, seq_len: int = 2) -> dict[str, np.ndarray]:
        valid_starts = self.valid_starts[self.valid_starts <= self.total_transitions - seq_len]
        if self.done_all is not None:
            done_flat = self.done_all.astype(bool).reshape(-1)
            valid_starts = np.array(
                [s for s in valid_starts if not done_flat[s:s + seq_len - 1].any()],
                dtype=np.int64,
            )
        if len(valid_starts) == 0:
            raise ValueError(f"No valid starts for sequence length {seq_len}")
        starts = np.random.choice(valid_starts, size=batch_size)
        obs_batch = np.stack([self.obs_all[s:s + seq_len] for s in starts])
        action_batch = np.stack([self.action_all[s:s + seq_len] for s in starts])
        reward_batch = (
            np.stack([self.reward_all[s:s + seq_len] for s in starts])
            if self.reward_all is not None
            else np.zeros((batch_size, seq_len), dtype=np.float32)
        )
        done_batch = (
            np.stack([self.done_all[s:s + seq_len] for s in starts]).astype(np.float32)
            if self.done_all is not None
            else np.zeros((batch_size, seq_len), dtype=np.float32)
        )
        return {
            "obs": obs_batch.astype(np.float32),
            "action": action_batch.astype(np.float32),
            "reward": reward_batch.astype(np.float32),
            "done": done_batch,
        }
