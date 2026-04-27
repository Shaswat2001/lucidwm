"""
Common dataset adapters which can be used across multiple world models
"""

import numpy as np

# Dataset Adapter
class MinariDatasetAdapter:
    """
    Adapter that loads a Minari dataset and provides a sampling interface
    compatible with LucidWM's training loops.

    Minari datasets store episodes as EpisodeData objects with:
      - observations: np.ndarray (T+1, obs_dim)
      - actions: np.ndarray (T, act_dim)
      - rewards: np.ndarray (T,)
      - terminations: np.ndarray (T,)
      - truncations: np.ndarray (T,)

    This adapter flattens all episodes into a single buffer and provides
    random sequence sampling.

    Args:
        dataset_id: Minari dataset identifier, e.g., "mujoco/halfcheetah/medium-v0"
    """

    def __init__(self, dataset_id: str):
        try:
            import minari
        except ImportError:
            raise ImportError(
                "Minari is required for MuJoCo offline datasets. "
                "Install it: pip install minari"
            )

        print(f"Loading Minari dataset: {dataset_id}")
        self.minari_dataset = minari.load_dataset(dataset_id, download=True)

        # Flatten all episodes into arrays
        all_obs, all_actions, all_rewards, all_dones = [], [], [], []

        for episode in self.minari_dataset.iterate_episodes():
            T = len(episode.actions)
            # observations has T+1 entries (includes terminal obs)
            all_obs.append(episode.observations[:T].astype(np.float32))
            all_actions.append(episode.actions.astype(np.float32))
            all_rewards.append(episode.rewards.astype(np.float32))
            dones = np.logical_or(episode.terminations, episode.truncations).astype(np.float32)
            all_dones.append(dones)

        self.obs = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_actions, axis=0)
        self.rewards = np.concatenate(all_rewards, axis=0)
        self.dones = np.concatenate(all_dones, axis=0)

        self.obs_dim = self.obs.shape[-1]
        self.action_dim = self.actions.shape[-1]
        self.total_transitions = len(self.obs)

        print(f"  Loaded {self.total_transitions:,} transitions "
              f"(obs_dim={self.obs_dim}, act_dim={self.action_dim})")
        print(f"  Episodes: {self.minari_dataset.total_episodes}, "
              f"Total steps: {self.minari_dataset.total_steps}")

    def sample(self, batch_size: int, seq_len: int = 2) -> dict[str, np.ndarray]:
        """Sample random consecutive sequences for training.

        Args:
            batch_size: number of sequences
            seq_len: consecutive steps per sequence

        Returns:
            dict with obs (B, T, O), action (B, T, A), reward (B, T), done (B, T)
        """
        max_start = self.total_transitions - seq_len
        starts = np.random.randint(0, max_start, size=batch_size)

        return {
            "obs": np.stack([self.obs[s:s + seq_len] for s in starts]),
            "action": np.stack([self.actions[s:s + seq_len] for s in starts]),
            "reward": np.stack([self.rewards[s:s + seq_len] for s in starts]),
            "done": np.stack([self.dones[s:s + seq_len] for s in starts]),
        }

    def sample_start_states(self, batch_size: int) -> np.ndarray:
        """Sample random observations for policy learning start states.

        Returns: (B, obs_dim)
        """
        indices = np.random.randint(0, self.total_transitions, size=batch_size)
        return self.obs[indices]

    def recover_environment(self):
        """Recover the Gymnasium environment from the Minari dataset."""
        return self.minari_dataset.recover_environment()
