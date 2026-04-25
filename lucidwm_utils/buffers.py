"""Replay buffers for world model training.

Episode-based storage with sequence sampling. All world model algorithms
(except model-free baselines) sample sequences from replay.
"""

from __future__ import annotations

import numpy as np

class ReplayBuffer:
    """Episode-based replay buffer with sequence sampling.

    Stores full episodes. Samples random sequences of length `seq_len`
    from random episodes. This is the standard buffer for Dreamer, PlaNet,
    TD-MPC2, DIAMOND, etc.

    Args:
        capacity: Maximum number of transitions (across all episodes).
        obs_shape: Shape of observations (e.g., (3, 64, 64) for images).
        action_dim: Dimension of action space.
        seq_len: Default sequence length for sampling.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_shape: tuple[int, ...] = (3, 64, 64),
        action_dim: int = 1,
        seq_len: int = 50,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.seq_len = seq_len

        # Store episodes as list of dicts
        self._episodes: list[dict[str, np.ndarray]] = []
        self._current_episode: dict[str, list] = self._new_episode()
        self._total_steps = 0

    def _new_episode(self) -> dict[str, list]:
        return {"obs": [], "action": [], "reward": [], "done": []}

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Add a single transition. Call at every env step.

        Args:
            obs: Observation. Shape: obs_shape.
            action: Action. Shape: (action_dim,).
            reward: Scalar reward.
            done: Whether episode terminated or was truncated.
        """
        self._current_episode["obs"].append(obs)
        self._current_episode["action"].append(action)
        self._current_episode["reward"].append(reward)
        self._current_episode["done"].append(done)
        self._total_steps += 1

        if done:
            # Convert lists to arrays and store
            episode = {
                "obs": np.array(self._current_episode["obs"]),
                "action": np.array(self._current_episode["action"]),
                "reward": np.array(self._current_episode["reward"], dtype=np.float32),
                "done": np.array(self._current_episode["done"], dtype=np.float32),
            }
            self._episodes.append(episode)
            self._current_episode = self._new_episode()

            # Evict oldest episodes if over capacity
            while self._total_steps > self.capacity and len(self._episodes) > 1:
                removed = self._episodes.pop(0)
                self._total_steps -= len(removed["reward"])

    def sample(
        self, batch_size: int, seq_len: int | None = None
    ) -> dict[str, np.ndarray]:
        """Sample a batch of sequences from random episodes.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Sequence length. Default: self.seq_len.

        Returns:
            dict with keys:
                obs:    (B, T, *obs_shape)
                action: (B, T, action_dim)
                reward: (B, T)
                done:   (B, T)
        """
        if seq_len is None:
            seq_len = self.seq_len

        # Filter episodes long enough
        valid_episodes = [ep for ep in self._episodes if len(ep["reward"]) >= seq_len]
        if not valid_episodes:
            raise ValueError(
                f"No episodes with length >= {seq_len}. "
                f"Longest episode: {max(len(ep['reward']) for ep in self._episodes)}"
            )

        obs_batch, action_batch, reward_batch, done_batch = [], [], [], []

        for _ in range(batch_size):
            # Random episode
            ep_idx = np.random.randint(len(valid_episodes))
            ep = valid_episodes[ep_idx]

            # Random start index
            max_start = len(ep["reward"]) - seq_len
            start = np.random.randint(0, max_start + 1)

            obs_batch.append(ep["obs"][start : start + seq_len])
            action_batch.append(ep["action"][start : start + seq_len])
            reward_batch.append(ep["reward"][start : start + seq_len])
            done_batch.append(ep["done"][start : start + seq_len])

        return {
            "obs": np.array(obs_batch),       # (B, T, *obs_shape)
            "action": np.array(action_batch),  # (B, T, action_dim)
            "reward": np.array(reward_batch),  # (B, T)
            "done": np.array(done_batch),      # (B, T)
        }

class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with priority-weighted sampling.

    Extends ReplayBuffer for MuZero and EfficientZero.
    Priorities are per-episode (based on max TD error in episode).

    Args:
        capacity: Maximum transitions.
        obs_shape: Observation shape.
        action_dim: Action dimension.
        seq_len: Default sequence length.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta: Importance sampling exponent.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_shape: tuple[int, ...] = (3, 64, 64),
        action_dim: int = 1,
        seq_len: int = 50,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(capacity, obs_shape, action_dim, seq_len)
        self.alpha = alpha
        self.beta = beta
        self._priorities: list[float] = []
        self._max_priority = 1.0

    def add_step(self, obs, action, reward, done):
        super().add_step(obs, action, reward, done)
        if done:
            # New episodes get max priority
            self._priorities.append(self._max_priority)

    def update_priorities(self, episode_indices: list[int], priorities: list[float]):
        """Update priorities for specific episodes.

        Args:
            episode_indices: Indices into self._episodes.
            priorities: New priority values.
        """
        for idx, pri in zip(episode_indices, priorities):
            if idx < len(self._priorities):
                self._priorities[idx] = pri
                self._max_priority = max(self._max_priority, pri)

    def sample(
        self, batch_size: int, seq_len: int | None = None
    ) -> dict[str, np.ndarray]:
        """Sample with priority weighting.

        Returns same dict as ReplayBuffer.sample, plus:
            indices:  (B,) episode indices (for priority update)
            weights:  (B,) importance sampling weights
        """
        if seq_len is None:
            seq_len = self.seq_len

        valid_mask = np.array([len(ep["reward"]) >= seq_len for ep in self._episodes])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            raise ValueError(f"No episodes with length >= {seq_len}")

        # Compute sampling probabilities
        priorities = np.array([self._priorities[i] for i in valid_indices])
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample episodes
        chosen = np.random.choice(len(valid_indices), size=batch_size, p=probs)
        ep_indices = valid_indices[chosen]

        # Importance sampling weights
        weights = (len(valid_indices) * probs[chosen]) ** (-self.beta)
        weights = weights / weights.max()

        obs_batch, action_batch, reward_batch, done_batch = [], [], [], []

        for idx in ep_indices:
            ep = self._episodes[idx]
            max_start = len(ep["reward"]) - seq_len
            start = np.random.randint(0, max_start + 1)
            obs_batch.append(ep["obs"][start : start + seq_len])
            action_batch.append(ep["action"][start : start + seq_len])
            reward_batch.append(ep["reward"][start : start + seq_len])
            done_batch.append(ep["done"][start : start + seq_len])

        return {
            "obs": np.array(obs_batch),
            "action": np.array(action_batch),
            "reward": np.array(reward_batch),
            "done": np.array(done_batch),
            "indices": ep_indices,
            "weights": weights.astype(np.float32),
        }