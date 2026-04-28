"""LucidWM: TD-MPC2 on DMControl

Paper:     "TD-MPC2: Scalable, Robust World Models for Continuous Control" (Hansen et al., 2024)
Reference: https://github.com/nicklashansen/tdmpc2

Architecture (implicit world model -- no decoder):
  Encoder h:       obs -> z          (MLP + SimNorm, deterministic latent)
  Dynamics d:      (z, a) -> z'      (MLP, deterministic transition)
  Reward r:        (z, a) -> R       (MLP -> 101 bins, discrete regression)
  Continue c:      (z, a) -> gamma   (MLP -> Bernoulli)
  Value Q:         (z, a) -> V       (ensemble of 5 MLPs -> 101 bins, discrete regression)
  Policy pi:       z -> (mu, sigma)  (MLP -> squashed Gaussian, SAC-style)

Key design choices (from paper Appendix A):
  - SimNorm: partition latent z into groups of V=8, softmax each group
  - LayerNorm + Mish activation throughout (not ReLU/ELU)
  - Discrete regression (log-spaced bins) for reward and value, trained with cross-entropy
  - Ensemble of 5 Q-functions, TD target from min of 2 randomly sampled heads
  - MPPI planning in latent space with policy prior for sampling
  - Soft actor-critic (SAC) loss for policy (max entropy)
  - EMA target network for Q-targets

Hyperparameters (model_size=5, ~5M params):
  - Latent dim: 512
  - Hidden dim: 512
  - Encoder layers: 2
  - Dynamics/reward/Q layers: 2
  - Planning horizon: 3
  - MPPI samples: 512, iterations: 6, temperature: 0.5
  - Discount: 0.99
  - Learning rate: 3e-4
  - Batch size: 256
  - Update-to-data ratio: 1

Components from Layer 1: MPPIPlanner
Env:     DMControl (walker-walk, cheetah-run, humanoid-walk, etc.)
Target:  Match published Table 1 within 1 std
Compute: ~8-24 hours per task on RTX 3080+

Modes:
  Online:  Agent collects its own data via MPPI planning (default, single-task)
  Offline: Train on pre-collected datasets from HuggingFace (multi-task)
           Datasets: nicklashansen/tdmpc2 (mt30: 345M transitions, mt80: 545M transitions)
           Usage: --offline --dataset mt30 --data-dir ./data
"""

import copy
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lucidwm_utils.logger import Logger
from lucidwm_utils.envs import make_env
from lucidwm_utils.metrics import evaluate
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.misc import get_device, set_seed

from lucidwm_components.networks import MLP
from lucidwm_components.distribution import SimNorm

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: TD-MPC2")

    # Standard args
    parser.add_argument("--env-id", type=str, default="walker-walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)

    # Algorithm-specific
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01, help="EMA target update rate")
    parser.add_argument("--prefill-steps", type=int, default=5000)
    parser.add_argument("--utd", type=int, default=1, help="Update-to-data ratio")
    parser.add_argument("--entropy-coef", type=float, default=1e-4, help="SAC entropy coefficient")
    parser.add_argument("--rho", type=float, default=0.5, help="Consistency loss weight")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--num-q", type=int, default=5)
    parser.add_argument("--simnorm-dim", type=int, default=8)
    parser.add_argument("--simnorm-temp", type=float, default=0.5)

    parser.add_argument("--mppi-n", type=int, default=512)
    parser.add_argument("--mppi-iter", type=int, default=6)
    parser.add_argument("--mppi-temp", type=float, default=0.5)

    # Offline mode
    parser.add_argument("--offline", action="store_true",
                        help="Train on pre-collected HuggingFace dataset instead of online interaction")
    parser.add_argument("--dataset", type=str, default="mt30", choices=["mt30", "mt80"],
                        help="Which dataset to use for offline training")
    parser.add_argument("--data-dir", type=str, default="data/tdmpc2",
                        help="Local directory to cache downloaded dataset chunks")
    parser.add_argument("--train-steps", type=int, default=500_000,
                        help="Number of gradient steps for offline training")
    parser.add_argument("--offline-batch-size", type=int, default=1024,
                        help="Batch size for offline training (paper uses 1024)")
    
    return parser.parse_args()
    
# Discrete Regression (log-spaced bins)
def two_hot_encode(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Encode scalar into two-hot vector over log-spaced bins.

    Bins are symmetric around 0 with log spacing: symlog applied to raw values,
    then linearly interpolated between nearest bins.

    Args:
        x: (...) scalar values

    Returns:
        (..., num_bins) two-hot encoded
    """
    bins = symlog_bins(num_bins, x.device)
    x = x.clamp(bins[0], bins[-1])
    below = torch.bucketize(x, bins) - 1
    below = below.clamp(0, num_bins - 2)
    above = below + 1
    weight = (x - bins[below]) / (bins[above] - bins[below] + 1e-8)
    target = torch.zeros(*x.shape, num_bins, device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), (1 - weight).unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), weight.unsqueeze(-1))
    return target

def decode_bins(logits: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Decode logits over bins to scalar via expected value.

    Args:
        logits: (..., num_bins)

    Returns:
        (...) scalar values
    """
    bins = symlog_bins(num_bins, logits.device)
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)

def symlog_bins(num_bins: int, device: torch.device) -> torch.Tensor:
    """Generate log-spaced symmetric bins in [-20, 20]."""
    return torch.linspace(-20, 20, num_bins, device=device)

# TD-MPC2 World Model
class TDMPC2Model(nn.Module):
    """TD-MPC2 implicit world model.

    Components (all MLP-based with LayerNorm + Mish):
      encoder:  obs_dim -> latent_dim (+ SimNorm)
      dynamics: (latent_dim + action_dim) -> latent_dim (+ SimNorm)
      reward:   (latent_dim + action_dim) -> NUM_BINS
      continue: (latent_dim + action_dim) -> 1
      Q-ensemble: (latent_dim + action_dim) -> NUM_BINS (x NUM_Q)
      policy:   latent_dim -> 2 * action_dim (mean + log_std for squashed Gaussian)
    """

    def __init__(self, obs_dim: int, action_dim: int, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            MLP(self.obs_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp)
        )

        self.dynamics = nn.Sequential(
            MLP(args.latent_dim + action_dim, args.latent_dim, args.hidden_dim, activation=nn.Mish),
            SimNorm(args.simnorm_dim, args.simnorm_temp)
        )

        self.rewards = MLP(args.latent_dim + action_dim, args.num_bins, args.hidden_dim, activation=nn.Mish)

        self.termination = MLP(args.latent_dim + action_dim, 1, args.hidden_dim, activation=nn.Mish)

        self.policy_prior = MLP(args.latent_dim, 2 * action_dim, args.hidden_dim, activation=nn.Mish)

        self.q = nn.ModuleList([
            MLP(args.latent_dim + action_dim, args.num_bins, args.hidden_dim, activation=nn.Mish) for _ in range(args.num_q)
        ])

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state.

        Args: obs (B, obs_dim)
        Returns: z (B, LATENT_DIM)
        """
        return self.encoder(obs)

    def next_state(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next latent state (deterministic).

        Args: z (B, LATENT_DIM), a (B, action_dim)
        Returns: z' (B, LATENT_DIM)
        """
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict reward logits (discrete regression).

        Args: z (B, LATENT_DIM), a (B, action_dim)
        Returns: logits (B, NUM_BINS)
        """
        return self.rewards(torch.cat([z, a], dim=-1))

    def cont(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict continue probability (logit).

        Args: z (B, LATENT_DIM), a (B, action_dim)
        Returns: logit (B, 1)
        """
        return self.termination(torch.cat([z, a], dim=-1))

    def q_values(self, z: torch.Tensor, a: torch.Tensor) -> list[torch.Tensor]:
        """Predict Q-value logits from all ensemble heads.

        Args: z (B, LATENT_DIM), a (B, action_dim)
        Returns: list of NUM_Q tensors, each (B, NUM_BINS)
        """
        za = torch.cat([z, a], dim=-1)
        return [head(za) for head in self.q]

    def policy(self, z: torch.Tensor, deterministic: bool = False):
        """Sample action from squashed Gaussian policy.

        Args:
            z: (B, LATENT_DIM)
            deterministic: if True, return mean action

        Returns:
            action: (B, action_dim) in [-1, 1] (tanh squashed)
            log_prob: (B,) log probability (for SAC)
        """
        out = self.policy_prior(z)  # (B, 2 * action_dim)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            return action, torch.zeros(z.shape[0], device=z.device)

        # Reparameterized sample
        noise = torch.randn_like(mean)
        raw_action = mean + std * noise
        action = torch.tanh(raw_action)

        # Log prob with tanh correction
        log_prob = (
            -0.5 * noise.pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1)

        return action, log_prob

# Offline Dataset Loading (HuggingFace: nicklashansen/tdmpc2)
class OfflineDataset:
    """Loader for TD-MPC2 offline datasets from HuggingFace.

    Datasets are stored as TensorDict .pt chunks on HuggingFace:
      - nicklashansen/tdmpc2 (mt30 and mt80 subdirectories)
      - Each chunk is ~13GB, contains obs, action, reward, task fields

    This loader downloads chunks on-demand using huggingface_hub,
    and samples random transitions for training.

    Args:
        dataset: "mt30" or "mt80"
        data_dir: local cache directory
        task_filter: optional task name to filter for single-task offline training
                     e.g., "walker-walk". None = use all tasks.
    """

    def __init__(self, dataset: str = "mt30", data_dir: str = "data/tdmpc2", task_filter: str | None = None):
        
        self.data_dir = Path(data_dir) / dataset
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.task_filter = task_filter

        # Download chunk index
        self.download_chunks()

        # Load all chunks into memory 
        self.chunks = []
        self.load_chunks()

    def download_chunks(self):
        """Download dataset chunks from HuggingFace if not cached."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_tree
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for offline datasets. "
                "Install it: pip install huggingface_hub"
            )
        
        repo_id = "nicklashansen/tdmpc2"
        prefix = self.dataset + "/"

        # List available chunks
        print(f"Checking HuggingFace repo for {self.dataset} dataset...")
        try:
            files = list_repo_tree(repo_id, path_in_repo=self.dataset, repo_type="dataset")
            chunk_files = [f.rfilename for f in files if f.rfilename.endswith(".pt")]
        except Exception:
            # Fallback: try known chunk naming pattern
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
        """Load downloaded chunks into memory."""
        
        chunk_files = sorted(self.data_dir.glob("chunk_*.pt"))
        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk files found in {self.data_dir}. "
                f"Download may have failed. Check your HuggingFace access."
            )

        print(f"Loading {len(chunk_files)} chunks from {self.data_dir}...")
        self.obs_data = []
        self.action_data = []
        self.reward_data = []
        self.task_data = []

        for chunk_path in chunk_files:
            print(f"  Loading {chunk_path.name}...")
            try:
                chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)

                # TensorDict format: extract raw tensors
                # The exact keys depend on the TensorDict structure
                if hasattr(chunk, "get"):
                    # TensorDict object
                    obs = chunk.get("obs", chunk.get("observation", None))
                    act = chunk.get("action", None)
                    rew = chunk.get("reward", None)
                    task = chunk.get("task", None)
                elif isinstance(chunk, dict):
                    obs = chunk.get("obs", chunk.get("observation", None))
                    act = chunk.get("action", None)
                    rew = chunk.get("reward", None)
                    task = chunk.get("task", None)
                else:
                    print(f"  Warning: unknown chunk format in {chunk_path.name}, skipping")
                    continue

                if obs is None or act is None:
                    print(f"  Warning: missing obs/action in {chunk_path.name}, skipping")
                    continue

                # Convert to numpy for compatibility with our buffer interface
                if isinstance(obs, torch.Tensor):
                    obs = obs.numpy()
                if isinstance(act, torch.Tensor):
                    act = act.numpy()
                if isinstance(rew, torch.Tensor):
                    rew = rew.numpy()

                self.obs_data.append(obs)
                self.action_data.append(act)
                if rew is not None:
                    self.reward_data.append(rew)

            except Exception as e:
                print(f"  Warning: failed to load {chunk_path.name}: {e}")
                continue

        if not self.obs_data:
            raise RuntimeError("No data successfully loaded from chunks")

        # Concatenate all chunks
        self.obs_all = np.concatenate(self.obs_data, axis=0)
        self.action_all = np.concatenate(self.action_data, axis=0)
        self.reward_all = np.concatenate(self.reward_data, axis=0) if self.reward_data else None

        self.total_transitions = len(self.obs_all)
        print(f"Loaded {self.total_transitions:,} transitions "
              f"(obs: {self.obs_all.shape}, action: {self.action_all.shape})")

        # Free individual chunks
        del self.obs_data, self.action_data, self.reward_data
        self.chunks = None

    def sample(self, batch_size: int, seq_len: int = 2) -> dict[str, np.ndarray]:
        """Sample random transitions for training.

        For TD-MPC2, we need short sequences (horizon + 1 steps).
        We sample random starting indices and extract consecutive transitions.

        Args:
            batch_size: number of sequences
            seq_len: number of consecutive transitions (default: horizon + 1)

        Returns:
            dict with obs (B, T, O), action (B, T, A), reward (B, T), done (B, T)
        """
        max_start = self.total_transitions - seq_len
        starts = np.random.randint(0, max_start, size=batch_size)

        obs_batch = np.stack([self.obs_all[s:s + seq_len] for s in starts])
        action_batch = np.stack([self.action_all[s:s + seq_len] for s in starts])

        if self.reward_all is not None:
            reward_batch = np.stack([self.reward_all[s:s + seq_len] for s in starts])
        else:
            reward_batch = np.zeros((batch_size, seq_len), dtype=np.float32)

        done_batch = np.zeros((batch_size, seq_len), dtype=np.float32)

        return {
            "obs": obs_batch.astype(np.float32),
            "action": action_batch.astype(np.float32),
            "reward": reward_batch.astype(np.float32),
            "done": done_batch,
        }

# MPPI planning
@torch.no_grad()
def plan_mppi(
    model: TDMPC2Model,
    args,
    z: torch.Tensor,
    prev_mean: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MPPI planning with policy prior in latent space.

    Uses the policy network to generate action priors, then refines
    with MPPI. Key difference from pure MPPI: initial samples come
    from the policy, not uniform noise.

    Args:
        model: TD-MPC2 model (eval mode)
        z: (1, LATENT_DIM) current latent state
        prev_mean: (horizon, action_dim) previous plan (for warm-start), or None
        horizon: planning horizon
        action_dim: action space dimension
        device: torch device

    Returns:
        action: (action_dim,) selected first action
        mean: (horizon, action_dim) updated plan for warm-start
    """
    z = z.expand(args.mppi_n, -1)  # (N, LATENT_DIM)

    # Warm-start: shift previous plan by 1 step
    if prev_mean is not None:
        mean = torch.cat([prev_mean[1:], prev_mean[-1:]], dim=0)  # (H, A)
    else:
        mean = torch.zeros(horizon, action_dim, device=device)

    std = 2.0 * torch.ones(horizon, action_dim, device=device)

    for _ in range(args.mppi_iter):
        # Sample actions: mix policy prior with Gaussian perturbation
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            args.mppi_n, horizon, action_dim, device=device
        )
        actions = actions.clamp(-1, 1)  # (N, H, A)

        # Replace half the samples with policy-generated actions
        n_policy = args.mppi_n // 2
        s = z[:n_policy]
        for t in range(horizon):
            pi_action, _ = model.policy(s)
            actions[:n_policy, t] = pi_action
            s = model.next_state(s, pi_action)

        # Evaluate all candidates
        s = z.clone()
        total_reward = torch.zeros(args.mppi_n, device=device)
        for t in range(horizon):
            r_logits = model.reward(s, actions[:, t])
            total_reward += decode_bins(r_logits, args.num_bins)
            s = model.next_state(s, actions[:, t])

        # Terminal value: min of 2 random Q-heads
        final_action, _ = model.policy(s)
        q_logits = model.q_values(s, final_action)
        idx = torch.randperm(args.num_q)[:2]
        q1 = decode_bins(q_logits[idx[0]], args.num_bins)
        q2 = decode_bins(q_logits[idx[1]], args.num_bins)
        total_reward += torch.min(q1, q2)

        # Softmax weighting
        weights = F.softmax(total_reward / args.mppi_temp, dim=0)  # (N,)
        mean = (weights[:, None, None] * actions).sum(dim=0)   # (H, A)
        std = (std * 0.5).clamp(min=0.05)

    return mean[0], mean

# Training
def update(
    model: TDMPC2Model,
    target_model: TDMPC2Model,
    optimizer: optim.Optimizer,
    batch: dict,
    args,
    device: torch.device,
) -> dict[str, float]:
    """Single TD-MPC2 update step.

    Jointly optimizes:
    1. Consistency loss: latent dynamics should be consistent with encoder
    2. Reward prediction loss: cross-entropy on binned rewards
    3. Value (Q) loss: TD learning with ensemble, discrete regression
    4. Policy loss: SAC-style maximum entropy

    Args:
        model: online model
        target_model: EMA target model
        optimizer: Adam optimizer (shared for all components)
        batch: dict with obs (B,H+1,O), action (B,H+1,A), reward (B,H+1), done (B,H+1)
        args: parsed arguments
        device: torch device

    Returns:
        metrics dict
    """
    obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)     # (B, T, O)
    action = torch.tensor(batch["action"], dtype=torch.float32, device=device)  # (B, T, A)
    reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)  # (B, T)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)      # (B, T)

    B, T = obs.shape[:2]
    H = min(T - 1, args.horizon)

    # Encode first observation
    z = model.encode(obs[:, 0])  # (B, LATENT_DIM)

    total_loss = torch.tensor(0.0, device=device)
    consistency_loss_sum = 0.0
    reward_loss_sum = 0.0
    value_loss_sum = 0.0

    for t in range(H):
        a_t = action[:, t]

        # ── Consistency loss ──────────────────────────────────────────
        # Predicted next latent should match encoder's encoding of true next obs
        z_pred = model.next_state(z, a_t)                          # (B, LATENT)
        with torch.no_grad():
            z_target = target_model.encode(obs[:, t + 1])          # (B, LATENT)
        consistency_loss = F.mse_loss(z_pred, z_target)
        total_loss = total_loss + args.rho * consistency_loss
        consistency_loss_sum += consistency_loss.item()

        # ── Reward loss (discrete regression, cross-entropy) ──────────
        r_logits = model.reward(z, a_t)                            # (B, NUM_BINS)
        r_target = two_hot_encode(reward[:, t], args.num_bins)                    # (B, NUM_BINS)
        reward_loss = -(r_target * F.log_softmax(r_logits, dim=-1)).sum(dim=-1).mean()
        total_loss = total_loss + reward_loss
        reward_loss_sum += reward_loss.item()

        # ── Q-value loss (TD target from target model) ────────────────
        with torch.no_grad():
            # Next state encoding from target
            z_next_target = target_model.encode(obs[:, t + 1])
            # Sample action from current policy
            a_next, log_prob_next = model.policy(z_next_target)
            # TD target: min of 2 random Q-heads from target
            q_target_logits = target_model.q_values(z_next_target, a_next)
            idx = torch.randperm(args.num_q)[:2]
            q1_target = decode_bins(q_target_logits[idx[0]], args.num_bins)
            q2_target = decode_bins(q_target_logits[idx[1]], args.num_bins)
            q_next = torch.min(q1_target, q2_target)
            # TD target with entropy bonus
            gamma = args.discount * (1 - done[:, t])
            td_target = reward[:, t] + gamma * (q_next - args.entropy_coef * log_prob_next)

        # Loss for each Q-head
        q_logits_all = model.q_values(z.detach(), a_t)
        td_target_encoded = two_hot_encode(td_target, args.num_bins)
        for q_logits in q_logits_all:
            q_loss = -(td_target_encoded * F.log_softmax(q_logits, dim=-1)).sum(-1).mean()
            total_loss = total_loss + q_loss / args.num_q
        value_loss_sum += q_loss.item()

        # Advance latent state (use predicted, not re-encoded)
        z = z_pred

    # ── Policy loss (SAC max entropy) ─────────────────────────────────
    # Detach latent to prevent gradients flowing through world model
    z_policy = model.encode(obs[:, 0]).detach()
    a_pi, log_prob = model.policy(z_policy)
    q_logits_all = model.q_values(z_policy, a_pi)
    # Use min of 2 random Q-heads for policy update
    idx = torch.randperm(args.num_q)[:2]
    q1 = decode_bins(q_logits_all[idx[0]], args.num_bins)
    q2 = decode_bins(q_logits_all[idx[1]], args.num_bins)
    q_min = torch.min(q1, q2)
    policy_loss = (args.entropy_coef * log_prob - q_min).mean()
    total_loss = total_loss + policy_loss

    # Backprop
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 20.0)
    optimizer.step()

    # EMA update target model
    with torch.no_grad():
        for p, tp in zip(model.parameters(), target_model.parameters()):
            tp.data.lerp_(p.data, args.tau)

    return {
        "losses/total": total_loss.item(),
        "losses/consistency": consistency_loss_sum / H,
        "losses/reward": reward_loss_sum / H,
        "losses/value": value_loss_sum / H,
        "losses/policy": policy_loss.item(),
    }

def train_offline(model, target_model, optimizer, dataset, args, device, logger):
    """Offline training loop using pre-collected dataset.

    Iterates for args.train_steps gradient steps, sampling from the
    offline dataset. No environment interaction.

    Args:
        model: online model
        target_model: EMA target
        optimizer: shared Adam optimizer
        dataset: OfflineDataset instance
        args: parsed arguments
        device: torch device
        logger: Logger instance
    """
    seq_len = args.horizon + 1
    batch_size = args.offline_batch_size

    print(f"\nOffline training: {args.train_steps} steps, "
          f"batch_size={batch_size}, horizon={args.horizon}")

    for step in range(args.train_steps):
        batch = dataset.sample(batch_size, seq_len=seq_len)
        metrics = update(model, target_model, optimizer, batch, args, device)

        if step % 1000 == 0:
            logger.log(metrics, step=step)
            print(f"  Step {step:>7d}/{args.train_steps}  "
                  f"loss={metrics['losses/total']:.4f}  "
                  f"consistency={metrics['losses/consistency']:.4f}  "
                  f"reward={metrics['losses/reward']:.4f}")

        # Periodic evaluation (requires env)
        if step > 0 and step % args.eval_freq == 0 and args.env_id:
            model.eval()

            def agent_fn(o):
                with torch.no_grad():
                    o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    z = model.encode(o_t)
                    a, _ = model.policy(z, deterministic=True)
                    return a.squeeze(0).cpu().numpy()

            try:
                eval_result = evaluate(
                    lambda: make_env(args.env_id, args.seed + 100,
                                     action_repeat=args.action_repeat),
                    agent_fn,
                    num_episodes=args.eval_episodes,
                )
                print(f"  Eval: {eval_result['mean_return']:.1f} "
                      f"+/- {eval_result['std_return']:.1f}")
                logger.log({
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                }, step=step)
            except Exception as e:
                print(f"  Eval failed (env not available?): {e}")

            model.train()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Offline mode
    if args.offline:
        print(f"TD-MPC2 OFFLINE | dataset={args.dataset} | device={device}")

        # Load dataset
        dataset = OfflineDataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            task_filter=args.env_id if args.env_id != "walker-walk" else None
        )

        obs_dim = dataset.obs_all.shape[-1]
        action_dim = dataset.action_all.shape[-1]

        # Models
        model = TDMPC2Model(obs_dim, action_dim, args).to(device)
        target_model = copy.deepcopy(model).to(device)
        target_model.requires_grad_(False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print(f"Model: obs={obs_dim}, act={action_dim}, "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        # Logger
        run_name = f"tdmpc2_offline_{args.dataset}_s{args.seed}"
        logger = Logger(
            project=args.wandb_project, name=run_name,
            config=vars(args), use_wandb=args.track,
        )

        # Train
        train_offline(model, target_model, optimizer, dataset, args, device, logger)

        # Save checkpoint
        torch.save({
            "model": model.state_dict(),
            "target_model": target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, f"tdmpc2_offline_{args.dataset}_s{args.seed}.pt")
        print("Saved checkpoint.")

        logger.close()
    else:
        # Environment
        env = make_env(args.env_id, args.seed, action_repeat=args.action_repeat)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Models
        model = TDMPC2Model(obs_dim, action_dim, args).to(device)
        target_model = copy.deepcopy(model).to(device)
        target_model.requires_grad_(False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Replay buffer
        seq_len = args.horizon + 1
        buffer = ReplayBuffer(
            capacity=1_000_000,
            obs_shape=(obs_dim,),
            action_dim=action_dim,
            seq_len=seq_len,
        )

        # Logger
        run_name = f"tdmpc2_{args.env_id}_s{args.seed}"
        logger = Logger(
            project=args.wandb_project, name=run_name,
            config=vars(args), use_wandb=args.track,
        )

        print(f"TD-MPC2 ONLINE | env={args.env_id} | obs={obs_dim} | act={action_dim} | "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        # ── Online training loop ──────────────────────────────────────
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0
        prev_mean = None

        for step in range(args.total_steps):
            # Act
            if step < args.prefill_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    z = model.encode(obs_t)
                    action, prev_mean = plan_mppi(
                        model, args, z, prev_mean, args.horizon, action_dim, device,
                    )
                    action = action.cpu().numpy()

            # Step env
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add_step(obs, action, reward, done)

            episode_return += reward
            episode_length += 1
            obs = next_obs

            if done:
                logger.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                }, step=step)
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
                prev_mean = None

            # Train
            if step >= args.prefill_steps and buffer.num_episodes >= 1:
                for _ in range(args.utd):
                    try:
                        batch = buffer.sample(args.batch_size, seq_len=seq_len)
                    except ValueError:
                        break
                    metrics = update(model, target_model, optimizer, batch, args, device)

                if step % 1000 == 0:
                    logger.log(metrics, step=step)

            # Evaluate
            if step > 0 and step % args.eval_freq == 0:
                model.eval()

                def agent_fn(o):
                    with torch.no_grad():
                        o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                        z = model.encode(o_t)
                        a, _ = model.policy(z, deterministic=True)
                        return a.squeeze(0).cpu().numpy()

                eval_result = evaluate(
                    lambda: make_env(args.env_id, args.seed + 100,
                                     action_repeat=args.action_repeat),
                    agent_fn,
                    num_episodes=args.eval_episodes,
                )
                model.train()

                print(f"Step {step:>7d} | eval={eval_result['mean_return']:.1f} "
                      f"+/- {eval_result['std_return']:.1f}")
                logger.log({
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                }, step=step)

        logger.close()
        print("Done.")