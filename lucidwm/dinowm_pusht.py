"""LucidWM: DINO-WM on PushT / PointMaze / MetaWorld

Paper:     "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot
            Planning" (Zhou et al., ICML 2025)
Reference: https://github.com/gaoyuezhou/dino_wm

Core idea:
  Freeze a pre-trained DINOv2 ViT-S/14 backbone. Extract spatial patch embeddings
  z_t in R^{N x E} from each image frame (N=256 patches, E=384 for ViT-S/14).
  Train a causal Vision Transformer to predict z_{t+1} from (z_{t-H:t}, a_{t-H:t}).
  Plan at test time via CEM in latent space: optimize action sequences to minimize
  ||z_predicted - z_goal||^2. No decoder, no reconstruction, no reward model.

Architecture:
  Observation model:  DINOv2 ViT-S/14, FROZEN. Image (3,224,224) -> patches (N, E).
                      N=256 patches for 224x224 images with patch_size=14.
                      E=384 embedding dim for ViT-S.
  Transition model:   Causal ViT (decoder-only transformer).
                      Input: sequence of [z_{t-H}, a_{t-H}, z_{t-H+1}, a_{t-H+1}, ...]
                      where actions are broadcast to patch dimension via learned MLP.
                      Causal mask: frame-level (each frame sees only past frames).
                      Output: predicted z_{t+1} (N, E).
                      Depth=6, heads=16, mlp_dim=2048, ~19M params.
  Planning:           CEM (Cross-Entropy Method) in latent DINOv2 space.
                      Cost = ||rollout_z_T - z_goal||^2 (L2 in patch feature space).
                      MPC loop: plan, execute first action, re-encode, repeat.

Training:
  - Offline: train on pre-collected trajectories of (obs, action) pairs.
  - Loss: MSE between predicted and true DINOv2 patch embeddings of next frame.
  - No reward model, no reconstruction loss, no task-specific signal.

Key hyperparameters (from paper Appendix A.9):
  - DINOv2 backbone: ViT-S/14 (frozen), embedding dim E=384
  - Image size: 224x224 (resized from env)
  - Transition ViT: depth=6, heads=16, mlp_dim=2048
  - Context length H: 3 (number of past frames)
  - Frameskip: env-dependent (1-5)
  - CEM: 200 candidates, 20 elites, 10 iterations, horizon 5-20
  - Learning rate: 1e-4, batch size: 64
  - Training epochs: 200

Components from Layer 1: CEMPlanner (adapted for latent goal-reaching cost)
Env:     PushT, PointMaze, MetaWorld, Wall
Target:  Zero-shot goal reaching without rewards or demonstrations
Compute: ~4-8 hours for world model training on RTX 3080+
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import set_seed, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: DINO-WM")

    # Standard args
    parser.add_argument("--env-id", type=str, default="pusht")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--logdir", type=str, default="exp/dinowm")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to offline trajectory data (npz files)")
    parser.add_argument("--frameskip", type=int, default=1,
                        help="Frameskip for prediction (env-dependent)")
    parser.add_argument("--img-size", type=int, default=224)

    # World model training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-len", type=int, default=3)

    # DINO
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino-embed-dim", type=int, default=384)
    parser.add_argument("--dino-patch-size", type=int, default=14)
    parser.add_argument("--num-patches", type=int, default=256)

    # Transition Model
    parser.add_argument("--vit-depth", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=16)
    parser.add_argument("--vit-mlp-dim", type=int, default=2048)

    # Planning (CEM)
    parser.add_argument("--plan-horizon", type=int, default=10)
    parser.add_argument("--cem-candidates", type=int, default=200)
    parser.add_argument("--cem-elites", type=int, default=20)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--mpc-steps", type=int, default=50,
                        help="Number of MPC steps per episode")
    parser.add_argument("--action-dim", type=int, default=2,
                        help="Action dimension (env-dependent)")

    # Flags
    parser.add_argument("--skip-train", action="store_true")

    return parser.parse_args()

# Observation model: Frozen DINOv2
class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 ViT-S/14 backbone for patch feature extraction.

    Extracts spatial patch embeddings from images. The backbone is loaded
    from torch.hub and kept FROZEN throughout training and planning.

    Input:  (B, 3, 224, 224) RGB images normalized to ImageNet stats
    Output: (B, N, E) patch embeddings, N=256 patches, E=384 for ViT-S
    """

    def __init__(self, model_name: str):
        super(DINOv2Encoder, self).__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 patch features.

        Args:
            img: (B, 3, H, W) in [0, 1]

        Returns:
            patches: (B, N, E) where N = (H/14)*(W/14), E = 384
        """
        img = (img - self.mean) / self.std
        # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
        features = self.backbone.forward_features(img)
        return features["x_norm_patchtokens"]  # (B, N, E)
    
# Transition Model: Causal Vision Transformer
class ActionEmbedding(nn.Module):
    """Project action vector to patch embedding space and broadcast.

    Maps action a_t in R^A to R^E, then broadcasts to (N, E) to be
    added to or concatenated with patch tokens at timestep t.

    Args:
        action_dim: dimension of action space
        embed_dim: target embedding dimension (E=384)
    """

    def __init__(self, action_dim: int, embedding_dim: int):
        super(ActionEmbedding, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action (B, A) -> (B, 1, E), to be broadcast across patches."""
        return self.net(action).unsqueeze(1)  # (B, 1, E)

def generate_frame_causal_mask(num_patches: int, num_frames: int) -> torch.Tensor:
    """Generate block-causal attention mask (matching official repo).

    Each frame is a block of num_patches tokens. Frame t can attend to
    all tokens in frames <= t (bidirectional within-frame, causal across frames).

    For num_frames=3, num_patches=256:
        Frame 0: [ones,  zeros, zeros]    <- sees frame 0 only
        Frame 1: [ones,  ones,  zeros]    <- sees frames 0, 1
        Frame 2: [ones,  ones,  ones]     <- sees frames 0, 1, 2

    Returns:
        mask: (1, 1, num_frames*num_patches, num_frames*num_patches)
              1 = attend, 0 = mask out
    """
    ones = torch.ones(num_patches, num_patches)
    zeros = torch.zeros(num_patches, num_patches)

    rows = []
    for i in range(num_frames):
        row = torch.cat([ones] * (i + 1) + [zeros] * (num_frames - i - 1), dim=1)
        rows.append(row)

    mask = torch.cat(rows, dim=0)  # (num_frames*num_patches, num_frames*num_patches)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


class FeedForward(nn.Module):
    """Pre-norm FFN block (matching official repo)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Attention(nn.Module):
    """Multi-head self-attention with frame-level causal masking.

    Matching the official DINO-WM repo implementation.
    Pre-norm, manual QKV projection, explicit mask application.

    Args:
        dim: input/output dimension
        heads: number of attention heads
        dim_head: dimension per head
        dropout: attention dropout rate
        num_patches: patches per frame (for mask construction)
        num_frames: number of context frames (for mask construction)
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, num_patches: int = 256,
                 num_frames: int = 3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # Pre-compute the block-causal mask
        self.register_buffer(
            "mask", generate_frame_causal_mask(num_patches, num_frames)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) where L = num_frames * num_patches."""
        B, T, C = x.shape

        # Pre-norm
        x_norm = self.norm(x)

        # QKV projection
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, T, self.heads, -1).permute(0, 2, 1, 3),
            qkv,
        )  # each: (B, heads, T, dim_head)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, T, T)

        # Apply frame-level causal mask
        dots = dots.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # Aggregate values
        out = torch.matmul(attn, v)  # (B, heads, T, dim_head)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, inner_dim)

        return self.to_out(out)

class TransitionViT(nn.Module):
    """
    Causal ViT for dynamics prediction in DINOv2 patch space.

    Takes a sequence of (patch_embeddings, action) pairs over H context frames
    and predicts the next frame's patch embeddings.

    Architecture: decoder-only transformer (no tokenization layer since inputs
    are already patch embeddings). Causal attention at frame level: each frame
    can only attend to itself and past frames.

    Input sequence construction for context H=3:
      Frame tokens: [z_{t-2} patches, z_{t-1} patches, z_t patches]
      Action tokens are added to each frame's patches via broadcasting.
      Total sequence: H * N tokens, each in R^E.

    Args:
        embed_dim: patch embedding dimension. Default: 384.
        depth: number of transformer blocks. Default: 6.
        num_heads: attention heads. Default: 16.
        mlp_dim: FFN hidden dimension. Default: 2048.
        num_patches: patches per frame. Default: 256.
        context_len: number of context frames H. Default: 3.
        action_dim: action space dimension.
    """

    def __init__(
        self,
        embedding_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_patches: int,
        context_len: int,
        action_dim: int = 2,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        
        super(TransitionViT, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.context_len = context_len

        total_tokens = context_len * num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, embedding_dim) * 0.02)

        self.action_embedding = ActionEmbedding(action_dim, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embedding_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(embedding_dim, heads=num_heads, dim_head=dim_head,
                          dropout=dropout, num_patches=num_patches,
                          num_frames=context_len),
                FeedForward(embedding_dim, mlp_dim, dropout=dropout),
            ]))

        # Prediction head: project output back to DINOv2 patch embedding space
        self.pred_head = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
        )

    def forward(
        self, z_seq: torch.Tensor, a_seq: torch.Tensor
    ) -> torch.Tensor:
        """Predict next frame's patch embeddings.

        Args:
            z_seq: (B, H, N, E) -- H context frames of DINOv2 patch embeddings
            a_seq: (B, H, A) -- actions taken at each context frame

        Returns:
            z_pred: (B, N, E) -- predicted next frame patch embeddings
        """
        B, H, N, E = z_seq.shape

        # Mix actions into patch tokens BEFORE the transformer
        # (action conditioning happens upstream of attention, matching repo)
        x = z_seq.clone()
        for t in range(H):
            a_emb = self.action_embed(a_seq[:, t])  # (B, 1, E)
            x[:, t] = x[:, t] + a_emb               # broadcast: (B, N, E)

        # Flatten to single sequence: (B, H*N, E)
        x = x.reshape(B, H * N, E)

        # Add single flat positional embedding (matching repo)
        n = x.shape[1]
        x = x + self.pos_embedding[:, :n]
        x = self.dropout_layer(x)

        # Transformer blocks with residual connections
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.norm(x)

        # Take last frame's tokens as prediction
        x_last = x[:, -N:]  # (B, N, E)

        return self.pred_head(x_last)  # (B, N, E)

# Dataset: PushT (matching official repo format)
# Adapted from: https://github.com/gaoyuezhou/dino_wm/blob/main/datasets/pusht_dset.py
class TrajectoryDataset(Dataset):
    """PushT dataset matching the official DINO-WM repo format.

    Loads video episodes (.mp4) via decord and action tensors (.pth).
    DINO-WM uses ONLY visual observations, not proprioceptive state.

    Args:
        data_path: path to dataset dir (e.g., "data/pusht_dataset/train")
        n_rollout: number of rollouts to load (None = all)
        normalize_action: normalize actions by precomputed stats
        relative: use relative actions (True) or absolute (False)
        action_scale: scale factor for actions
        with_velocity: include agent velocity in state
        img_size: resize images to this size
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: int | None = None,
        normalize_action: bool = True,
        relative: bool = True,
        action_scale: float = 100.0,
        with_velocity: bool = True,
        img_size: int= 224,
    ):
        
        self.data_path = Path(data_path)
        self.img_size = img_size

        self.pusht_action_mean = torch.tensor([-0.0087, 0.0068])
        self.pusht_action_std = torch.tensor([0.2019, 0.2002])

        self.states = torch.load(self.data_path / "states.pth", weights_only=True).float()

        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth", weights_only=True)
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth", weights_only=True)
        self.actions = self.actions.float() / action_scale

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        n = n_rollout if n_rollout else len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        # Proprio: first 2 dims of state (agent position)
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

    def get_frames(self, idx: int, frames: list[int] | range):
        """Load video frames and actions for an episode.

        Returns:
            obs: {"visual": (T, 3, H, W) float [0,1], "proprio": (T, D)}
            actions: (T, A)
            states: (T, S)

        DINO-WM uses only obs["visual"]. Proprio is for baselines.
        """
        try:
            from decord import VideoReader
            import decord
            decord.bridge.set_bridge("torch")
        except ImportError:
            raise ImportError("decord required for PushT videos: pip install decord")

        vid_path = self.data_path / "obses" / f"episode_{idx:03d}.mp4"
        reader = VideoReader(str(vid_path), num_threads=1)

        image = reader.get_batch(frames).float() / 255.0  # (T, H, W, C)
        image = image.permute(0, 3, 1, 2)                  # (T, C, H, W)

        if image.shape[-1] != self.img_size:
            image = F.interpolate(image, size=(self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False)

        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        return obs, self.actions[idx, frames], self.states[idx, frames]

    def __len__(self):
        return len(self.seq_lengths)

    def __getitem__(self, idx):
        T = self.get_seq_length(idx)
        return self.get_frames(idx, range(T))


class TrajSlicerDataset(Dataset):
    """Slice trajectories into fixed-length windows for training.

    Matching the official repo's TrajSlicerDataset.

    Args:
        traj_dataset: PushTDataset instance
        num_frames: window size (context_len + 1)
        frameskip: skip between frames (0 = consecutive)
    """

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
        frames = list(range(start, start + self.num_frames * self.effective_skip,
                            self.effective_skip))
        obs, actions, states = self.traj_dataset.get_frames(ep_idx, frames)
        return obs, actions, states

# Training
def train_world_model(encoder, transition, dataset, args, device, logger):
    """Train the transition model to predict next-frame DINOv2 features.

    Loss: MSE between predicted and true patch embeddings of the next frame.
    The DINOv2 encoder is frozen; only the transition ViT is trained.
    """
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(transition.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training DINO-WM transition model: {args.epochs} epochs, "
          f"{len(dataset)} samples")

    step = 0
    for epoch in range(args.epochs):
        transition.train()
        epoch_loss = 0.0

        for batch in loader:
            frames = batch["frames"].to(device)   # (B, H+1, 3, 224, 224)
            actions = batch["actions"].to(device)  # (B, H, A)

            B, T, C, H_img, W_img = frames.shape

            # Encode ALL frames with frozen DINOv2
            with torch.no_grad():
                all_z = encoder(frames.reshape(B * T, C, H_img, W_img))  # (B*T, N, E)
                all_z = all_z.reshape(B, T, args.num_patches, args.dino_embed_dim)

            # Split into context and target
            z_context = all_z[:, :-1]  # (B, H, N, E) -- context frames
            z_target = all_z[:, -1]    # (B, N, E) -- target (next frame)

            # Predict next frame
            z_pred = transition(z_context, actions)  # (B, N, E)

            # MSE loss in patch embedding space
            loss = F.mse_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(transition.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        scheduler.step()
        avg = epoch_loss / len(loader)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{args.epochs}  loss={avg:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
        logger.log({"wm/loss": avg, "wm/lr": scheduler.get_last_lr()[0]}, step=step)

    # Save
    ckpt_path = os.path.join(args.logdir, "dinowm_transition.pt")
    torch.save(transition.state_dict(), ckpt_path)
    print(f"  Saved transition model to {ckpt_path}")
    return transition

# Planning
@torch.no_grad()
def plan_cem(
    encoder: DINOv2Encoder,
    transition: TransitionViT,
    current_frames: torch.Tensor,
    goal_z: torch.Tensor,
    action_dim: int,
    args,
    device: torch.device,
) -> np.ndarray:
    """Plan action sequence via CEM to reach goal in DINOv2 latent space.

    Cost function: ||z_predicted_at_horizon - z_goal||^2

    Args:
        encoder: frozen DINOv2
        transition: trained dynamics model
        current_frames: (H, 3, 224, 224) recent context frames
        goal_z: (N, E) goal patch embeddings
        action_dim: action space dimension
        args: planning hyperparameters
        device: torch device

    Returns:
        action: (A,) first action of best sequence
    """
    H = args.context_len
    horizon = args.plan_horizon
    n_cand = args.cem_candidates
    n_elite = args.cem_elites
    n_iter = args.cem_iterations

    # Encode current context frames
    z_context = encoder(current_frames.unsqueeze(0).to(device))  # this won't work for batch
    # Actually encode each frame separately
    z_frames = []
    for i in range(current_frames.shape[0]):
        z_i = encoder(current_frames[i:i+1].to(device))  # (1, N, E)
        z_frames.append(z_i)
    z_context = torch.stack(z_frames, dim=1)  # (1, H, N, E)

    goal_z = goal_z.unsqueeze(0).to(device)  # (1, N, E)

    # CEM optimization
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    for iteration in range(n_iter):
        # Sample action sequences: (n_cand, horizon, A)
        noise = torch.randn(n_cand, horizon, action_dim, device=device)
        actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1, 1)

        # Evaluate each candidate by rolling out through the world model
        costs = torch.zeros(n_cand, device=device)
        z_ctx = z_context.expand(n_cand, -1, -1, -1).clone()  # (n_cand, H, N, E)

        for t in range(horizon):
            a_ctx = actions[:, max(0, t - H + 1):t + 1]  # actions for context window
            # Pad actions to match context length
            if a_ctx.shape[1] < H:
                pad = torch.zeros(n_cand, H - a_ctx.shape[1], action_dim, device=device)
                a_ctx = torch.cat([pad, a_ctx], dim=1)

            z_pred = transition(z_ctx, a_ctx)  # (n_cand, N, E)

            # Shift context window: drop oldest, append prediction
            z_ctx = torch.cat([z_ctx[:, 1:], z_pred.unsqueeze(1)], dim=1)

        # Cost: L2 distance to goal in patch embedding space
        costs = ((z_ctx[:, -1] - goal_z) ** 2).sum(dim=(-2, -1))  # (n_cand,)

        # Select elites
        elite_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]  # (n_elite, horizon, A)
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.01)

    return mean[0].cpu().numpy()  # first action

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(
        project=args.wandb_project,
        name=f"dinowm_{args.env_id}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    # ── Load frozen DINOv2 encoder ────────────────────────────────────
    print("Loading DINOv2 encoder (frozen)...")
    encoder = DINOv2Encoder().to(device)
    print(f"  DINOv2 loaded: {args.dino_model}, embed_dim={args.dino_embed_dim}, "
          f"patches={args.num_patches}")

    # ── Build transition model ────────────────────────────────────────
    transition = TransitionViT(
        embedding_dim=args.embedding_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        mlp_dim=args.vit_mlp_dim,
        num_patches=args.num_patches,
        action_dim=args.action_dim,
        context_len=args.context_len,
    ).to(device)
    n_params = sum(p.numel() for p in transition.parameters() if p.requires_grad)
    print(f"  Transition model: {n_params / 1e6:.1f}M trainable params")

    # ── Phase 1: Train world model ────────────────────────────────────
    if not args.skip_train:
        print(f"\n{'='*60}\nPhase 1: Training DINO-WM\n{'='*60}")
        pusht_dataset = TrajectoryDataset(
            data_path=args.data_dir,
                img_size=args.img_size,
            )
        num_frames = args.context_len + 1
        dataset = TrajSlicerDataset(pusht_dataset, num_frames, frameskip=args.frameskip)
        transition = train_world_model(encoder, transition, dataset, args, device, logger)
    else:
        print("Loading existing transition model...")
        ckpt_path = os.path.join(args.logdir, "dinowm_transition.pt")
        transition.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )

    # ── Phase 2: Planning (CEM in latent space) ──────────────────────
    print(f"\n{'='*60}\nPhase 2: Visual Goal Planning (CEM)\n{'='*60}")
    print("To plan, provide a goal image and current observation frames.")
    print("Example usage in code:")
    print("  goal_img = load_and_preprocess(goal_path)  # (3, 224, 224)")
    print("  goal_z = encoder(goal_img.unsqueeze(0))     # (1, N, E)")
    print("  action = plan_cem(encoder, transition, context_frames, goal_z, ...)")
    print()
    print("For MPC loop integration with an environment:")
    print("  for step in range(max_steps):")
    print("    action = plan_cem(encoder, transition, recent_frames, goal_z, ...)")
    print("    obs = env.step(action)")
    print("    recent_frames = update_context(recent_frames, obs)")

    logger.close()
    print("\nDone.")
