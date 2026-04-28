"""LucidWM: LeWM (LeWorldModel) on PushT / Reacher / OGBench

Paper:     "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
            Architecture from Pixels" (Maes et al., 2026)
Reference: https://github.com/lucas-maes/le-wm

Core idea:
  The simplest possible JEPA world model. Learn an encoder and predictor
  end-to-end from raw pixels with ONLY TWO loss terms: MSE prediction loss
  + SIGReg (a regularizer that prevents collapse by enforcing a Gaussian
  latent distribution). No stop-gradient, no EMA, no frozen backbone,
  no reconstruction loss, no reward model. Just predict next latent and
  regularize. That's it.

Architecture:
  Encoder:    ViT-Tiny (~5M params). Image (3,96,96) -> CLS token -> z in R^192.
  Projectors: TWO separate MLP+BatchNorm heads (matching repo):
              - Encoder projector: projects encoder CLS token into embedding space
              - Predictor projector: projects predictor output into same space
              MSE loss is computed between these projected representations.
  Predictor:  Transformer (~10M params). (z_t, a_t) -> z_hat_{t+1}.
              Action is projected to latent_dim and added as a token.
              Dropout 0.1 in predictor (critical for stability).
  Planning:   CEM in latent space. Cost = ||z_pred - z_goal||^2.
              Each frame is a single 192-dim token (vs 256 tokens for DINO-WM).
              Planning completes in ~1 second (48x faster than DINO-WM).

Training objective (the whole thing):
  L = L_pred + lambda * SIGReg(Z_all)
  L_pred = MSE(pred_projector(predictor(z_t, a_t)), enc_projector(encoder(o_{t+1})))
  SIGReg = Epps-Pulley normality test on ALL encoder embeddings (z_t AND z_{t+1})
  Note: SIGReg is applied to the FULL sequence of embeddings, not just current frame.

Key hyperparameters:
  - Encoder: ViT-Tiny (patch_size=8, embed_dim=192, depth=12, heads=3)
  - Latent dim: 192 (CLS token from ViT-Tiny)
  - Predictor: depth=6, heads=8, dim=512
  - Predictor dropout: 0.1 (critical)
  - SIGReg lambda: 0.1
  - SIGReg projections M: 512
  - Total params: ~15M
  - Training: single GPU, few hours

Components from Layer 1: CEMPlanner (adapted for goal-reaching)
Env:     PushT, Reacher, Two-Room, OGBench-Cube
Target:  Competitive with DINO-WM without pre-trained backbone
Compute: ~2-4 hours on single GPU
"""

import os
import math
import argparse
import subprocess
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
    parser = argparse.ArgumentParser(description="LucidWM: LeWM")

    # Standard
    parser.add_argument("--env-id", type=str, default="pusht")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--logdir", type=str, default="exp/lewm")

    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to pusht_expert_train.h5. If not provided, "
                             "downloads from HuggingFace automatically.")
    parser.add_argument("--hf-repo", type=str, default="quentinll/lewm-pusht",
                        help="HuggingFace dataset repo for auto-download")
    parser.add_argument("--hf-filename", type=str, default="pusht_expert_train.h5.zst",
                        help="Filename within the HF repo (zstd-compressed)")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--frameskip", type=int, default=1)

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigreg-lambda", type=float, default=0.1)
    parser.add_argument("--sigreg-projections", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=192)

    # Encoder
    parser.add_argument("--enc-patch-size", type=int, default=8)
    parser.add_argument("--enc-embed-dim", type=int, default=192)
    parser.add_argument("--enc-depth", type=int, default=12)
    parser.add_argument("--enc-heads", type=int, default=96)

    # Predictor
    parser.add_argument("--pred-dim", type=int, default=512)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=8)
    parser.add_argument("--pred-dropout", type=float, default=0.1)

    # Planning
    parser.add_argument("--plan-horizon", type=int, default=10)
    parser.add_argument("--cem-candidates", type=int, default=200)
    parser.add_argument("--cem-elites", type=int, default=20)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=2)

    parser.add_argument("--skip-train", action="store_true")

    return parser.parse_args()

# SIGReg
def sigreg(z: torch.Tensor, M: int) -> torch.Tensor:
    """
    SIGReg: enforce isotropic Gaussian distribution on latent embeddings.

    Uses the Cramer-Wold theorem: a distribution is Gaussian iff ALL its
    1D projections are Gaussian. We sample M random directions, project z
    onto each, and compute the Epps-Pulley test statistic measuring
    departure from normality.

    Args:
        z: (B, D) latent embeddings (batch of CLS tokens)
        M: number of random projection directions

    Returns:
        scalar: mean Epps-Pulley statistic (lower = more Gaussian)
    """

    B, D = z.shape
    device = z.device

    # Random projection directions on unit sphere
    directions = torch.randn(D, M, device=device)
    directions = F.normalize(directions, dim=0)  # (D, M)

    # Project: (B, D) @ (D, M) -> (B, M)
    projections = z @ directions  # (B, M)

    # Standardize each projection
    projections = (projections - projections.mean(dim=0)) / (projections.std(dim=0) + 1e-8)

    # Epps-Pulley test statistic for each projection
    # EP(x) = (2/n) * sum_i sum_j exp(-||x_i - x_j||^2 / 2) - sqrt(2) * (2/n) * sum_i exp(-x_i^2 / 4) + 1/sqrt(3)
    # Simplified batch version using pairwise distances
    ep_stats = []
    for m in range(M):
        x = projections[:, m]  # (B,)

        # Term 1: mean of exp(-|xi - xj|^2 / 2) over all pairs
        diffs = x.unsqueeze(0) - x.unsqueeze(1)  # (B, B)
        term1 = torch.exp(-0.5 * diffs.pow(2)).mean()

        # Term 2: mean of exp(-xi^2 / 4)
        term2 = torch.exp(-0.25 * x.pow(2)).mean()

        ep = term1 - math.sqrt(2) * term2 + 1.0 / math.sqrt(3)
        ep_stats.append(ep)

    return torch.stack(ep_stats).mean()

# Encoder: ViT-Tiny

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dim."""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int, in_channels: int = 3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """(B, 3, H, W) -> (B, num_patches, embed_dim)."""
        x = self.proj(x)                    # (B, E, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, E)

class LeWMEncoder(nn.Module):
    """
    ViT-Tiny encoder with CLS token.

    Image -> patch embedding -> ViT-Tiny (12 layers) -> CLS token.
    Output is a raw CLS vector in R^embed_dim. The projection into
    the embedding space is handled by a SEPARATE Projector module
    (matching the repo architecture).
    """

    def __init__(self, args):
        super(LeWMEncoder, self).__init__()

        self.embed_dim = args.enc_embed_dim
        self.patch_embed = PatchEmbedding(args.img_size, args.enc_patch_size, args.enc_embed_dim, 3)
        num_patches = self.patch_embed.num_patches

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + num_patches, self.embed_dim) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim, nhead=args.enc_heads,
                dim_feedforward=self.embed_dim * 4,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(args.enc_depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image to CLS token vector.

        Args: img (B, 3, H, W) in [0, 1]
        Returns: z (B, embed_dim) -- raw CLS token, NOT projected
        """
        B = img.shape[0]
        x = self.patch_embed(img)  # (B, N, E)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, E)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x[:, 0]  # (B, embed_dim) -- CLS token

class Projector(nn.Module):
    """MLP projection head with BatchNorm (matching repo).

    Two separate projectors are used in LeWM:
      1. Encoder projector: projects encoder CLS tokens into embedding space
      2. Predictor projector: projects predictor outputs into the same space

    The MSE prediction loss is computed between the outputs of these two
    projectors, NOT between raw encoder/predictor outputs. This is a
    standard JEPA pattern (also used in VICReg, Barlow Twins).

    Args:
        in_dim: input dimension
        proj_dim: projection output dimension (embedding space)
        hidden_dim: MLP hidden dimension
    """

    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> (B, proj_dim)."""
        return self.net(x)
    
class LeWMPredictor(nn.Module):
    """Transformer predictor for next-latent prediction.

    Takes current encoder output z_t and action a_t, predicts z_{t+1}.
    Action is projected to pred_dim and treated as an additional token.
    Dropout 0.1 throughout (critical for stability per paper).

    Output is in pred_dim space. The separate predictor Projector maps
    this to the embedding space where MSE is computed.

    Args:
        embed_dim: encoder output dimension. Default: 192.
        action_dim: action space dimension.
        pred_dim: transformer hidden dim. Default: 512.
        depth: transformer layers. Default: 6.
        num_heads: attention heads. Default: 8.
        dropout: dropout rate. Default: 0.1.
    """

    def __init__(self, args, action_dim=2):
        super().__init__()
        self.pred_dim = args.pred_dim
        self.embed_dim = args.enc_embed_dim

        # Project encoder output and action to predictor dim
        self.z_proj = nn.Linear(self.embed_dim, self.pred_dim)
        self.a_proj = nn.Linear(action_dim, self.pred_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.pred_dim, nhead=args.pred_heads,
                dim_feedforward=self.pred_dim * 4,
                activation="gelu", batch_first=True, norm_first=True,
                dropout=args.pred_dropout,
            )
            for _ in range(args.pred_depth)
        ])
        self.norm = nn.LayerNorm(self.pred_dim)
        self.dropout = nn.Dropout(args.pred_dropout)

        # Project back to embed_dim for chaining during planning
        self.out_proj = nn.Linear(self.pred_dim, self.embed_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent.

        Args:
            z: (B, embed_dim) encoder CLS token
            action: (B, action_dim) current action

        Returns:
            z_pred: (B, embed_dim) predicted next latent (chainable)
        """
        z_tok = self.z_proj(z).unsqueeze(1)        # (B, 1, pred_dim)
        a_tok = self.a_proj(action).unsqueeze(1)    # (B, 1, pred_dim)
        x = torch.cat([z_tok, a_tok], dim=1)        # (B, 2, pred_dim)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        out = self.dropout(x[:, 0])  # (B, pred_dim)
        return self.out_proj(out)     # (B, embed_dim)

    def forward_raw(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning pred_dim output (for projector during training).

        Args:
            z: (B, embed_dim)
            action: (B, action_dim)

        Returns:
            (B, pred_dim) -- raw transformer output before out_proj
        """
        z_tok = self.z_proj(z).unsqueeze(1)
        a_tok = self.a_proj(action).unsqueeze(1)
        x = torch.cat([z_tok, a_tok], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return self.dropout(x[:, 0])  # (B, pred_dim)

def train(encoder, predictor, enc_projector, pred_projector, dataset, args, device, logger):
    """Train LeWM end-to-end with the two-term objective.

    L = L_pred + lambda * SIGReg(Z_all)

    Where:
      L_pred = MSE(pred_projector(predictor(z_t, a_t)), enc_projector(encoder(o_{t+1})))
      SIGReg is applied to ALL encoder embeddings in the batch (z_t AND z_{t+1})

    Both encoder, predictor, and projectors are optimized jointly.
    No stop-gradient, no EMA, no frozen parameters.
    """
    has_gpu = torch.cuda.is_available()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=has_gpu, drop_last=True,
        worker_init_fn=LeWMH5Dataset.worker_init_fn,
        persistent_workers=True,
    )
    
    params = (list(encoder.parameters()) + list(predictor.parameters()) +
              list(enc_projector.parameters()) + list(pred_projector.parameters()))
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training LeWM: {args.epochs} epochs, {len(dataset)} samples")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    print(f"  Predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.1f}M params")
    print(f"  Enc projector: {sum(p.numel() for p in enc_projector.parameters())/1e3:.1f}K params")
    print(f"  Pred projector: {sum(p.numel() for p in pred_projector.parameters())/1e3:.1f}K params")
    print(f"  SIGReg lambda={args.sigreg_lambda}, projections={args.sigreg_projections}")

    step = 0
    for epoch in range(args.epochs):
        encoder.train()
        predictor.train()
        enc_projector.train()
        pred_projector.train()
        epoch_pred_loss = 0.0
        epoch_sig_loss = 0.0

        for batch in loader:
            obs_t = batch["obs_t"].to(device)          # (B, 3, H, W)
            obs_next = batch["obs_next"].to(device)    # (B, 3, H, W)
            action = batch["action"].to(device)         # (B, A)

            # Encode both frames (gradients flow through encoder)
            z_t = encoder(obs_t)          # (B, embed_dim)
            z_next = encoder(obs_next)    # (B, embed_dim)

            # Predict next latent (raw pred_dim output for projector)
            z_pred_raw = predictor.forward_raw(z_t, action)  # (B, pred_dim)

            # Project both into embedding space for MSE comparison
            z_next_proj = enc_projector(z_next)    # (B, proj_dim)
            z_pred_proj = pred_projector(z_pred_raw)  # (B, proj_dim)

            # L_pred: MSE in projected embedding space
            pred_loss = F.mse_loss(z_pred_proj, z_next_proj)

            # SIGReg: applied to FULL sequence of encoder embeddings
            # Concatenate z_t and z_next to regularize all representations
            z_all = torch.cat([z_t, z_next], dim=0)  # (2B, embed_dim)
            sig_loss = sigreg(z_all, M=args.sigreg_projections)

            # Total loss
            loss = pred_loss + args.sigreg_lambda * sig_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_pred_loss += pred_loss.item()
            epoch_sig_loss += sig_loss.item()
            step += 1

        scheduler.step()
        avg_pred = epoch_pred_loss / len(loader)
        avg_sig = epoch_sig_loss / len(loader)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{args.epochs}  pred={avg_pred:.6f}  "
                  f"sigreg={avg_sig:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
        logger.log({"train/pred_loss": avg_pred, "train/sigreg": avg_sig}, step=step)

    return encoder, predictor, enc_projector, pred_projector

@torch.no_grad()
def plan_cem(encoder, predictor, obs_current, goal_z,
             action_dim, args, device):
    """CEM planning in LeWM latent space.

    Cost = ||z_predicted_at_horizon - z_goal||^2.
    Because each frame is a single 192-dim vector (not 256 patches),
    this is ~48x faster than DINO-WM planning.

    Args:
        encoder: trained LeWM encoder
        predictor: trained LeWM predictor
        obs_current: (3, H, W) current observation
        goal_z: (latent_dim,) goal latent embedding
        action_dim: action space dimension
        args: planning hyperparameters
        device: torch device

    Returns:
        action: (A,) first action of best sequence
    """
    z = encoder(obs_current.unsqueeze(0).to(device))  # (1, D)
    goal_z = goal_z.unsqueeze(0).to(device)            # (1, D)

    horizon = args.plan_horizon
    n_cand = args.cem_candidates
    n_elite = args.cem_elites

    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    for _ in range(args.cem_iterations):
        actions = (mean + std * torch.randn(n_cand, horizon, action_dim, device=device))
        actions = actions.clamp(-1, 1)

        # Rollout each candidate
        z_expanded = z.expand(n_cand, -1)  # (N, D)
        for t in range(horizon):
            z_expanded = predictor(z_expanded, actions[:, t])

        # Cost: L2 to goal
        costs = ((z_expanded - goal_z) ** 2).sum(dim=-1)  # (N,)

        # Elites
        elite_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.01)

    return mean[0].cpu().numpy()

# Downloading Dataset
def download_h5_dataset(repo_id, filename, cache_dir="data"):
    """Download and decompress the LeWM dataset from HuggingFace.

    The file is zstd-compressed (.h5.zst). Downloads via huggingface_hub
    and decompresses to produce a plain .h5 file.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)

    h5_name = filename[:-4] if filename.endswith(".zst") else filename
    h5_path = os.path.join(cache_dir, h5_name)

    if os.path.exists(h5_path):
        print(f"Dataset already cached: {h5_path}")
        return h5_path

    print(f"Downloading {filename} from {repo_id}...")
    downloaded = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        cache_dir=os.path.join(cache_dir, ".hf_cache"),
    )

    if filename.endswith(".zst"):
        print(f"Decompressing -> {h5_path}")
        try:
            subprocess.run(["zstd", "-d", downloaded, "-o", h5_path],
                           check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with open(downloaded, "rb") as ifh, open(h5_path, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
        print(f"Decompressed: {h5_path}")
    else:
        import shutil
        shutil.copy2(downloaded, h5_path)

    return h5_path

class LeWMH5Dataset(Dataset):
    """Dataset loader for the official LeWM HDF5 format.
 
    The stable-worldmodel HDF5 layout stores all episodes concatenated:
        pixels:    (Total_Steps, H, W, C) uint8
        action:    (Total_Steps, Action_Dim) float32
        ep_len:    (Num_Episodes,) int32    — length of each episode
        ep_offset: (Num_Episodes,) int64    — start index of each episode
 
    This dataset yields (obs_t, action_t, obs_{t+1}) transition triples,
    respecting episode boundaries so we never cross episodes.
 
    Args:
        h5_path: path to the .h5 file
        img_size: target image size (resized if needed)
        frameskip: number of steps between obs_t and obs_{t+1}
    """
 
    def __init__(self, h5_path: str, img_size: int = 96, frameskip: int = 1):
        self._ensure_hdf5plugin()
        import h5py
        
        self.h5_path = h5_path
        self.img_size = img_size
        self.frameskip = frameskip

        # Read metadata and small arrays into memory
        with h5py.File(h5_path, "r") as f:
            self.ep_len = f["ep_len"][:]
            self.ep_offset = f["ep_offset"][:]
            self.actions = f["action"][:]  # small, fits in RAM

            pixels_shape = f["pixels"].shape
            self.stored_h, self.stored_w = pixels_shape[1], pixels_shape[2]

            # Check what compression the pixels dataset uses
            pix_dset = f["pixels"]
            filter_ids = getattr(pix_dset, "filter_ids", ())
            if filter_ids:
                print(f"  Pixels compression filters: {filter_ids}")
                print(f"  (hdf5plugin is {'loaded' if 'hdf5plugin' in dir() else 'NOT loaded'})")

            print(f"Loaded H5: {h5_path}")
            print(f"  Episodes: {len(self.ep_len)}, Total steps: {pixels_shape[0]}")
            print(f"  Pixels: {pixels_shape[1:]}, Actions: {self.actions.shape[1:]}")

        # # Build index of valid transitions (respecting episode boundaries)
        # self.index = []
        # for ep_idx in range(len(self.ep_len)):
        #     offset = int(self.ep_offset[ep_idx])
        #     length = int(self.ep_len[ep_idx])
        #     for t in range(length - frameskip):
        #         self.index.append(offset + t)

        # print(f"  Valid transitions: {len(self.index)}")

        n_eps = len(self.ep_len) if 100 is None else min(100, len(self.ep_len))
        self.index = []
        for ep_idx in range(n_eps):
            offset = int(self.ep_offset[ep_idx])
            length = int(self.ep_len[ep_idx])
            for t in range(length - frameskip):
                self.index.append(offset + t)

        print(f"  Using {n_eps}/{len(self.ep_len)} episodes, {len(self.index)} transitions")

        # Per-worker HDF5 handle (set in worker_init_fn)
        self._h5 = None

    def _get_h5(self):
        """Get HDF5 handle, opening if needed (num_workers=0 fallback)."""
        if self._h5 is None:
            self._ensure_hdf5plugin()
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def open_h5(self):
        """Open a fresh HDF5 handle. Called by worker_init_fn."""
        self._ensure_hdf5plugin()
        import h5py
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
        self._h5 = h5py.File(self.h5_path, "r")

    @staticmethod
    def _ensure_hdf5plugin():
        """Ensure HDF5 plugin path is set and hdf5plugin filters are registered."""
        import os
        os.environ.setdefault("HDF5_PLUGIN_PATH", "")
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass

    @staticmethod
    def worker_init_fn(worker_id):
        """DataLoader worker_init_fn: opens a per-worker HDF5 handle."""
        import os
        os.environ.setdefault("HDF5_PLUGIN_PATH", "")
        try:
            import hdf5plugin  # noqa: F401 -- must register in EACH worker process
        except ImportError:
            pass
        import torch.utils.data as data
        worker_info = data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
            if isinstance(dataset, LeWMH5Dataset):
                dataset.open_h5()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f = self._get_h5()
        global_t = self.index[idx]
        t_next = global_t + self.frameskip

        # Read single frames from HDF5 (decompression handled by hdf5plugin)
        obs_t = self._preprocess(f["pixels"][global_t])
        obs_next = self._preprocess(f["pixels"][t_next])
        action = self.actions[global_t]

        return {
            "obs_t": obs_t,
            "obs_next": obs_next,
            "action": torch.from_numpy(action.copy()),
        }

    def _preprocess(self, img):
        """(H, W, C) uint8 -> (3, img_size, img_size) float32 in [0, 1]."""
        import cv2
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(project=args.wandb_project, name=f"lewm_{args.env_id}_s{args.seed}",
                    config=vars(args), use_wandb=args.track)

    # Build model components
    encoder = LeWMEncoder(args).to(device)
    predictor = LeWMPredictor(args, action_dim=args.action_dim).to(device)

    # Two separate projectors (matching repo architecture)
    proj_dim = args.latent_dim  # embedding space dimension
    enc_projector = Projector(encoder.embed_dim, proj_dim).to(device)
    pred_projector = Projector(predictor.pred_dim, proj_dim).to(device)

    if not args.skip_train:
        print(f"\n{'='*60}\nTraining LeWM\n{'='*60}")
        # Resolve dataset path: use provided path or auto-download from HF
        if args.data_path is not None:
            h5_path = args.data_path
        else:
            h5_path = download_h5_dataset(
                repo_id=args.hf_repo,
                filename=args.hf_filename,
                cache_dir=os.path.join(args.logdir, "data"),
            )
 
        dataset = LeWMH5Dataset(
            h5_path=h5_path,
            img_size=args.img_size,
            frameskip=args.frameskip,
        )

        encoder, predictor, enc_projector, pred_projector = train(
            encoder, predictor, enc_projector, pred_projector,
            dataset, args, device, logger,
        )

        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "enc_projector": enc_projector.state_dict(),
            "pred_projector": pred_projector.state_dict(),
        }, os.path.join(args.logdir, "lewm.pt"))
    else:
        ckpt = torch.load(os.path.join(args.logdir, "lewm.pt"),
                          map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        enc_projector.load_state_dict(ckpt["enc_projector"])
        pred_projector.load_state_dict(ckpt["pred_projector"])

    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [encoder, predictor, enc_projector, pred_projector]
    )
    print(f"\nLeWM trained. Total: {total_params/1e6:.1f}M params")
    print("Use plan_cem() for CEM goal-reaching in latent space.")
    print("Note: planning uses encoder directly (not projectors).")

    logger.close()
    print("Done.")
