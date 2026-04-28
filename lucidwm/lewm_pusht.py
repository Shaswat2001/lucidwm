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
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to pusht_expert_train.h5. If not provided, downloads from HuggingFace automatically.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="quentinll/lewm-pusht",
        help="HuggingFace dataset repo for auto-download",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="pusht_expert_train.h5.zst",
        help="Filename within the HF repo (zstd-compressed)",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--frameskip", type=int, default=1)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--sigreg-lambda", type=float, default=0.09)
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--sigreg-knots", type=int, default=17)

    # Encoder
    parser.add_argument("--enc-patch-size", type=int, default=14)
    parser.add_argument("--enc-embed-dim", type=int, default=192)
    parser.add_argument("--enc-depth", type=int, default=12)
    parser.add_argument("--enc-heads", type=int, default=3)

    # Action encoder / predictor
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--action-smoothed-dim", type=int, default=16)
    parser.add_argument("--action-emb-dim", type=int, default=192)
    parser.add_argument("--pred-dim", type=int, default=512)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=16)
    parser.add_argument("--pred-dim-head", type=int, default=64)
    parser.add_argument("--pred-mlp-dim", type=int, default=2048)
    parser.add_argument("--pred-dropout", type=float, default=0.1)
    parser.add_argument("--pred-emb-dropout", type=float, default=0.0)

    # Planning
    parser.add_argument("--plan-horizon", type=int, default=10)
    parser.add_argument("--cem-candidates", type=int, default=200)
    parser.add_argument("--cem-elites", type=int, default=20)
    parser.add_argument("--cem-iterations", type=int, default=10)

    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()

class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer from the official LeWM release."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D)."""
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a / (a.norm(p=2, dim=0, keepdim=True) + 1e-8)

        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, in_channels: int = 3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class LeWMEncoder(nn.Module):
    """Small ViT encoder producing a CLS token representation."""

    def __init__(self, args):
        super(LeWMEncoder, self).__init__()
        self.embed_dim = args.enc_embed_dim
        self.patch_embed = PatchEmbedding(args.img_size, args.enc_patch_size, args.enc_embed_dim, 3)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, self.embed_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=args.enc_heads,
                    dim_feedforward=self.embed_dim * 4,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
             
                for _ in range(args.enc_depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        x = self.patch_embed(img)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

class Projector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ActionEncoder(nn.Module):
    """Official-style action embedder with a 1x1 temporal smoothing conv."""

    def __init__(
        self,
        input_dim: int,
        smoothed_dim: int,
        emb_dim: int,
        mlp_scale: int = 4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

class FeedForward(nn.Module):
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
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        b, t, _ = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            tensor.view(b, t, self.heads, self.dim_head).permute(0, 2, 1, 3)
            for tensor in qkv
        ]
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.permute(0, 2, 1, 3).reshape(b, t, self.inner_dim)
        return self.to_out(out)

class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class LeWMPredictor(nn.Module):
    """Autoregressive predictor over a history window of projected embeddings."""

    def __init__(self, args):
        super().__init__()
        self.num_frames = args.history_size
        self.input_dim = args.latent_dim
        self.hidden_dim = args.pred_dim
        self.output_dim = args.pred_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_frames, self.input_dim) * 0.02)
        self.emb_dropout = nn.Dropout(args.pred_emb_dropout)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.cond_proj = nn.Linear(args.action_emb_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                ConditionalBlock(
                    dim=self.hidden_dim,
                    heads=args.pred_heads,
                    dim_head=args.pred_dim_head,
                    mlp_dim=args.pred_mlp_dim,
                    dropout=args.pred_dropout,
                )
                for _ in range(args.pred_depth)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        t = emb.size(1)
        x = emb + self.pos_embedding[:, :t]
        x = self.emb_dropout(x)
        x = self.input_proj(x)
        c = self.cond_proj(act_emb)
        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)

def train(
    encoder,
    action_encoder,
    predictor,
    enc_projector,
    pred_projector,
    sigreg,
    dataset,
    args,
    device,
    logger,
):
    has_gpu = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=has_gpu,
        drop_last=True,
        worker_init_fn=LeWMH5Dataset.worker_init_fn,
        persistent_workers=True,
    )

    params = (
        list(encoder.parameters())
        + list(action_encoder.parameters())
        + list(predictor.parameters())
        + list(enc_projector.parameters())
        + list(pred_projector.parameters())
    )
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training LeWM: {args.epochs} epochs, {len(dataset)} samples")
    print(f"  History size: {args.history_size}")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    print(f"  Action encoder: {sum(p.numel() for p in action_encoder.parameters())/1e3:.1f}K params")
    print(f"  Predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.1f}M params")
    print(f"  Enc projector: {sum(p.numel() for p in enc_projector.parameters())/1e3:.1f}K params")
    print(f"  Pred projector: {sum(p.numel() for p in pred_projector.parameters())/1e3:.1f}K params")

    step = 0
    for epoch in range(args.epochs):
        encoder.train()
        action_encoder.train()
        predictor.train()
        enc_projector.train()
        pred_projector.train()

        epoch_pred_loss = 0.0
        epoch_sig_loss = 0.0

        for batch in loader:
            obs = batch["obs"].to(device)          # (B, H+1, 3, img, img)
            action = batch["action"].to(device)    # (B, H, A)

            bsz, seq_len, c, h_img, w_img = obs.shape
            flat_obs = obs.reshape(bsz * seq_len, c, h_img, w_img)

            z_raw = encoder(flat_obs).reshape(bsz, seq_len, -1)
            emb = enc_projector(z_raw.reshape(bsz * seq_len, -1)).reshape(bsz, seq_len, -1)
            act_emb = action_encoder(action)

            pred_raw = predictor(emb[:, :-1], act_emb)
            pred = pred_projector(pred_raw.reshape(bsz * args.history_size, -1)).reshape(
                bsz, args.history_size, -1
            )
            target = emb[:, 1:]

            pred_loss = F.mse_loss(pred, target)
            sig_loss = sigreg(emb.transpose(0, 1))
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
            print(
                f"  Epoch {epoch}/{args.epochs}  pred={avg_pred:.6f}  "
                f"sigreg={avg_sig:.6f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )
        logger.log({"train/pred_loss": avg_pred, "train/sigreg": avg_sig}, step=step)

    return encoder, action_encoder, predictor, enc_projector, pred_projector

@torch.no_grad()
def plan_cem(
    encoder,
    action_encoder,
    predictor,
    enc_projector,
    pred_projector,
    obs_history,
    goal_z,
    action_dim,
    args,
    device,
    action_history=None,
):
    """CEM planning using LeWM's history-conditioned autoregressive latent model."""

    history_size = args.history_size
    if obs_history.dim() == 3:
        obs_history = obs_history.unsqueeze(0)
    obs_history = obs_history.to(device)

    if obs_history.size(1) != history_size:
        raise ValueError(f"obs_history must have exactly {history_size} frames")

    bsz, hist_len, c, h_img, w_img = obs_history.shape
    z_raw = encoder(obs_history.reshape(bsz * hist_len, c, h_img, w_img)).reshape(bsz, hist_len, -1)
    emb_hist = enc_projector(z_raw.reshape(bsz * hist_len, -1)).reshape(bsz, hist_len, -1)

    if action_history is None:
        action_history = torch.zeros(bsz, history_size, action_dim, device=device)
    else:
        if action_history.dim() == 2:
            action_history = action_history.unsqueeze(0)
        action_history = action_history.to(device)
        if action_history.size(1) != history_size:
            raise ValueError(f"action_history must have exactly {history_size} actions")

    goal_z = goal_z.unsqueeze(0).to(device) if goal_z.dim() == 1 else goal_z.to(device)

    horizon = args.plan_horizon
    n_cand = args.cem_candidates
    n_elite = args.cem_elites

    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    for _ in range(args.cem_iterations):
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_cand, horizon, action_dim, device=device)
        actions = actions.clamp(-1, 1)

        emb_roll = emb_hist.expand(n_cand, -1, -1).clone()
        act_roll = action_history.expand(n_cand, -1, -1).clone()

        for t in range(horizon):
            act_emb = action_encoder(act_roll[:, -history_size:])
            pred_raw = predictor(emb_roll[:, -history_size:], act_emb)
            pred = pred_projector(pred_raw[:, -1])
            emb_roll = torch.cat([emb_roll, pred.unsqueeze(1)], dim=1)
            act_roll = torch.cat([act_roll, actions[:, t : t + 1]], dim=1)

        costs = ((emb_roll[:, -1] - goal_z.expand(n_cand, -1)) ** 2).sum(dim=-1)
        elite_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.01)

    return mean[0].cpu().numpy()

def download_h5_dataset(repo_id, filename, cache_dir="data"):
    """Download and decompress the LeWM dataset from HuggingFace."""
    from huggingface_hub import hf_hub_download

    os.makedirs(cache_dir, exist_ok=True)

    h5_name = filename[:-4] if filename.endswith(".zst") else filename
    h5_path = os.path.join(cache_dir, h5_name)

    if os.path.exists(h5_path):
        print(f"Dataset already cached: {h5_path}")
        return h5_path

    print(f"Downloading {filename} from {repo_id}...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=os.path.join(cache_dir, ".hf_cache"),
    )

    if filename.endswith(".zst"):
        print(f"Decompressing -> {h5_path}")
        try:
            subprocess.run(["zstd", "-d", downloaded, "-o", h5_path], check=True, capture_output=True)
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
    """Dataset loader that returns LeWM history windows."""

    def __init__(self, h5_path: str, img_size: int = 224, frameskip: int = 1, history_size: int = 3):
        self._ensure_hdf5plugin()
        import h5py

        self.h5_path = h5_path
        self.img_size = img_size
        self.frameskip = frameskip
        self.history_size = history_size

        with h5py.File(h5_path, "r") as f:
            self.ep_len = f["ep_len"][:]
            self.ep_offset = f["ep_offset"][:]
            self.actions = f["action"][:]

            pixels_shape = f["pixels"].shape
            self.stored_h, self.stored_w = pixels_shape[1], pixels_shape[2]

            print(f"Loaded H5: {h5_path}")
            print(f"  Episodes: {len(self.ep_len)}, Total steps: {pixels_shape[0]}")
            print(f"  Pixels: {pixels_shape[1:]}, Actions: {self.actions.shape[1:]}")

        self.index = []
        stride = self.frameskip
        max_offset = self.history_size * stride
        for ep_idx in range(len(self.ep_len)):
            offset = int(self.ep_offset[ep_idx])
            length = int(self.ep_len[ep_idx])
            for t in range(length - max_offset):
                self.index.append(offset + t)

        print(f"  Using all {len(self.ep_len)} episodes, {len(self.index)} history windows")
        self._h5 = None

    def _get_h5(self):
        if self._h5 is None:
            self._ensure_hdf5plugin()
            import h5py

            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def open_h5(self):
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
        os.environ.setdefault("HDF5_PLUGIN_PATH", "")
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass

    @staticmethod
    def worker_init_fn(worker_id):
        os.environ.setdefault("HDF5_PLUGIN_PATH", "")
        try:
            import hdf5plugin  # noqa: F401
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
        step = self.frameskip

        frame_ids = [global_t + i * step for i in range(self.history_size + 1)]
        action_ids = [global_t + i * step for i in range(self.history_size)]

        obs = torch.stack([self._preprocess(f["pixels"][frame_id]) for frame_id in frame_ids], dim=0)
        action = torch.from_numpy(self.actions[action_ids].copy()).float()

        return {
            "obs": obs,
            "action": action,
        }

    def _preprocess(self, img):
        import cv2

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(
        project=args.wandb_project,
        name=f"lewm_{args.env_id}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    encoder = LeWMEncoder(args).to(device)
    action_encoder = ActionEncoder(input_dim=args.action_dim, smoothed_dim=args.action_smoothed_dim, emb_dim=args.action_emb_dim).to(device)
    predictor = LeWMPredictor(args).to(device)
    enc_projector = Projector(encoder.embed_dim, args.latent_dim).to(device)
    pred_projector = Projector(args.pred_dim, args.latent_dim).to(device)
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_projections).to(device)

    if not args.skip_train:
        print(f"\n{'='*60}\nTraining LeWM\n{'='*60}")
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
            history_size=args.history_size,
        )

        encoder, action_encoder, predictor, enc_projector, pred_projector = train(
            encoder,
            action_encoder,
            predictor,
            enc_projector,
            pred_projector,
            sigreg,
            dataset,
            args,
            device,
            logger,
        )

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "action_encoder": action_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "enc_projector": enc_projector.state_dict(),
                "pred_projector": pred_projector.state_dict(),
            },
            os.path.join(args.logdir, "lewm.pt"),
        )
    else:
        ckpt = torch.load(os.path.join(args.logdir, "lewm.pt"), map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        action_encoder.load_state_dict(ckpt["action_encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        enc_projector.load_state_dict(ckpt["enc_projector"])
        pred_projector.load_state_dict(ckpt["pred_projector"])

    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [encoder, action_encoder, predictor, enc_projector, pred_projector]
    )
    print(f"\nLeWM ready. Total: {total_params/1e6:.1f}M params")
    print("Use plan_cem() with obs_history and action_history for latent goal-reaching.")

    logger.close()
    print("Done.")
