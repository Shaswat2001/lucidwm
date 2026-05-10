"""LeWorldModel: Predictor Model"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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