"""DINO-WM: Frozen DINOv2 encoder + causal ViT transition model"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = (img - self.mean) / self.std
        features = self.backbone.forward_features(img)
        return features["x_norm_patchtokens"]

def generate_frame_causal_mask(num_patches: int, num_frames: int) -> torch.Tensor:
    ones = torch.ones(num_patches, num_patches)
    zeros = torch.zeros(num_patches, num_patches)
    rows = []
    for i in range(num_frames):
        row = torch.cat([ones] * (i + 1) + [zeros] * (num_frames - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0)
    return mask.unsqueeze(0).unsqueeze(0)

class ActionEmbedding(nn.Module):
    def __init__(self, action_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action).unsqueeze(1)

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
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, num_patches: int = 256, num_frames: int = 3):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.register_buffer("mask", generate_frame_causal_mask(num_patches, num_frames))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, T, self.heads, -1).permute(0, 2, 1, 3), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(out)

class TransitionViT(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, num_heads: int, mlp_dim: int, num_patches: int, context_len: int, action_dim: int = 2, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.context_len = context_len
        total_tokens = context_len * num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, embedding_dim) * 0.02)
        self.action_embedding = ActionEmbedding(action_dim, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(embedding_dim, heads=num_heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches, num_frames=context_len),
                FeedForward(embedding_dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])
        self.pred_head = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
        )

    def forward(self, z_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        B, H, N, E = z_seq.shape
        x = z_seq.clone()
        for t in range(H):
            a_emb = self.action_embedding(a_seq[:, t])
            x[:, t] = x[:, t] + a_emb
        x = x.reshape(B, H * N, E)
        n = x.shape[1]
        x = x + self.pos_embedding[:, :n]
        x = self.dropout_layer(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.norm(x)
        x_last = x[:, -N:]
        return self.pred_head(x_last)
