"""LeWorldModel: Encoder"""
from __future__ import annotations
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, in_channels: int = 3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)

class LeWMEncoder(nn.Module):
    """ViT-Tiny encoder producing a CLS token representation."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.enc_embed_dim
        self.patch_embed = PatchEmbedding(args.img_size, args.enc_patch_size, args.enc_embed_dim, 3)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, self.embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=args.enc_heads,
                dim_feedforward=self.embed_dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(args.enc_depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        x = self.patch_embed(img)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]
