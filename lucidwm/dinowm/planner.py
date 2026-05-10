"""DINO-WM: CEM planner in DINOv2 latent space"""
from __future__ import annotations
import numpy as np
import torch
from .model import DINOv2Encoder, TransitionViT

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
    H = args.context_len
    horizon = args.plan_horizon
    n_cand = args.cem_candidates
    n_elite = args.cem_elites
    n_iter = args.cem_iterations
    z_frames = []
    for i in range(current_frames.shape[0]):
        z_i = encoder(current_frames[i:i+1].to(device))
        z_frames.append(z_i)
    z_context = torch.stack(z_frames, dim=1)
    goal_z = goal_z.unsqueeze(0).to(device)
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)
    for _ in range(n_iter):
        noise = torch.randn(n_cand, horizon, action_dim, device=device)
        actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1, 1)
        z_ctx = z_context.expand(n_cand, -1, -1, -1).clone()
        for t in range(horizon):
            a_ctx = actions[:, max(0, t - H + 1):t + 1]
            if a_ctx.shape[1] < H:
                pad = torch.zeros(n_cand, H - a_ctx.shape[1], action_dim, device=device)
                a_ctx = torch.cat([pad, a_ctx], dim=1)
            z_pred = transition(z_ctx, a_ctx)
            z_ctx = torch.cat([z_ctx[:, 1:], z_pred.unsqueeze(1)], dim=1)
        costs = ((z_ctx[:, -1] - goal_z) ** 2).sum(dim=(-2, -1))
        elite_idx = costs.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.01)
    return mean[0].cpu().numpy()
