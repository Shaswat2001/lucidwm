"""TD-MPC2: MPPI planner"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import TDMPC2Model

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
    z = z.expand(args.mppi_n, -1)
    if prev_mean is not None:
        mean = torch.cat([prev_mean[1:], prev_mean[-1:]], dim=0)
    else:
        mean = torch.zeros(horizon, action_dim, device=device)
    std = 2.0 * torch.ones(horizon, action_dim, device=device)
    for _ in range(args.mppi_iter):
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            args.mppi_n, horizon, action_dim, device=device
        )
        actions = actions.clamp(-1, 1)
        n_policy = args.mppi_n // 2
        s = z[:n_policy]
        for t in range(horizon):
            pi_action, _ = model.policy(s)
            actions[:n_policy, t] = pi_action
            s = model.next_state(s, pi_action)
        s = z.clone()
        total_reward = torch.zeros(args.mppi_n, device=device)
        for t in range(horizon):
            r_logits = model.reward(s, actions[:, t])
            total_reward += model.two_hot.decode(r_logits).squeeze(-1)
            s = model.next_state(s, actions[:, t])
        final_action, _ = model.policy(s)
        q_logits = model.q_values(s, final_action)
        idx = torch.randperm(args.num_q)[:2]
        q1 = model.two_hot.decode(q_logits[idx[0]]).squeeze(-1)
        q2 = model.two_hot.decode(q_logits[idx[1]]).squeeze(-1)
        total_reward += torch.min(q1, q2)
        weights = F.softmax(total_reward / args.mppi_temp, dim=0)
        mean = (weights[:, None, None] * actions).sum(dim=0)
        std = (std * 0.5).clamp(min=0.05)
    return mean[0], mean
