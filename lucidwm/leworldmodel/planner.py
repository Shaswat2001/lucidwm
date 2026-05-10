"""LeWorldModel: CEM Planner"""

import torch

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