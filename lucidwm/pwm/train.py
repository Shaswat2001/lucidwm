"""PWM: World model pre-training and policy update"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .model import PWMModel, PWMPolicy, PWMCriticEnsemble

def pretrain_world_model(world_model: PWMModel, dataset, args, device, logger):
    optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr)
    H = args.wm_horizon
    seq_len = H + 1
    print(f"Pre-training world model: {args.wm_epochs} epochs, H={H}")
    step = 0
    for epoch in range(args.wm_epochs):
        world_model.train()
        epoch_loss = 0.0
        n_batches = max(1, min(dataset.total_transitions // (args.wm_batch_size * seq_len), 1000))
        for _ in range(n_batches):
            batch = dataset.sample(args.wm_batch_size, seq_len=seq_len)
            obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
            action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
            reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
            z = world_model.encode(obs[:, 0])
            total_loss = torch.tensor(0.0, device=device)
            for t in range(H):
                z_pred = world_model.next_state(z, action[:, t])
                with torch.no_grad():
                    z_target = world_model.encode(obs[:, t + 1])
                consistency = F.mse_loss(z_pred, z_target)
                r_logits = world_model.reward(z, action[:, t])
                r_target = world_model.reward_encode(reward[:, t])
                reward_loss = -(r_target * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()
                total_loss = total_loss + (args.discount ** t) * (consistency + reward_loss)
                z = z_pred
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), 20.0)
            optimizer.step()
            epoch_loss += total_loss.item()
            step += 1
        avg = epoch_loss / n_batches
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{args.wm_epochs}  loss={avg:.4f}")
        logger.log({"wm/loss": avg}, step=step)
    return world_model

def pretrain_world_model_minari(world_model: PWMModel, dataset, args, device, logger):
    optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr)
    H = args.wm_horizon
    seq_len = H + 1
    print(f"Pre-training world model on Minari: {args.wm_epochs} steps, H={H}")
    for step in range(args.wm_epochs):
        batch = dataset.sample(args.wm_batch_size, seq_len=seq_len)
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
        action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
        reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
        z = world_model.encode(obs[:, 0])
        total_loss = torch.tensor(0.0, device=device)
        for t in range(H):
            z_pred = world_model.next_state(z, action[:, t])
            with torch.no_grad():
                z_target = world_model.encode(obs[:, t + 1])
            consistency = F.mse_loss(z_pred, z_target)
            r_logits = world_model.reward(z, action[:, t])
            r_target = world_model.reward_encode(reward[:, t])
            reward_loss = -(r_target * F.log_softmax(r_logits, dim=-1)).sum(-1).mean()
            total_loss = total_loss + (args.discount ** t) * (consistency + reward_loss)
            z = z_pred
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(world_model.parameters(), 20.0)
        optimizer.step()
        if step % 10 == 0:
            print(f"  Step {step}/{args.wm_epochs}  loss={total_loss.item():.4f}")
            logger.log({"wm/loss": total_loss.item()}, step=step)
    return world_model

def td_lambda_targets(rewards: torch.Tensor, values: torch.Tensor, discount: float, lmbd: float) -> torch.Tensor:
    B, H = rewards.shape
    targets = torch.zeros_like(rewards)
    next_value = values[:, -1]
    for t in reversed(range(H)):
        if t == H - 1:
            next_target = next_value
        else:
            next_target = (1 - lmbd) * values[:, t + 1] + lmbd * targets[:, t + 1]
        targets[:, t] = rewards[:, t] + discount * next_target
    return targets

def policy_update(
    world_model: PWMModel,
    policy: PWMPolicy,
    critic: PWMCriticEnsemble,
    policy_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    start_states: torch.Tensor,
    args,
) -> dict[str, float]:
    H = args.policy_horizon
    gamma = args.discount
    with torch.no_grad():
        z = world_model.encode(start_states)
    imagined_rewards = []
    imagined_values = []
    z_current = z
    for h in range(H):
        with torch.no_grad():
            v = critic.mean_value(z_current)
        imagined_values.append(v)
        action, _ = policy(z_current)
        reward = world_model.reward_scalar(z_current, action)
        imagined_rewards.append(reward)
        z_current = world_model.next_state(z_current, action)
    with torch.no_grad():
        terminal_value = critic.mean_value(z_current)
    rewards_tensor = torch.stack(imagined_rewards, dim=1)
    values_tensor = torch.stack(imagined_values, dim=1)
    discounts = gamma ** torch.arange(H, device=z.device, dtype=torch.float32)
    actor_loss = -(rewards_tensor * discounts).sum(dim=1).mean()
    actor_loss -= (gamma ** H) * terminal_value.mean()
    policy_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    policy_optimizer.step()
    with torch.no_grad():
        z_c = world_model.encode(start_states)
        rew_for_critic = []
        for h in range(H):
            a, _ = policy(z_c)
            r = world_model.reward_scalar(z_c, a)
            rew_for_critic.append(r)
            z_c = world_model.next_state(z_c, a)
        rew_critic = torch.stack(rew_for_critic, dim=1)
        terminal_v = critic.mean_value(z_c)
    z_c2 = world_model.encode(start_states).detach()
    val_preds = []
    for h in range(H):
        with torch.no_grad():
            a, _ = policy(z_c2)
        v = critic.mean_value(z_c2)
        val_preds.append(v)
        with torch.no_grad():
            z_c2 = world_model.next_state(z_c2, a)
    val_tensor = torch.stack(val_preds, dim=1)
    with torch.no_grad():
        all_values = torch.cat([val_tensor.detach(), terminal_v.unsqueeze(1)], dim=1)
        targets = td_lambda_targets(rew_critic, all_values[:, :-1], gamma, args.lmbd)
    critic_loss = F.mse_loss(val_tensor, targets)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_optimizer.step()
    return {
        "losses/actor": actor_loss.item(),
        "losses/critic": critic_loss.item(),
        "algo/mean_imagined_reward": rewards_tensor.detach().mean().item(),
        "algo/mean_value": values_tensor.mean().item(),
    }
