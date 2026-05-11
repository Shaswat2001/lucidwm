"""PWM: World model pre-training and policy update"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .model import PWMModel, PWMPolicy, PWMCriticEnsemble

class RunningMeanStd:
    """Welford online mean/variance estimator for return normalization."""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch = x.detach().float()
        n = batch.numel()
        if n == 0:
            return
        b_mean = batch.mean().item()
        b_var = batch.var().item() if n > 1 else 0.0
        delta = b_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m_a = self.var * self.count
        m_b = b_var * n
        self.var = (m_a + m_b + delta ** 2 * self.count * n / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return self.var ** 0.5

def pretrain_world_model(world_model: PWMModel, dataset, args, device, logger, ckpt_dir=None):
    optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr, betas=(0.7, 0.95))
    H = args.wm_horizon
    seq_len = H + 1
    save_freq = getattr(args, "save_freq", 0)
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
                r_hat = world_model.reward(z, action[:, t])      # (B,)
                reward_loss = F.mse_loss(r_hat, reward[:, t])
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
        if ckpt_dir is not None and save_freq > 0 and (epoch + 1) % save_freq == 0:
            path = ckpt_dir / f"world_model_epoch{epoch + 1:06d}.pt"
            torch.save({"epoch": epoch + 1, "step": step, "world_model": world_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": {"obs_dim": world_model.obs_dim, "action_dim": world_model.action_dim}}, path)
            print(f"  Saved WM checkpoint → {path.name}")
    return world_model

def pretrain_world_model_minari(world_model: PWMModel, dataset, args, device, logger, ckpt_dir=None):
    optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr, betas=(0.7, 0.95))
    H = args.wm_horizon
    seq_len = H + 1
    save_freq = getattr(args, "save_freq", 0)
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
            r_hat = world_model.reward(z, action[:, t])          # (B,)
            reward_loss = F.mse_loss(r_hat, reward[:, t])
            total_loss += (args.discount ** t) * (consistency + reward_loss)
            z = z_pred
        total_loss /= H
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(world_model.parameters(), 20.0)
        optimizer.step()
        if step % 10 == 0:
            print(f"  Step {step}/{args.wm_epochs}  loss={total_loss.item():.4f}")
            logger.log({"wm/loss": total_loss.item()}, step=step)
        if ckpt_dir is not None and save_freq > 0 and (step + 1) % save_freq == 0:
            path = ckpt_dir / f"world_model_step{step + 1:07d}.pt"
            torch.save({"step": step + 1, "world_model": world_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": {"obs_dim": world_model.obs_dim, "action_dim": world_model.action_dim}}, path)
            print(f"  Saved WM checkpoint → {path.name}")
    return world_model

def td_lambda_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discount: float,
    lmbd: float,
) -> torch.Tensor:
    """Compute TD(λ) targets.

    Args:
        rewards: (B, H) imagined rewards r_0 .. r_{H-1}
        values:  (B, H+1) critic predictions V(z_0) .. V(z_H);
                 values[:, -1] is the terminal bootstrap V(z_H)
        discount: γ
        lmbd:    λ mixing coefficient
    """
    B, H = rewards.shape
    targets = torch.zeros(B, H, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(H)):
        # values[:, t+1] is V(z_{t+1}); at t=H-1 this is the terminal V(z_H)
        if t == H - 1:
            next_target = values[:, t + 1]
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
    ret_rms: RunningMeanStd | None = None,
) -> dict[str, float]:
    H = args.policy_horizon
    gamma = args.discount

    with torch.no_grad():
        z = world_model.encode(start_states)

    # --- Actor update (FoG: gradients flow through dynamics and reward) ---
    imagined_rewards: list[torch.Tensor] = []
    imagined_values: list[torch.Tensor] = []
    z_current = z
    for _ in range(H):
        with torch.no_grad():
            # Pessimistic min across ensemble (suppresses Q overestimation)
            v = critic.min_value(z_current)
        imagined_values.append(v)
        action, _ = policy(z_current)
        reward = world_model.reward(z_current, action)            # (B,) scalar MSE head
        imagined_rewards.append(reward)
        z_current = world_model.next_state(z_current, action)

    with torch.no_grad():
        terminal_value = critic.min_value(z_current)

    rewards_tensor = torch.stack(imagined_rewards, dim=1)   # (B, H)
    values_tensor = torch.stack(imagined_values, dim=1)     # (B, H)
    discounts = gamma ** torch.arange(H, device=z.device, dtype=torch.float32)

    actor_loss = -(rewards_tensor * discounts).sum(dim=1).mean()
    actor_loss = actor_loss - (gamma ** H) * terminal_value.mean()

    # Variance-adaptive return normalization
    if ret_rms is not None:
        ret_rms.update((rewards_tensor.detach() * discounts).sum(dim=1))
        actor_loss = actor_loss / (ret_rms.std + 1e-5)

    policy_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    policy_optimizer.step()

    # --- Critic update: TD(λ) targets from a fresh frozen rollout ---
    with torch.no_grad():
        z_c = world_model.encode(start_states)
        rew_for_critic: list[torch.Tensor] = []
        for _ in range(H):
            a, _ = policy(z_c)
            r = world_model.reward(z_c, a)                        # (B,) scalar MSE head
            rew_for_critic.append(r)
            z_c = world_model.next_state(z_c, a)
        rew_critic = torch.stack(rew_for_critic, dim=1)       # (B, H)
        terminal_v = critic.min_value(z_c)                    # (B,) pessimistic bootstrap

    # Collect per-head predictions: critic.forward() returns (num_critics, B)
    # We need (num_critics, B, H) — one value per head per timestep
    z_c2 = world_model.encode(start_states).detach()
    head_preds: list[torch.Tensor] = []   # will be list of (num_critics, B) tensors
    for _ in range(H):
        with torch.no_grad():
            a, _ = policy(z_c2)
        head_preds.append(critic.forward(z_c2))               # (num_critics, B)
        with torch.no_grad():
            z_c2 = world_model.next_state(z_c2, a)

    # (num_critics, B, H): head h, batch b, timestep t
    head_tensor = torch.stack(head_preds, dim=2)

    with torch.no_grad():
        # Use mean of heads for TD-λ bootstrap values (target network equivalent)
        mean_preds = head_tensor.mean(dim=0)                   # (B, H)
        all_values = torch.cat([mean_preds, terminal_v.unsqueeze(1)], dim=1)  # (B, H+1)
        targets = td_lambda_targets(rew_critic, all_values, gamma, args.lmbd)  # (B, H)

    # Train all heads simultaneously against the same targets
    critic_loss = F.mse_loss(head_tensor, targets.unsqueeze(0).expand_as(head_tensor))
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
