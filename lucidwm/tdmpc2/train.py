"""TD-MPC2: Training update and offline training loop"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .model import TDMPC2Model

def update(
    model: TDMPC2Model,
    target_model: TDMPC2Model,
    optimizer: optim.Optimizer,
    batch: dict,
    args,
    device: torch.device,
) -> dict[str, float]:
    obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
    reward = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
    done = torch.tensor(batch["done"], dtype=torch.float32, device=device)
    B, T = obs.shape[:2]
    H = min(T - 1, args.horizon)
    z = model.encode(obs[:, 0])
    total_loss = torch.tensor(0.0, device=device)
    consistency_loss_sum = 0.0
    reward_loss_sum = 0.0
    value_loss_sum = 0.0
    cont_loss_sum = 0.0
    policy_states: list[torch.Tensor] = [z.detach()]
    for t in range(H):
        a_t = action[:, t]
        z_pred = model.next_state(z, a_t)
        with torch.no_grad():
            z_target = target_model.encode(obs[:, t + 1])
        consistency_loss = F.mse_loss(z_pred, z_target)
        total_loss = total_loss + args.rho * consistency_loss
        consistency_loss_sum += consistency_loss.item()
        r_logits = model.reward(z, a_t)
        r_target = model.two_hot.encode(reward[:, t])
        reward_loss = -(r_target * F.log_softmax(r_logits, dim=-1)).sum(dim=-1).mean()
        total_loss = total_loss + reward_loss
        reward_loss_sum += reward_loss.item()
        cont_logits = model.cont(z, a_t).squeeze(-1)
        cont_target = 1.0 - done[:, t]
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)
        total_loss = total_loss + cont_loss
        cont_loss_sum += cont_loss.item()
        with torch.no_grad():
            z_next_target = target_model.encode(obs[:, t + 1])
            a_next, log_prob_next = model.policy(z_next_target)
            q_target_logits = target_model.q_values(z_next_target, a_next)
            idx = torch.randperm(args.num_q)[:2]
            q1_target = model.two_hot.decode(q_target_logits[idx[0]]).squeeze(-1)
            q2_target = model.two_hot.decode(q_target_logits[idx[1]]).squeeze(-1)
            q_next = torch.min(q1_target, q2_target)
            gamma = args.discount * (1 - done[:, t])
            td_target = reward[:, t] + gamma * (q_next - args.entropy_coef * log_prob_next)
        q_logits_all = model.q_values(z.detach(), a_t)
        td_target_encoded = model.two_hot.encode(td_target)
        q_loss_sum = 0.0
        for q_logits in q_logits_all:
            q_loss = -(td_target_encoded * F.log_softmax(q_logits, dim=-1)).sum(-1).mean()
            total_loss = total_loss + q_loss / args.num_q
            q_loss_sum += q_loss.item()
        value_loss_sum += q_loss_sum / args.num_q
        z = z_pred
        policy_states.append(z.detach())
    policy_loss = torch.tensor(0.0, device=device)
    rho_norm = sum(args.policy_rho ** t for t in range(len(policy_states)))
    for t, z_policy in enumerate(policy_states):
        a_pi, log_prob = model.policy(z_policy)
        q_logits_all = model.q_values(z_policy, a_pi)
        idx = torch.randperm(args.num_q)[:2]
        q1 = model.two_hot.decode(q_logits_all[idx[0]]).squeeze(-1)
        q2 = model.two_hot.decode(q_logits_all[idx[1]]).squeeze(-1)
        q_min = torch.min(q1, q2)
        step_loss = (args.entropy_coef * log_prob - q_min).mean()
        policy_loss = policy_loss + (args.policy_rho ** t) * step_loss
    policy_loss = policy_loss / rho_norm
    total_loss = total_loss + policy_loss
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 20.0)
    optimizer.step()
    with torch.no_grad():
        for p, tp in zip(model.parameters(), target_model.parameters()):
            tp.data.lerp_(p.data, args.tau)
    return {
        "losses/total": total_loss.item(),
        "losses/consistency": consistency_loss_sum / H,
        "losses/reward": reward_loss_sum / H,
        "losses/continue": cont_loss_sum / H,
        "losses/value": value_loss_sum / H,
        "losses/policy": policy_loss.item(),
    }

def train_offline(model, target_model, optimizer, dataset, args, device, logger):
    seq_len = args.horizon + 1
    batch_size = args.offline_batch_size
    print(f"\nOffline training: {args.train_steps} steps, batch_size={batch_size}, horizon={args.horizon}")
    for step in range(args.train_steps):
        batch = dataset.sample(batch_size, seq_len=seq_len)
        metrics = update(model, target_model, optimizer, batch, args, device)
        if step % 1000 == 0:
            logger.log(metrics, step=step)
            print(f"  Step {step:>7d}/{args.train_steps}  "
                  f"loss={metrics['losses/total']:.4f}  "
                  f"consistency={metrics['losses/consistency']:.4f}  "
                  f"reward={metrics['losses/reward']:.4f}")
        if step > 0 and step % args.eval_freq == 0 and args.env_id:
            model.eval()
            try:
                from lucidwm_utils.envs import make_env
                from lucidwm_utils.metrics import evaluate
                def agent_fn(o):
                    with torch.no_grad():
                        o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                        z = model.encode(o_t)
                        a, _ = model.policy(z, deterministic=True)
                        return a.squeeze(0).cpu().numpy()
                eval_result = evaluate(
                    lambda: make_env(args.env_id, args.seed + 100, action_repeat=args.action_repeat),
                    agent_fn,
                    num_episodes=args.eval_episodes,
                )
                print(f"  Eval: {eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}")
                logger.log({
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                }, step=step)
            except Exception as e:
                print(f"  Eval failed: {e}")
            model.train()
