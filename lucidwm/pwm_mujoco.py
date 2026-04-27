"""LucidWM: PWM on MuJoCo (via Minari offline datasets)

This is a variant of pwm_dmc.py that trains on Minari's MuJoCo offline
datasets instead of TD-MPC2's custom dataset format. Same algorithm,
different data pipeline.

Minari datasets used:
  mujoco/halfcheetah/medium-v0     (1M steps, medium SAC policy)
  mujoco/halfcheetah/expert-v0     (1M steps, expert SAC policy)
  mujoco/hopper/medium-v0          (1M steps)
  mujoco/walker2d/medium-v0        (1M steps)
  mujoco/ant/medium-v0             (1M steps)
  D4RL/door/human-v2               (6.7k steps, human demos)
  D4RL/pen/human-v2                (5k steps, human demos)

Install: pip install "lucidwm[minari]"

Usage:
  # Train world model + policy on HalfCheetah medium dataset
  python -m lucidwm.pwm_mujoco --dataset mujoco/halfcheetah/medium-v0

  # Train on Hopper expert, evaluate in env
  python -m lucidwm.pwm_mujoco --dataset mujoco/hopper/expert-v0 --eval

  # Just learn policy from existing world model checkpoint
  python -m lucidwm.pwm_mujoco --dataset mujoco/ant/medium-v0 \\
      --wm-checkpoint pwm_wm_ant.pt --policy-steps 50000

Architecture: Same as pwm_dmc.py (TD-MPC2 world model + FoG policy learning).
See pwm_dmc.py docstring for full algorithm details.
"""

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import set_seed, get_device
from lucidwm_utils.datasets import MinariDatasetAdapter
from lucidwm.pwm_dmcontrol import (
    PWMModel, 
    PWMCriticEnsemble,
    PWMPolicy,
    policy_update
)

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: PWM")

    # Standard args
    parser.add_argument("--dataset", type=str, default="mujoco/halfcheetah/medium-v0",
                        help="Minari dataset ID to train on")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)

    # World model pre-training
    parser.add_argument("--wm-checkpoint", type=str, default=None,
                        help="Path to pre-trained world model checkpoint. "
                             "If None, trains from scratch on offline data.")
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--wm-epochs", type=int, default=100)
    parser.add_argument("--wm-batch-size", type=int, default=256)
    parser.add_argument("--wm-horizon", type=int, default=16,
                        help="Training horizon for world model (16 for better FoG gradients)")

    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--simnorm-dim", type=int, default=8)

    # Policy learning (phase 2)
    parser.add_argument("--policy-steps", type=int, default=10_000,
                        help="Gradient steps for policy learning (<10 min)")
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--policy-batch-size", type=int, default=32,
                        help="Small batches work better for FoG (paper finding)")
    parser.add_argument("--policy-horizon", type=int, default=5,
                        help="Imagination horizon for policy training")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lmbd", type=float, default=0.95, help="TD(lambda)")
    parser.add_argument("--num-critics", type=int, default=3)

    return parser.parse_args()
    
# World Model training on Minari data
def pretrain_world_model_minari(world_model, dataset, args, device, logger):
    """Pre-train world model on Minari dataset.

    Same loss as PWM/TD-MPC2 (consistency + reward cross-entropy),
    but sampling from Minari's flattened buffer.
    """
    optimizer = optim.Adam(world_model.parameters(), lr=args.wm_lr)
    H = args.wm_horizon
    seq_len = H + 1

    print(f"Pre-training world model: {args.wm_steps} steps, H={H}, "
          f"batch={args.wm_batch_size}")

    for step in range(args.wm_steps):
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

        if step % 1000 == 0:
            print(f"  Step {step}/{args.wm_steps}  loss={total_loss.item():.4f}")
            logger.log({"wm/loss": total_loss.item()}, step=step)

    return world_model

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Load Minari dataset
    dataset = MinariDatasetAdapter(args.dataset)
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    # Logger
    dataset_short = args.dataset.replace("/", "_")
    logger = Logger(
        project=args.wandb_project,
        name=f"pwm_minari_{dataset_short}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    # ── Phase 1: World Model ─────────────────────────────────────────
    if args.wm_checkpoint:
        print(f"Loading world model from {args.wm_checkpoint}")
        world_model = PWMModel(obs_dim, action_dim, args).to(device)
        ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
        if "world_model" in ckpt:
            world_model.load_state_dict(ckpt["world_model"])
        else:
            world_model.load_state_dict(ckpt, strict=False)
    else:
        print(f"\n{'='*60}\nPhase 1: Training world model on {args.dataset}\n{'='*60}")
        world_model = PWMModel(obs_dim, action_dim, args).to(device)
        world_model = pretrain_world_model_minari(
            world_model, dataset, args, device, logger
        )
        ckpt_path = f"pwm_wm_{dataset_short}_s{args.seed}.pt"
        torch.save({"world_model": world_model.state_dict(),
                     "obs_dim": obs_dim, "action_dim": action_dim}, ckpt_path)
        print(f"  Saved world model to {ckpt_path}")

    # Freeze
    world_model.eval()
    world_model.requires_grad_(False)
    print(f"World model frozen ({sum(p.numel() for p in world_model.parameters())/1e6:.1f}M)")

    # ── Phase 2: Policy Learning ─────────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: FoG policy learning ({args.policy_steps} steps)\n{'='*60}")

    policy = PWMPolicy(action_dim).to(device)
    critic = PWMCriticEnsemble(num_critics=args.num_critics).to(device)
    policy_opt = optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)

    print(f"  Policy: {sum(p.numel() for p in policy.parameters()):,} params")
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()):,} params")

    # Optionally recover env for evaluation
    eval_env_fn = None
    if args.eval:
        try:
            test_env = dataset.recover_environment()
            test_env.close()
            eval_env_fn = lambda: dataset.recover_environment()
            print(f"  Evaluation env recovered from dataset")
        except Exception as e:
            print(f"  Could not recover eval env: {e}")
            args.eval = False

    for step in range(args.policy_steps):
        start_obs = torch.tensor(
            dataset.sample_start_states(args.policy_batch_size),
            dtype=torch.float32, device=device,
        )
        metrics = policy_update(
            world_model, policy, critic,
            policy_opt, critic_opt,
            start_obs, args,
        )

        if step % 500 == 0:
            logger.log(metrics, step=step)
            print(f"  Step {step:>6d}/{args.policy_steps}  "
                  f"actor={metrics['losses/actor']:.4f}  "
                  f"critic={metrics['losses/critic']:.4f}  "
                  f"reward={metrics['algo/mean_imagined_reward']:.3f}")

        # Evaluate
        if args.eval and step > 0 and step % args.eval_freq == 0:
            policy.eval()

            def agent_fn(o):
                with torch.no_grad():
                    o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    z = world_model.encode(o_t)
                    a, _ = policy(z, deterministic=True)
                    return a.squeeze(0).cpu().numpy()

            from lucidwm_utils.metrics import evaluate
            eval_result = evaluate(eval_env_fn, agent_fn, num_episodes=args.eval_episodes)
            print(f"  EVAL: {eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}")
            logger.log({"charts/eval_return": eval_result["mean_return"]}, step=step)
            policy.train()

    # Save
    policy_path = f"pwm_policy_{dataset_short}_s{args.seed}.pt"
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()}, policy_path)
    print(f"\nSaved policy to {policy_path}")

    # Final eval
    if args.eval:
        policy.eval()

        def agent_fn(o):
            with torch.no_grad():
                o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                z = world_model.encode(o_t)
                a, _ = policy(z, deterministic=True)
                return a.squeeze(0).cpu().numpy()

        from lucidwm_utils.metrics import evaluate
        result = evaluate(eval_env_fn, agent_fn, num_episodes=50)
        print(f"\nFinal eval (50 eps): {result['mean_return']:.1f} +/- {result['std_return']:.1f}")

    logger.close()
    print("Done.")

