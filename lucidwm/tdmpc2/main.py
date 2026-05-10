"""TD-MPC2: Entry point

Usage:
  python -m lucidwm.tdmpc2.main --env-id walker-walk --seed 1
  python -m lucidwm.tdmpc2.main --offline --dataset mt30 --data-dir ./data
"""
from __future__ import annotations
import copy
import argparse
import numpy as np
import torch
import torch.optim as optim
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.envs import make_env
from lucidwm_utils.logger import Logger
from lucidwm_utils.metrics import evaluate
from lucidwm_utils.misc import get_device, set_seed
from .model import TDMPC2Model, OfflineDataset
from .planner import plan_mppi
from .train import update, train_offline

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: TD-MPC2")
    parser.add_argument("--env-id", type=str, default="walker-walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--prefill-steps", type=int, default=5000)
    parser.add_argument("--utd", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=1e-4)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--policy-rho", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--num-q", type=int, default=5)
    parser.add_argument("--simnorm-dim", type=int, default=8)
    parser.add_argument("--simnorm-temp", type=float, default=0.5)
    parser.add_argument("--mppi-n", type=int, default=512)
    parser.add_argument("--mppi-iter", type=int, default=6)
    parser.add_argument("--mppi-temp", type=float, default=0.5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dataset", type=str, default="mt30", choices=["mt30", "mt80"])
    parser.add_argument("--data-dir", type=str, default="data/tdmpc2")
    parser.add_argument("--train-steps", type=int, default=500_000)
    parser.add_argument("--offline-batch-size", type=int, default=1024)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.offline:
        print(f"TD-MPC2 OFFLINE | dataset={args.dataset} | device={device}")
        dataset = OfflineDataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            task_filter=args.env_id if args.env_id != "walker-walk" else None,
        )
        obs_dim = dataset.obs_all.shape[-1]
        action_dim = dataset.action_all.shape[-1]
        model = TDMPC2Model(obs_dim, action_dim, args).to(device)
        target_model = copy.deepcopy(model).to(device)
        target_model.requires_grad_(False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print(f"Model: obs={obs_dim}, act={action_dim}, "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        logger = Logger(
            project=args.wandb_project,
            name=f"tdmpc2_offline_{args.dataset}_s{args.seed}",
            config=vars(args), use_wandb=args.track,
        )
        train_offline(model, target_model, optimizer, dataset, args, device, logger)
        torch.save({
            "model": model.state_dict(),
            "target_model": target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, f"tdmpc2_offline_{args.dataset}_s{args.seed}.pt")
        logger.close()
    else:
        env = make_env(args.env_id, args.seed, action_repeat=args.action_repeat)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = TDMPC2Model(obs_dim, action_dim, args).to(device)
        target_model = copy.deepcopy(model).to(device)
        target_model.requires_grad_(False)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        seq_len = args.horizon + 1
        buffer = ReplayBuffer(
            capacity=1_000_000,
            obs_shape=(obs_dim,),
            action_dim=action_dim,
            seq_len=seq_len,
        )
        logger = Logger(
            project=args.wandb_project,
            name=f"tdmpc2_{args.env_id}_s{args.seed}",
            config=vars(args), use_wandb=args.track,
        )
        print(f"TD-MPC2 ONLINE | env={args.env_id} | obs={obs_dim} | act={action_dim} | "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0
        prev_mean = None
        for step in range(args.total_steps):
            if step < args.prefill_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    z = model.encode(obs_t)
                    action, prev_mean = plan_mppi(model, args, z, prev_mean, args.horizon, action_dim, device)
                    action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add_step(obs, action, reward, done)
            episode_return += reward
            episode_length += 1
            obs = next_obs
            if done:
                logger.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_length,
                }, step=step)
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
                prev_mean = None
            if step >= args.prefill_steps and buffer.num_episodes >= 1:
                for _ in range(args.utd):
                    try:
                        batch = buffer.sample(args.batch_size, seq_len=seq_len)
                    except ValueError:
                        break
                    metrics = update(model, target_model, optimizer, batch, args, device)
                if step % 1000 == 0:
                    logger.log(metrics, step=step)
            if step > 0 and step % args.eval_freq == 0:
                model.eval()
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
                model.train()
                print(f"Step {step:>7d} | eval={eval_result['mean_return']:.1f} "
                      f"+/- {eval_result['std_return']:.1f}")
                logger.log({
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                }, step=step)
        logger.close()
        env.close()
