"""Dreamer V1: Entry point

Usage:
  python -m lucidwm.dreamer.main --env-id walker-walk --seed 1 --track
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import numpy as np
import torch
import torch.optim as optim
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed
from .model import DreamerModel
from .loss import preprocess_batch, world_model_loss, behavior_losses
from .env import make_env_fn, reset_env_batch, collect_batched_steps, evaluate_agent

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: Dreamer V1 on DMControl")
    parser.add_argument("--env-id", type=str, default="walker-walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--prefill-steps", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--fast", action="store_true",
                        help="T4-friendly preset: smaller batches/models, fewer updates, AMP on.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--time-limit", type=int, default=1_000)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-length", type=int, default=50)
    parser.add_argument("--train-every", type=int, default=1_000)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--model-lr", type=float, default=6e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--value-lr", type=float, default=8e-5)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--stoch-size", type=int, default=30)
    parser.add_argument("--deter-size", type=int, default=200)
    parser.add_argument("--rssm-hidden-size", type=int, default=200)
    parser.add_argument("--num-units", type=int, default=400)
    parser.add_argument("--cnn-depth", type=int, default=32)
    parser.add_argument("--free-nats", type=float, default=3.0)
    parser.add_argument("--kl-scale", type=float, default=1.0)
    parser.add_argument("--pcont-scale", type=float, default=10.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--action-init-std", type=float, default=5.0)
    parser.add_argument("--min-std", type=float, default=1e-4)
    parser.add_argument("--mean-scale", type=float, default=5.0)
    parser.add_argument("--expl-amount", type=float, default=0.3)
    parser.add_argument("--eval-noise", type=float, default=0.0)
    parser.add_argument("--learn-cont", action="store_true")
    args = parser.parse_args()
    if args.fast:
        args.batch_size = 16
        args.batch_length = 16
        args.train_steps = 25
        args.train_every = max(args.train_every, 500)
        args.cnn_depth = 24
        args.deter_size = 128
        args.rssm_hidden_size = 128
        args.num_units = 256
        args.prefill_steps = min(args.prefill_steps, 2_000)
        args.amp = True
    return args

def train_step(model, model_opt, actor_opt, value_opt, batch, args, device, scaler=None):
    batch_t = preprocess_batch(batch, device)
    model_opt.zero_grad(set_to_none=True)
    actor_opt.zero_grad(set_to_none=True)
    value_opt.zero_grad(set_to_none=True)
    amp_enabled = bool(args.amp and device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
    with autocast_ctx:
        model_loss, post, wm_metrics = world_model_loss(model, batch_t, args)
        actor_loss, value_loss, behavior_metrics = behavior_losses(model, post, args)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(model_loss).backward()
        scaler.scale(actor_loss).backward()
        scaler.scale(value_loss).backward()
        scaler.unscale_(model_opt)
        scaler.unscale_(actor_opt)
        scaler.unscale_(value_opt)
    else:
        model_loss.backward()
        actor_loss.backward()
        value_loss.backward()
    model_params = (
        list(model.encoder.parameters()) + list(model.rssm.parameters())
        + list(model.decoder.parameters()) + list(model.reward.parameters())
        + ([] if model.cont is None else list(model.cont.parameters()))
    )
    model_grad = torch.nn.utils.clip_grad_norm_(model_params, args.grad_clip)
    actor_grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), args.grad_clip)
    value_grad = torch.nn.utils.clip_grad_norm_(model.value.parameters(), args.grad_clip)
    if scaler is not None and scaler.is_enabled():
        scaler.step(model_opt)
        scaler.step(actor_opt)
        scaler.step(value_opt)
        scaler.update()
    else:
        model_opt.step()
        actor_opt.step()
        value_opt.step()
    metrics = dict(wm_metrics)
    metrics.update(behavior_metrics)
    metrics["grads/model"] = float(model_grad)
    metrics["grads/actor"] = float(actor_grad)
    metrics["grads/value"] = float(value_grad)
    return metrics

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    env = make_env_fn(args, 0)()
    eval_env_fn = make_env_fn(args, 100)
    obs_shape = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))

    model = DreamerModel(action_dim, args).to(device)
    model_opt = optim.Adam(
        list(model.encoder.parameters()) + list(model.rssm.parameters())
        + list(model.decoder.parameters()) + list(model.reward.parameters())
        + ([] if model.cont is None else list(model.cont.parameters())),
        lr=args.model_lr,
    )
    actor_opt = optim.Adam(model.actor.parameters(), lr=args.actor_lr)
    value_opt = optim.Adam(model.value.parameters(), lr=args.value_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    buffer = ReplayBuffer(
        capacity=args.buffer_size, obs_shape=obs_shape,
        action_dim=action_dim, seq_len=args.batch_length + 1,
    )
    logger = Logger(
        project=args.wandb_project, name=f"dreamer_{args.env_id}_s{args.seed}",
        config=vars(args), use_wandb=args.track,
    )
    print(f"Dreamer V1 DMControl | env={args.env_id} | obs={obs_shape} | act={action_dim} | "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    env.close()
    envs = [make_env_fn(args, i)() for i in range(args.num_envs)]
    observations = reset_env_batch(envs, base_seed=args.seed)
    latent_states = [None for _ in range(args.num_envs)]
    prev_actions = [None for _ in range(args.num_envs)]
    episode_storage = [{"obs": [], "action": [], "reward": [], "done": []} for _ in range(args.num_envs)]
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    total_env_steps = 0
    train_counter = 0

    while total_env_steps < args.total_steps:
        envs, observations, latent_states, prev_actions, episode_storage, episode_returns, episode_lengths = collect_batched_steps(
            envs, observations, model, buffer, args, device, total_env_steps,
            latent_states, prev_actions, episode_storage, episode_returns, episode_lengths, logger,
        )
        total_env_steps += args.num_envs
        train_counter += args.num_envs

        if total_env_steps >= args.prefill_steps and buffer.num_episodes >= 1 and train_counter >= args.train_every:
            train_counter = 0
            metrics = None
            for _ in range(args.train_steps):
                try:
                    batch = buffer.sample(args.batch_size, seq_len=args.batch_length + 1)
                except ValueError:
                    break
                metrics = train_step(model, model_opt, actor_opt, value_opt, batch, args, device, scaler)
            if metrics is not None:
                logger.log(metrics, step=total_env_steps)

        if total_env_steps > 0 and total_env_steps % args.eval_freq < args.num_envs:
            model.eval()
            eval_result = evaluate_agent(model, eval_env_fn, num_episodes=args.eval_episodes, device=device, eval_noise=args.eval_noise)
            print(f"Step {total_env_steps:>7d} | eval={eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}")
            logger.log({
                "charts/eval_return": eval_result["mean_return"],
                "charts/eval_std": eval_result["std_return"],
            }, step=total_env_steps)
            model.train()

    torch.save({"model": model.state_dict(), "action_dim": action_dim, "args": vars(args)},
               f"dreamer_{args.env_id}_s{args.seed}.pt")
    logger.close()
    for env in envs:
        env.close()
