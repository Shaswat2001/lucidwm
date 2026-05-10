"""Dreamer V2: Entry point

Usage:
  python -m lucidwm.dreamerv2.main --env-id ALE/Pong-v5 --seed 1
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from lucidwm_utils.buffers import ReplayBuffer
from lucidwm_utils.envs import make_env
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed
from .model import DreamerV2Model, RSSMState
from .loss import preprocess_batch, world_model_loss, behavior_losses, preprocess_obs

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: DreamerV2 on Atari")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--prefill-steps", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-length", type=int, default=50)
    parser.add_argument("--train-every", type=int, default=1_000)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--model-lr", type=float, default=2e-4)
    parser.add_argument("--actor-lr", type=float, default=4e-5)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--stoch-size", type=int, default=32)
    parser.add_argument("--stoch-classes", type=int, default=32)
    parser.add_argument("--deter-size", type=int, default=600)
    parser.add_argument("--rssm-hidden-size", type=int, default=600)
    parser.add_argument("--num-units", type=int, default=600)
    parser.add_argument("--cnn-depth", type=int, default=48)
    parser.add_argument("--discount", type=float, default=0.999)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--imagine-horizon", type=int, default=15)
    parser.add_argument("--kl-scale", type=float, default=1.0)
    parser.add_argument("--kl-balance", type=float, default=0.8)
    parser.add_argument("--pcont-scale", type=float, default=10.0)
    parser.add_argument("--actor-entropy-scale", type=float, default=1e-3)
    parser.add_argument("--explore-entropy-scale", type=float, default=3e-4)
    parser.add_argument("--learn-cont", action="store_true")
    return parser.parse_args()

@torch.no_grad()
def act(model, obs, prev_state, prev_action, device, deterministic, entropy_scale):
    obs_t = torch.tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0)
    embed = model.encoder(preprocess_obs(obs_t))
    if prev_state is None:
        prev_state = model.rssm.init_state(1, device)
    if prev_action is None:
        prev_action = torch.zeros(1, model.action_dim, device=device)
    post, _ = model.rssm.obs_step(prev_state, prev_action, embed)
    feat = model.rssm.get_feat(post)
    action_dist = model.actor(feat)
    if deterministic:
        action_idx = action_dist.probs.argmax(dim=-1)
    else:
        action_idx = action_dist.sample()
    action = F.one_hot(action_idx, model.action_dim).float()
    return int(action_idx.item()), post, action

@torch.no_grad()
def evaluate_agent(model, env_fn, num_episodes, device):
    returns, lengths = [], []
    for _ in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        latent_state = None
        prev_action = None
        ep_return = 0.0
        ep_length = 0
        while not done:
            action, latent_state, prev_action = act(model, obs, latent_state, prev_action, device, deterministic=True, entropy_scale=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(ep_length)
        env.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
    }

def train_step(model, model_opt, actor_opt, value_opt, batch, args, device):
    batch_t = preprocess_batch(batch, device)
    model_opt.zero_grad(set_to_none=True)
    actor_opt.zero_grad(set_to_none=True)
    value_opt.zero_grad(set_to_none=True)
    model_loss, post, wm_metrics = world_model_loss(model, batch_t, args)
    actor_loss, value_loss, behavior_metrics = behavior_losses(model, post, args)
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

    env = make_env(args.env_id, seed=args.seed, img_size=args.img_size, frame_stack=args.frame_stack)
    eval_env_fn = lambda: make_env(args.env_id, seed=args.seed + 100, img_size=args.img_size, frame_stack=args.frame_stack)

    obs_shape = tuple(env.observation_space.shape)
    action_dim = int(env.action_space.n)

    model = DreamerV2Model(obs_shape, action_dim, args).to(device)
    model_params = (
        list(model.encoder.parameters()) + list(model.rssm.parameters())
        + list(model.decoder.parameters()) + list(model.reward.parameters())
        + ([] if model.cont is None else list(model.cont.parameters()))
    )
    model_opt = optim.Adam(model_params, lr=args.model_lr)
    actor_opt = optim.Adam(model.actor.parameters(), lr=args.actor_lr)
    value_opt = optim.Adam(model.value.parameters(), lr=args.value_lr)

    buffer = ReplayBuffer(capacity=args.buffer_size, obs_shape=obs_shape, action_dim=action_dim, seq_len=args.batch_length + 1)
    logger = Logger(
        project=args.wandb_project,
        name=f"dreamerv2_{args.env_id.replace('/', '_')}_s{args.seed}",
        config=vars(args), use_wandb=args.track,
    )
    print(f"DreamerV2 Atari | env={args.env_id} | obs={obs_shape} | act={action_dim} | "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    obs, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    latent_state = None
    prev_action = None

    for step in range(args.total_steps):
        if step < args.prefill_steps:
            env_action = int(env.action_space.sample())
            action_vec = F.one_hot(torch.tensor(env_action), action_dim).float().cpu().numpy()
        else:
            env_action, latent_state, prev_action = act(
                model, obs, latent_state, prev_action, device,
                deterministic=False, entropy_scale=args.explore_entropy_scale,
            )
            action_vec = prev_action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        buffer.add_step(obs, action_vec.astype(np.float32), float(reward), done)
        episode_return += reward
        episode_length += 1
        obs = next_obs

        if done:
            logger.log({"charts/episodic_return": episode_return, "charts/episodic_length": episode_length}, step=step)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            latent_state = None
            prev_action = None

        if step >= args.prefill_steps and buffer.num_episodes >= 1 and step % args.train_every == 0:
            metrics = None
            for _ in range(args.train_steps):
                try:
                    batch = buffer.sample(args.batch_size, seq_len=args.batch_length + 1)
                except ValueError:
                    break
                metrics = train_step(model, model_opt, actor_opt, value_opt, batch, args, device)
            if metrics is not None:
                logger.log(metrics, step=step)

        if step > 0 and step % args.eval_freq == 0:
            model.eval()
            eval_result = evaluate_agent(model, eval_env_fn, num_episodes=args.eval_episodes, device=device)
            print(f"Step {step:>7d} | eval={eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}")
            logger.log({"charts/eval_return": eval_result["mean_return"], "charts/eval_std": eval_result["std_return"]}, step=step)
            model.train()

    torch.save({"model": model.state_dict(), "action_dim": action_dim, "obs_shape": obs_shape, "args": vars(args)},
               f"dreamerv2_{args.env_id.replace('/', '_')}_s{args.seed}.pt")
    logger.close()
    env.close()
