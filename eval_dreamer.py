"""Evaluate a trained Dreamer V1 DMControl policy and save rollout video."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from lucidwm.dreamer_dmcontrol import DreamerModel, act
from lucidwm_utils.envs import make_env
from lucidwm_utils.misc import get_device, set_seed
from lucidwm_utils.video import save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dreamer V1 on DMControl and save MP4")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved Dreamer checkpoint")
    parser.add_argument("--env-id", type=str, default=None, help="Override env id from checkpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--img-size", type=int, default=None, help="Override checkpoint image size")
    parser.add_argument("--action-repeat", type=int, default=None, help="Override checkpoint action repeat")
    parser.add_argument("--time-limit", type=int, default=None, help="Override checkpoint time limit")
    parser.add_argument("--eval-noise", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="videos/dreamer_dmcontrol_eval.mp4")
    return parser.parse_args()


def frame_from_env(env, obs: np.ndarray) -> np.ndarray:
    frame = env.render()
    if frame is not None:
        return np.asarray(frame).astype(np.uint8)
    if obs.ndim == 3 and obs.shape[0] in (1, 3):
        return (np.clip(obs.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
    raise RuntimeError("Could not obtain a render frame from the environment.")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = SimpleNamespace(**ckpt["args"])

    env_id = args.env_id or getattr(ckpt_args, "env_id", "walker-walk")
    img_size = args.img_size or getattr(ckpt_args, "img_size", 64)
    action_repeat = args.action_repeat or getattr(ckpt_args, "action_repeat", 2)
    time_limit = args.time_limit or getattr(ckpt_args, "time_limit", 1_000)
    action_dim = ckpt["action_dim"]

    env = make_env(
        env_id,
        seed=args.seed,
        obs="rgb",
        img_size=img_size,
        action_repeat=action_repeat,
        time_limit=time_limit,
    )

    model = DreamerModel(action_dim, ckpt_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_returns = []
    video_frames = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        latent_state = None
        prev_action = None
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < args.max_steps:
            if ep == 0:
                video_frames.append(frame_from_env(env, obs))
            action, latent_state, prev_action = act(
                model,
                obs,
                latent_state,
                prev_action,
                device,
                deterministic=True,
                expl_amount=args.eval_noise,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            steps += 1
            done = terminated or truncated

        if ep == 0:
            video_frames.append(frame_from_env(env, obs))
        all_returns.append(ep_return)

    env.close()
    save_video(video_frames, args.output, fps=args.fps)

    mean_return = float(np.mean(all_returns))
    std_return = float(np.std(all_returns))
    print(f"Saved video to {args.output}")
    print(f"Return over {args.episodes} episode(s): {mean_return:.1f} +/- {std_return:.1f}")


if __name__ == "__main__":
    main()
