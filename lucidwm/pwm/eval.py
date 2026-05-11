"""PWM: Evaluation script

Usage:
  # DMControl
  python -m lucidwm.pwm.eval \
    --policy-ckpt checkpoints/pwm_dmcontrol_walker-walk_s1/policy_final.pt \
    --wm-ckpt    checkpoints/pwm_dmcontrol_walker-walk_s1/world_model.pt \
    --env-id walker-walk --episodes 10 --record

  # MuJoCo via Minari
  python -m lucidwm.pwm.eval \
    --policy-ckpt checkpoints/pwm_minari_mujoco_halfcheetah_medium-v0_s1/policy_final.pt \
    --wm-ckpt    checkpoints/pwm_minari_mujoco_halfcheetah_medium-v0_s1/world_model.pt \
    --minari-id mujoco/halfcheetah/medium-v0 --episodes 10 --record
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch

from lucidwm_utils.envs import make_env
from lucidwm_utils.misc import set_seed, get_device
from lucidwm_utils.video import save_video, save_video_grid


def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: PWM evaluation")
    parser.add_argument("--policy-ckpt", type=str, required=True,
                        help="Path to policy checkpoint (policy_final.pt or policy_step*.pt)")
    parser.add_argument("--wm-ckpt", type=str, required=True,
                        help="Path to world model checkpoint (world_model.pt or world_model_step*.pt)")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Environment ID for DMControl/Gym (e.g., walker-walk)")
    parser.add_argument("--minari-id", type=str, default=None,
                        help="Minari dataset ID to recover MuJoCo env (e.g., mujoco/halfcheetah/medium-v0)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-repeat", type=int, default=2,
                        help="Action repeat for DMControl environments")
    parser.add_argument("--record", action="store_true",
                        help="Record and save a video for every episode")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save recorded videos and results.json")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip the combined grid video")
    return parser.parse_args()


def load_models(policy_ckpt_path: str, wm_ckpt_path: str, device: torch.device):
    from .model import PWMModel, PWMPolicy

    wm_ckpt = torch.load(wm_ckpt_path, map_location=device, weights_only=False)
    policy_ckpt = torch.load(policy_ckpt_path, map_location=device, weights_only=False)

    wm_state = wm_ckpt["world_model"]
    policy_state = policy_ckpt["policy"]

    # obs_dim: first Linear in encoder takes (hidden_dim, obs_dim) weight → shape[1]
    obs_dim = wm_ckpt.get("args", {}).get("obs_dim") or int(wm_state["encoder.0.net.0.weight"].shape[1])
    # action_dim: log_std parameter has shape (action_dim,)
    action_dim = wm_ckpt.get("args", {}).get("action_dim") or int(policy_state["log_std"].shape[0])

    model_args = types.SimpleNamespace(**policy_ckpt["args"])

    world_model = PWMModel(obs_dim, action_dim, model_args).to(device)
    world_model.load_state_dict(wm_state)
    world_model.eval()
    world_model.requires_grad_(False)

    policy = PWMPolicy(model_args, action_dim).to(device)
    policy.load_state_dict(policy_state)
    policy.eval()

    return world_model, policy, obs_dim, action_dim


def make_eval_env(env_id, minari_id, seed, action_repeat):
    """Create one evaluation environment instance.

    DMControl envs already use render_mode='rgb_array' in make_env.
    For Minari/MuJoCo, we pass render_mode explicitly so env.render() works.
    """
    if minari_id is not None:
        import minari
        dataset = minari.load_dataset(minari_id, download=False)
        try:
            return dataset.recover_environment(render_mode="rgb_array")
        except TypeError:
            return dataset.recover_environment()

    return make_env(env_id, seed=seed, action_repeat=action_repeat)


def run_episode(
    env,
    world_model: torch.nn.Module,
    policy: torch.nn.Module,
    device: torch.device,
    seed: int,
    record: bool,
) -> tuple[float, int, list[np.ndarray]]:
    obs, _ = env.reset(seed=seed)
    done = False
    total_return = 0.0
    length = 0
    frames: list[np.ndarray] = []

    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            z = world_model.encode(obs_t)
            action, _ = policy(z, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_return += float(reward)
        length += 1
        done = terminated or truncated

        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    return total_return, length, frames


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.env_id is None and args.minari_id is None:
        raise ValueError("Provide --env-id (DMControl/Gym) or --minari-id (MuJoCo/Minari)")

    print(f"Loading checkpoints...")
    print(f"  WM:     {args.wm_ckpt}")
    print(f"  Policy: {args.policy_ckpt}")
    world_model, policy, obs_dim, action_dim = load_models(
        args.policy_ckpt, args.wm_ckpt, device,
    )
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}")

    out_dir = Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {args.episodes} episodes...")
    returns: list[float] = []
    lengths: list[int] = []
    all_frames: list[list[np.ndarray]] = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        env = make_eval_env(args.env_id, args.minari_id, seed=ep_seed,
                            action_repeat=args.action_repeat)
        ep_return, ep_length, frames = run_episode(
            env, world_model, policy, device, seed=ep_seed, record=args.record,
        )
        env.close()

        returns.append(ep_return)
        lengths.append(ep_length)
        if frames:
            all_frames.append(frames)

        print(f"  ep {ep + 1:>3d}/{args.episodes}  "
              f"return={ep_return:>9.2f}  steps={ep_length}")

        if args.record and frames:
            save_video(frames, out_dir / f"episode_{ep + 1:03d}.mp4", fps=args.fps)

    # --- Summary ---
    mean_ret = float(np.mean(returns))
    std_ret  = float(np.std(returns))
    mean_len = float(np.mean(lengths))

    print(f"\n{'─' * 46}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Return   : {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  Min/Max  : {min(returns):.2f} / {max(returns):.2f}")
    print(f"  Length   : {mean_len:.1f} steps avg")
    print(f"{'─' * 46}")

    # --- Grid video ---
    if args.record and all_frames and not args.no_grid:
        grid_path = out_dir / "eval_grid.mp4"
        save_video_grid(all_frames, grid_path, fps=args.fps)
        print(f"\nVideos → {out_dir}/")
        print(f"  episode_001.mp4 … episode_{args.episodes:03d}.mp4")
        print(f"  eval_grid.mp4")

    # --- JSON results ---
    results = {
        "policy_ckpt": str(args.policy_ckpt),
        "wm_ckpt": str(args.wm_ckpt),
        "env_id": args.env_id,
        "minari_id": args.minari_id,
        "episodes": args.episodes,
        "seed": args.seed,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "min_return": float(min(returns)),
        "max_return": float(max(returns)),
        "mean_length": mean_len,
        "per_episode_returns": returns,
        "per_episode_lengths": lengths,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results → {results_path}")
