"""PWM: Entry point

Usage:
  # DMControl offline dataset (HuggingFace mt30/mt80)
  python -m lucidwm.pwm.main --dataset dmcontrol --env-id walker-walk --seed 1

  # MuJoCo/Minari offline dataset
  python -m lucidwm.pwm.main --dataset mujoco --minari-id mujoco/halfcheetah/medium-v0

  # Use pre-trained world model checkpoint
  python -m lucidwm.pwm.main --dataset dmcontrol --wm-checkpoint pwm_wm.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.optim as optim

from lucidwm_utils.envs import make_env
from lucidwm_utils.metrics import evaluate
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import set_seed, get_device

from .model import PWMModel, PWMPolicy, PWMCriticEnsemble
from .train import pretrain_world_model, pretrain_world_model_minari, policy_update

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: PWM")
    parser.add_argument("--env-id", type=str, default="walker-walk")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="dmcontrol", choices=["dmcontrol", "mujoco"],
                        help="Dataset type: dmcontrol (HuggingFace mt30/mt80) or mujoco (Minari)")
    parser.add_argument("--hf-dataset", type=str, default="mt30", choices=["mt30", "mt80"],
                        help="HuggingFace dataset split (dmcontrol only)")
    parser.add_argument("--data-dir", type=str, default="data/tdmpc2",
                        help="Local cache dir for HuggingFace dataset (dmcontrol only)")
    parser.add_argument("--minari-id", type=str, default="mujoco/halfcheetah/medium-v0",
                        help="Minari dataset ID (mujoco only)")
    parser.add_argument("--wm-checkpoint", type=str, default=None)
    parser.add_argument("--wm-lr", type=float, default=3e-4)
    parser.add_argument("--wm-epochs", type=int, default=100)
    parser.add_argument("--wm-batch-size", type=int, default=256)
    parser.add_argument("--wm-horizon", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-bins", type=int, default=101)
    parser.add_argument("--simnorm-dim", type=int, default=8)
    parser.add_argument("--simnorm-temp", type=float, default=0.5)
    parser.add_argument("--policy-steps", type=int, default=10_000)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--policy-batch-size", type=int, default=32)
    parser.add_argument("--policy-horizon", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lmbd", type=float, default=0.95)
    parser.add_argument("--num-critics", type=int, default=3)
    return parser.parse_args()

def load_dataset(args):
    if args.dataset == "dmcontrol":
        from lucidwm.tdmpc2.model import OfflineDataset
        return OfflineDataset(dataset=args.hf_dataset, data_dir=args.data_dir)
    else:
        from lucidwm_utils.datasets import MinariDatasetAdapter
        return MinariDatasetAdapter(args.minari_id)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    run_name = (
        f"pwm_{args.dataset}_{args.env_id}_s{args.seed}"
        if args.dataset == "dmcontrol"
        else f"pwm_minari_{args.minari_id.replace('/', '_')}_s{args.seed}"
    )
    logger = Logger(
        project=args.wandb_project, name=run_name,
        config=vars(args), use_wandb=args.track,
    )

    if args.wm_checkpoint:
        print(f"Loading world model from {args.wm_checkpoint}")
        ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
        obs_dim = ckpt.get("args", {}).get("obs_dim", None)
        action_dim = ckpt.get("args", {}).get("action_dim", None)
        if obs_dim is None or action_dim is None:
            env = make_env(args.env_id, args.seed, action_repeat=args.action_repeat)
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            env.close()
        world_model = PWMModel(obs_dim, action_dim, args).to(device)
        state_dict = ckpt.get("world_model", ckpt.get("model", ckpt))
        world_model.load_state_dict(state_dict, strict=False)
        dataset = load_dataset(args)
        print(f"World model loaded: obs={obs_dim}, act={action_dim}")
    else:
        print(f"Phase 1: Pre-training world model on {args.dataset} data...")
        dataset = load_dataset(args)
        obs_dim = dataset.obs_all.shape[-1] if hasattr(dataset, "obs_all") else dataset.obs_dim
        action_dim = dataset.action_all.shape[-1] if hasattr(dataset, "action_all") else dataset.action_dim
        world_model = PWMModel(obs_dim, action_dim, args).to(device)
        if args.dataset == "dmcontrol":
            world_model = pretrain_world_model(world_model, dataset, args, device, logger)
        else:
            world_model = pretrain_world_model_minari(world_model, dataset, args, device, logger)
        wm_path = f"pwm_wm_{args.dataset}_s{args.seed}.pt"
        torch.save({
            "world_model": world_model.state_dict(),
            "args": {"obs_dim": obs_dim, "action_dim": action_dim},
        }, wm_path)
        print(f"Saved world model to {wm_path}")

    world_model.eval()
    world_model.requires_grad_(False)
    print(f"World model frozen ({sum(p.numel() for p in world_model.parameters())/1e6:.1f}M params)")

    print(f"\nPhase 2: FoG policy learning ({args.policy_steps} steps)")
    policy = PWMPolicy(args, action_dim).to(device)
    critic = PWMCriticEnsemble(args, num_critics=args.num_critics).to(device)
    policy_opt = optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr)

    eval_env_fn = None
    if args.eval and args.dataset == "mujoco":
        try:
            test_env = dataset.recover_environment()
            test_env.close()
            eval_env_fn = lambda: dataset.recover_environment()
        except Exception as e:
            print(f"  Could not recover eval env: {e}")
            args.eval = False

    for step in range(args.policy_steps):
        if args.dataset == "dmcontrol":
            batch = dataset.sample(args.policy_batch_size, seq_len=1)
            start_obs = torch.tensor(batch["obs"][:, 0], dtype=torch.float32, device=device)
        else:
            start_obs = torch.tensor(
                dataset.sample_start_states(args.policy_batch_size),
                dtype=torch.float32, device=device,
            )
        metrics = policy_update(world_model, policy, critic, policy_opt, critic_opt, start_obs, args)
        if step % 500 == 0:
            logger.log(metrics, step=step)
            print(f"  Step {step:>6d}/{args.policy_steps}  "
                  f"actor={metrics['losses/actor']:.4f}  "
                  f"critic={metrics['losses/critic']:.4f}")
        if step > 0 and step % args.eval_freq == 0:
            policy.eval()
            def agent_fn(o):
                with torch.no_grad():
                    o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    z = world_model.encode(o_t)
                    a, _ = policy(z, deterministic=True)
                    return a.squeeze(0).cpu().numpy()
            try:
                env_fn = eval_env_fn if eval_env_fn else lambda: make_env(args.env_id, args.seed + 100, action_repeat=args.action_repeat)
                eval_result = evaluate(env_fn, agent_fn, num_episodes=args.eval_episodes)
                print(f"  EVAL step {step}: {eval_result['mean_return']:.1f} +/- {eval_result['std_return']:.1f}")
                logger.log({
                    "charts/eval_return": eval_result["mean_return"],
                    "charts/eval_std": eval_result["std_return"],
                }, step=step)
            except Exception as e:
                print(f"  Eval failed: {e}")
            policy.train()

    policy_path = f"pwm_policy_{args.dataset}_s{args.seed}.pt"
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict(), "args": vars(args)}, policy_path)
    print(f"\nSaved policy to {policy_path}")
    logger.close()
