"""World Models (Ha & Schmidhuber): Entry point

Usage:
  python -m lucidwm.wm.main --env-id CarRacing-v3 --seed 1
  python -m lucidwm.wm.main --env-id CarRacing-v3 --skip-data --skip-vae  # resume from rnn
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed
from .model import VAE, MDNRNN, Controller
from .train import train_vae, train_rnn, train_controller_cmaes
from .env import collect_data, rollout_agent

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: World Models (Ha & Schmidhuber)")
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--logdir", type=str, default="exp/wm_carracing")
    parser.add_argument("--num-rollouts", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--vae-epochs", type=int, default=30)
    parser.add_argument("--vae-batch-size", type=int, default=128)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-latent-size", type=int, default=32)
    parser.add_argument("--rnn-epochs", type=int, default=30)
    parser.add_argument("--rnn-batch-size", type=int, default=64)
    parser.add_argument("--rnn-lr", type=float, default=1e-3)
    parser.add_argument("--rnn-seq-len", type=int, default=32)
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument("--rnn-n-gauss", type=int, default=5)
    parser.add_argument("--cma-pop-size", type=int, default=64)
    parser.add_argument("--cma-generations", type=int, default=300)
    parser.add_argument("--cma-target-return", type=float, default=900.0)
    parser.add_argument("--cma-n-rollouts", type=int, default=16)
    parser.add_argument("--cma-sigma", type=float, default=0.1)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-rnn", action="store_true")
    parser.add_argument("--skip-cma", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(
        project=args.wandb_project,
        name=f"wm_carracing_s{args.seed}",
        config=vars(args), use_wandb=args.track,
    )

    data_dir = os.path.join(args.logdir, "data")

    if not args.skip_data:
        print(f"\n{'='*60}\nPhase 1: Collecting {args.num_rollouts} rollouts\n{'='*60}")
        collect_data(args.env_id, args.num_rollouts, args.max_steps, data_dir, args.seed)

    if not args.skip_vae:
        print(f"\n{'='*60}\nPhase 2: Training VAE\n{'='*60}")
        vae = train_vae(args, device, logger)
    else:
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(os.path.join(args.logdir, "vae.pt"), map_location=device, weights_only=True))

    if not args.skip_rnn:
        print(f"\n{'='*60}\nPhase 3: Training MDN-RNN\n{'='*60}")
        rnn = train_rnn(args, vae, device, logger)
    else:
        rnn = MDNRNN(latent_dim=args.vae_latent_size, hidden_dim=args.rnn_hidden_size, n_gauss=args.rnn_n_gauss).to(device)
        rnn.load_state_dict(torch.load(os.path.join(args.logdir, "rnn.pt"), map_location=device, weights_only=True))

    if not args.skip_cma:
        print(f"\n{'='*60}\nPhase 4: CMA-ES Controller\n{'='*60}")
        controller = train_controller_cmaes(args, vae, rnn, device, logger)
    else:
        controller = Controller(latent_dim=args.vae_latent_size, hidden_dim=args.rnn_hidden_size).to(device)
        controller.load_state_dict(torch.load(os.path.join(args.logdir, "controller.pt"), map_location=device, weights_only=True))

    if controller is not None:
        print(f"\n{'='*60}\nFinal Evaluation (100 rollouts)\n{'='*60}")
        rewards = []
        for i in range(100):
            r = rollout_agent(args.env_id, vae, rnn, controller, device)
            rewards.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/100  mean={np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        print(f"\nFinal: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        logger.log({"eval/mean_return": float(np.mean(rewards))}, step=0)

    logger.close()
    print("Done.")
