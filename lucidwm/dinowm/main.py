"""DINO-WM: Entry point

Usage:
  python -m lucidwm.dinowm.main --env-id pusht --data-dir data/pusht --seed 1
  python -m lucidwm.dinowm.main --env-id pusht --data-dir data/pusht --skip-train
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import set_seed, get_device
from .model import DINOv2Encoder, TransitionViT
from .dataset import TrajectoryDataset, TrajSlicerDataset
from .train import train_world_model
from .planner import plan_cem

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: DINO-WM")
    parser.add_argument("--env-id", type=str, default="pusht")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--logdir", type=str, default="exp/dinowm")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino-embed-dim", type=int, default=384)
    parser.add_argument("--dino-patch-size", type=int, default=14)
    parser.add_argument("--num-patches", type=int, default=256)
    parser.add_argument("--vit-depth", type=int, default=6)
    parser.add_argument("--vit-heads", type=int, default=16)
    parser.add_argument("--vit-mlp-dim", type=int, default=2048)
    parser.add_argument("--plan-horizon", type=int, default=10)
    parser.add_argument("--cem-candidates", type=int, default=200)
    parser.add_argument("--cem-elites", type=int, default=20)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--mpc-steps", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(
        project=args.wandb_project,
        name=f"dinowm_{args.env_id}_s{args.seed}",
        config=vars(args), use_wandb=args.track,
    )

    print("Loading DINOv2 encoder (frozen)...")
    encoder = DINOv2Encoder(args.dino_model).to(device)
    print(f"  DINOv2 loaded: {args.dino_model}, embed_dim={args.dino_embed_dim}, patches={args.num_patches}")

    transition = TransitionViT(
        embedding_dim=args.dino_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        mlp_dim=args.vit_mlp_dim,
        num_patches=args.num_patches,
        action_dim=args.action_dim,
        context_len=args.context_len,
    ).to(device)
    n_params = sum(p.numel() for p in transition.parameters() if p.requires_grad)
    print(f"  Transition model: {n_params / 1e6:.1f}M trainable params")

    if not args.skip_train:
        print(f"\n{'='*60}\nPhase 1: Training DINO-WM\n{'='*60}")
        pusht_dataset = TrajectoryDataset(data_path=args.data_dir, img_size=args.img_size)
        num_frames = args.context_len + 1
        dataset = TrajSlicerDataset(pusht_dataset, num_frames, frameskip=args.frameskip)
        transition = train_world_model(encoder, transition, dataset, args, device, logger)
    else:
        print("Loading existing transition model...")
        ckpt_path = os.path.join(args.logdir, "dinowm_transition.pt")
        transition.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    print(f"\n{'='*60}\nPhase 2: Visual Goal Planning (CEM)\n{'='*60}")
    print("To plan, provide a goal image and current observation frames.")
    print("Example usage in code:")
    print("  goal_img = load_and_preprocess(goal_path)  # (3, 224, 224)")
    print("  goal_z = encoder(goal_img.unsqueeze(0))     # (1, N, E)")
    print("  action = plan_cem(encoder, transition, context_frames, goal_z, ...)")

    logger.close()
    print("\nDone.")
