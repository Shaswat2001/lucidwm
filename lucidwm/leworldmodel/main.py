"""LeWorldModel: Entry point

Usage:
  python -m lucidwm.leworldmodel.main --env-id pusht
  python -m lucidwm.leworldmodel.main --skip-train --logdir exp/lewm
"""
import os
import argparse
from pathlib import Path

import torch

from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import set_seed, get_device
from .encoder import LeWMEncoder
from .predictor import LeWMPredictor, Projector, ActionEncoder
from .loss import SIGReg
from .planner import plan_cem
from .dataset import LeWMH5Dataset, download_h5_dataset
from .train import train

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: LeWM")
    parser.add_argument("--env-id", type=str, default="pusht")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--logdir", type=str, default="exp/lewm")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to pusht_expert_train.h5. If not provided, downloads automatically.")
    parser.add_argument("--hf-repo", type=str, default="quentinll/lewm-pusht")
    parser.add_argument("--hf-filename", type=str, default="pusht_expert_train.h5.zst")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--sigreg-lambda", type=float, default=0.09)
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--enc-patch-size", type=int, default=14)
    parser.add_argument("--enc-embed-dim", type=int, default=192)
    parser.add_argument("--enc-depth", type=int, default=12)
    parser.add_argument("--enc-heads", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--action-smoothed-dim", type=int, default=16)
    parser.add_argument("--action-emb-dim", type=int, default=192)
    parser.add_argument("--pred-dim", type=int, default=512)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=16)
    parser.add_argument("--pred-dim-head", type=int, default=64)
    parser.add_argument("--pred-mlp-dim", type=int, default=2048)
    parser.add_argument("--pred-dropout", type=float, default=0.1)
    parser.add_argument("--pred-emb-dropout", type=float, default=0.0)
    parser.add_argument("--plan-horizon", type=int, default=10)
    parser.add_argument("--cem-candidates", type=int, default=200)
    parser.add_argument("--cem-elites", type=int, default=20)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(
        project=args.wandb_project,
        name=f"lewm_{args.env_id}_s{args.seed}",
        config=vars(args),
        use_wandb=args.track,
    )

    encoder = LeWMEncoder(args).to(device)
    action_encoder = ActionEncoder(
        input_dim=args.action_dim,
        smoothed_dim=args.action_smoothed_dim,
        emb_dim=args.action_emb_dim,
    ).to(device)
    predictor = LeWMPredictor(args).to(device)
    enc_projector = Projector(encoder.embed_dim, args.latent_dim).to(device)
    pred_projector = Projector(args.pred_dim, args.latent_dim).to(device)
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_projections).to(device)

    if not args.skip_train:
        print(f"\n{'='*60}\nTraining LeWM\n{'='*60}")
        if args.data_path is not None:
            h5_path = args.data_path
        else:
            h5_path = download_h5_dataset(
                repo_id=args.hf_repo,
                filename=args.hf_filename,
                cache_dir=os.path.join(args.logdir, "data"),
            )
        dataset = LeWMH5Dataset(
            h5_path=h5_path,
            img_size=args.img_size,
            frameskip=args.frameskip,
            history_size=args.history_size,
        )
        encoder, action_encoder, predictor, enc_projector, pred_projector = train(
            encoder, action_encoder, predictor, enc_projector, pred_projector,
            sigreg, dataset, args, device, logger,
        )
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "action_encoder": action_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "enc_projector": enc_projector.state_dict(),
                "pred_projector": pred_projector.state_dict(),
            },
            os.path.join(args.logdir, "lewm.pt"),
        )
    else:
        ckpt = torch.load(os.path.join(args.logdir, "lewm.pt"), map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        action_encoder.load_state_dict(ckpt["action_encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        enc_projector.load_state_dict(ckpt["enc_projector"])
        pred_projector.load_state_dict(ckpt["pred_projector"])

    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [encoder, action_encoder, predictor, enc_projector, pred_projector]
    )
    print(f"\nLeWM ready. Total: {total_params/1e6:.1f}M params")
    print("Use plan_cem() with obs_history and action_history for latent goal-reaching.")
    logger.close()
    print("Done.")
