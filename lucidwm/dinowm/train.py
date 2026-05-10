"""DINO-WM: World model training"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import DINOv2Encoder, TransitionViT

def train_world_model(encoder: DINOv2Encoder, transition: TransitionViT, dataset, args, device, logger):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(transition.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Training DINO-WM transition model: {args.epochs} epochs, {len(dataset)} samples")
    step = 0
    for epoch in range(args.epochs):
        transition.train()
        epoch_loss = 0.0
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                obs_dict, actions, _ = batch
                frames = obs_dict["visual"].to(device)
                actions = actions[:, :-1].to(device)
            else:
                frames = batch["frames"].to(device)
                actions = batch["actions"].to(device)
            B, T, C, H_img, W_img = frames.shape
            with torch.no_grad():
                all_z = encoder(frames.reshape(B * T, C, H_img, W_img))
                all_z = all_z.reshape(B, T, args.num_patches, args.dino_embed_dim)
            z_context = all_z[:, :-1]
            z_target = all_z[:, -1]
            z_pred = transition(z_context, actions)
            loss = F.mse_loss(z_pred, z_target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(transition.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        scheduler.step()
        avg = epoch_loss / len(loader)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{args.epochs}  loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")
        logger.log({"wm/loss": avg, "wm/lr": scheduler.get_last_lr()[0]}, step=step)
    ckpt_path = os.path.join(args.logdir, "dinowm_transition.pt")
    torch.save(transition.state_dict(), ckpt_path)
    print(f"  Saved transition model to {ckpt_path}")
    return transition
