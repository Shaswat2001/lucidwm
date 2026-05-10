"""LeWorldModel: Training loop"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import LeWMH5Dataset

def train(encoder, action_encoder, predictor, enc_projector, pred_projector, sigreg, dataset, args, device, logger):
    has_gpu = torch.cuda.is_available()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=has_gpu,
        drop_last=True,
        worker_init_fn=LeWMH5Dataset.worker_init_fn,
        persistent_workers=True,
    )
    params = (
        list(encoder.parameters())
        + list(action_encoder.parameters())
        + list(predictor.parameters())
        + list(enc_projector.parameters())
        + list(pred_projector.parameters())
    )
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"Training LeWM: {args.epochs} epochs, {len(dataset)} samples")
    print(f"  History size: {args.history_size}")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    print(f"  Action encoder: {sum(p.numel() for p in action_encoder.parameters())/1e3:.1f}K params")
    print(f"  Predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.1f}M params")
    print(f"  Enc projector: {sum(p.numel() for p in enc_projector.parameters())/1e3:.1f}K params")
    print(f"  Pred projector: {sum(p.numel() for p in pred_projector.parameters())/1e3:.1f}K params")
    step = 0
    for epoch in range(args.epochs):
        encoder.train()
        action_encoder.train()
        predictor.train()
        enc_projector.train()
        pred_projector.train()
        epoch_pred_loss = 0.0
        epoch_sig_loss = 0.0
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            bsz, seq_len, c, h_img, w_img = obs.shape
            flat_obs = obs.reshape(bsz * seq_len, c, h_img, w_img)
            z_raw = encoder(flat_obs).reshape(bsz, seq_len, -1)
            emb = enc_projector(z_raw.reshape(bsz * seq_len, -1)).reshape(bsz, seq_len, -1)
            act_emb = action_encoder(action)
            pred_raw = predictor(emb[:, :-1], act_emb)
            pred = pred_projector(pred_raw.reshape(bsz * args.history_size, -1)).reshape(
                bsz, args.history_size, -1
            )
            target = emb[:, 1:]
            pred_loss = F.mse_loss(pred, target)
            sig_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + args.sigreg_lambda * sig_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            epoch_pred_loss += pred_loss.item()
            epoch_sig_loss += sig_loss.item()
            step += 1
        scheduler.step()
        avg_pred = epoch_pred_loss / len(loader)
        avg_sig = epoch_sig_loss / len(loader)
        if epoch % 10 == 0:
            print(
                f"  Epoch {epoch}/{args.epochs}  pred={avg_pred:.6f}  "
                f"sigreg={avg_sig:.6f}  lr={scheduler.get_last_lr()[0]:.2e}"
            )
        logger.log({"train/pred_loss": avg_pred, "train/sigreg": avg_sig}, step=step)
    return encoder, action_encoder, predictor, enc_projector, pred_projector
