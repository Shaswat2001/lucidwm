"""World Models: Training functions for VAE, MDN-RNN, and CMA-ES controller"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import VAE, MDNRNN, Controller, vae_loss, gmm_loss
from .dataset import FrameDataset, SequenceDataset
from .env import rollout_agent

def train_vae(args, device, logger):
    dataset = FrameDataset(os.path.join(args.logdir, "data"))
    loader = DataLoader(dataset, batch_size=args.vae_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    vae = VAE().to(device)
    opt = optim.Adam(vae.parameters(), lr=args.vae_lr)
    print(f"Training VAE on {len(dataset)} frames, {args.vae_epochs} epochs")
    step = 0
    for epoch in range(args.vae_epochs):
        vae.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            step += 1
        avg = total / len(loader)
        print(f"  Epoch {epoch+1}/{args.vae_epochs}  loss={avg:.4f}")
        logger.log({"vae/loss": avg}, step=step)
    torch.save(vae.state_dict(), os.path.join(args.logdir, "vae.pt"))
    return vae

def train_rnn(args, vae, device, logger):
    dataset = SequenceDataset(os.path.join(args.logdir, "data"), vae, device, args.rnn_seq_len)
    loader = DataLoader(dataset, batch_size=args.rnn_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    rnn = MDNRNN(latent_dim=args.vae_latent_size, hidden_dim=args.rnn_hidden_size, n_gauss=args.rnn_n_gauss).to(device)
    opt = optim.Adam(rnn.parameters(), lr=args.rnn_lr)
    import torch.nn.functional as F
    print(f"Training MDN-RNN on {len(dataset)} sequences, {args.rnn_epochs} epochs")
    step = 0
    for epoch in range(args.rnn_epochs):
        rnn.train()
        total_gmm, total_rew = 0.0, 0.0
        for batch in loader:
            z = batch["z"].to(device)
            act = batch["action"].to(device)
            z_next = batch["z_next"].to(device)
            rew = batch["reward"].to(device)
            done = batch["done"].to(device)
            log_pi, mu, sigma, pred_rew, pred_done, _ = rnn(z, act)
            loss_g = gmm_loss(z_next, log_pi, mu, sigma)
            loss_r = F.mse_loss(pred_rew.squeeze(-1), rew)
            loss_d = F.binary_cross_entropy_with_logits(pred_done.squeeze(-1), done)
            loss = loss_g + loss_r + loss_d
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            opt.step()
            total_gmm += loss_g.item()
            total_rew += loss_r.item()
            step += 1
        print(f"  Epoch {epoch+1}/{args.rnn_epochs}  gmm={total_gmm/len(loader):.4f}  reward={total_rew/len(loader):.4f}")
        logger.log({"rnn/gmm_loss": total_gmm/len(loader)}, step=step)
    torch.save(rnn.state_dict(), os.path.join(args.logdir, "rnn.pt"))
    return rnn

def train_controller_cmaes(args, vae, rnn, device, logger):
    try:
        import cma
    except ImportError:
        print("ERROR: pip install cma")
        return None
    controller = Controller(latent_dim=args.vae_latent_size, hidden_dim=args.rnn_hidden_size).to(device)
    print(f"CMA-ES: {controller.num_params} params, pop={args.cma_pop_size}")
    es = cma.CMAEvolutionStrategy(
        controller.get_params(), args.cma_sigma,
        {"popsize": args.cma_pop_size, "seed": args.seed},
    )
    best_reward, best_params = -float("inf"), None
    for gen in range(args.cma_generations):
        solutions = es.ask()
        fitnesses = []
        for params in solutions:
            controller.set_params(np.array(params))
            rewards = [rollout_agent(args.env_id, vae, rnn, controller, device) for _ in range(args.cma_n_rollouts)]
            fitnesses.append(-np.mean(rewards))
        es.tell(solutions, fitnesses)
        gen_best = -min(fitnesses)
        if gen_best > best_reward:
            best_reward = gen_best
            best_params = solutions[np.argmin(fitnesses)].copy()
        print(f"  Gen {gen+1}/{args.cma_generations}  best={gen_best:.1f}  all-time={best_reward:.1f}")
        logger.log({"cma/gen_best": gen_best, "cma/best": best_reward}, step=gen)
        if best_reward >= args.cma_target_return:
            print(f"  Target {args.cma_target_return} reached!")
            break
        if es.stop():
            print(f"  CMA-ES converged at gen {gen+1}")
            break
    controller.set_params(np.array(best_params))
    torch.save(controller.state_dict(), os.path.join(args.logdir, "controller.pt"))
    print(f"  Best reward: {best_reward:.1f}")
    return controller
