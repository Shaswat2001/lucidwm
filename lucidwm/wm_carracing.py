"""LucidWM: World Models (Ha & Schmidhuber, 2018) on CarRacing

Paper: "World Models" (Ha & Schmidhuber, 2018)
Reference: https://worldmodels.github.io/

Architecture: VAE (32-dim latent) + MDN-RNN (LSTM + mixture density) + CMA-ES controller
Training:     3-phase: (1) collect random data, (2) train VAE, (3) train MDN-RNN, (4) CMA-ES in dream
Components:   None from Layer 1 (self-contained, tutorial entry point)
Env:          CarRacing-v2 (Gymnasium)
Target:       Score > 850 (human ~900)
"""

import os
import cv2
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gymnasium as gym

from lucidwm_utils.logger import Logger
from lucidwm_utils.misc import get_device, set_seed

ASIZE = 3        # action dimension (steer, gas, brake)
IMG_SIZE = 64    # frame size (64x64)

def parse_args():
    parser = argparse.ArgumentParser(description="LucidWM: World Models (Ha & Schmidhuber)")

    # Standard args
    parser.add_argument("--env-id", type=str, default="CarRacing-v3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", action="store_true", help="Track with W&B")
    parser.add_argument("--wandb-project", type=str, default="lucidwm")
    parser.add_argument("--logdir", type=str, default="exp/wm_carracing")

    # Phase 1: Data collection
    parser.add_argument("--num-rollouts", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)

    # Phase 2: VAE training
    parser.add_argument("--vae-epochs", type=int, default=30)
    parser.add_argument("--vae-batch-size", type=int, default=128)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-latent-size", type=int, default=32)

    # Phase 3: MDN-RNN training
    parser.add_argument("--rnn-epochs", type=int, default=30)
    parser.add_argument("--rnn-batch-size", type=int, default=64)
    parser.add_argument("--rnn-lr", type=float, default=1e-3)
    parser.add_argument("--rnn-seq-len", type=int, default=32)
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument("--rnn-n-gauss", type=int, default=5)

    # Phase 4: CMA-ES controller
    parser.add_argument("--cma-pop-size", type=int, default=64)
    parser.add_argument("--cma-generations", type=int, default=300)
    parser.add_argument("--cma-target-return", type=float, default=900.0)
    parser.add_argument("--cma-n-rollouts", type=int, default=16)
    parser.add_argument("--cma-sigma", type=float, default=0.1)

    # Flags to skip phases
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-rnn", action="store_true")
    parser.add_argument("--skip-cma", action="store_true")

    return parser.parse_args()

def collect_data(env_id, num_rollouts, max_steps, data_dir, seed):
    """Collect rollouts with brownian random policy."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_rollouts):
        env = gym.make(env_id, render_mode=None)
        obs, _ = env.reset(seed=seed + i)
        frames, actions, rewards, dones = [], [], [], []
        action = np.zeros(ASIZE, dtype=np.float32)

        for _ in range(max_steps):
            frame = cv2.resize(obs, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            frames.append(frame)

            # Brownian policy: small random perturbation each step
            action = np.clip(action + 0.1 * np.random.randn(ASIZE).astype(np.float32), -1, 1)
            env_action = action.copy()
            env_action[1] = np.clip((env_action[1] + 1) / 2, 0, 1)  # gas [0,1]
            env_action[2] = np.clip((env_action[2] + 1) / 2, 0, 1)  # brake [0,1]

            obs, reward, terminated, truncated, _ = env.step(env_action)
            actions.append(action.copy())
            rewards.append(reward)
            dones.append(terminated or truncated)
            if terminated or truncated:
                break

        env.close()
        np.savez_compressed(
            os.path.join(data_dir, f"episode_{i:05d}.npz"),
            obs=np.array(frames, dtype=np.uint8),
            action=np.array(actions, dtype=np.float32),
            reward=np.array(rewards, dtype=np.float32),
            done=np.array(dones, dtype=np.bool_),
        )
        if (i + 1) % 100 == 0:
            print(f"  Collected {i+1}/{num_rollouts} rollouts")

# Vision Model (V): Variational Autoencoder
class VAE(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 4 conv layers, stride 2, output (B, 256, 2, 2) = (B, 1024) flat
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2),  # -> (B, 32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),            # -> (B, 64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),           # -> (B, 128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),          # -> (B, 256, 2, 2)
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        # Decoder: linear -> reshape -> 4 deconv layers
        self.fc_decode = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),  # -> (B, 128, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),    # -> (B, 64, 13, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),     # -> (B, 32, 30, 30)
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 6, stride=2),  # -> (B, 3, 64, 64)
            nn.Sigmoid(),
        )

    def encode(self, x):
        """(B, 3, 64, 64) -> mu (B, L), logvar (B, L)."""
        h = self.encoder(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def sample(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        """(B, L) -> (B, 3, 64, 64) in [0, 1]."""
        h = self.fc_decode(z).reshape(-1, 1024, 1, 1)
        return self.decoder(h)

    def forward(self, x):
        """Returns: recon (B,3,64,64), mu (B,L), logvar (B,L)."""
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon, target, mu, logvar):
    """VAE loss = MSE reconstruction + KL(q(z|x) || N(0,1))."""
    recon_loss = F.mse_loss(recon, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / target.size(0)

# Memory Model (M): MDN-RNN
class MDNRNN(nn.Module):
    """Mixture Density Network + LSTM.

    Predicts p(z_{t+1} | a_t, z_t, h_t) as mixture of N_GAUSS Gaussians.
    Also predicts reward and done logits.

    Input per step: [z_t, a_t] concatenated
    Output: mixture params (log_pi, mu, sigma) for next latent, plus reward/done
    """

    def __init__(self, latent_dim, hidden_dim, n_gauss, action_dim=ASIZE):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gauss = n_gauss

        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)

        # MDN head: per Gaussian, output mu (L) + log_sigma (L) + logit (1)
        self.mdn_linear = nn.Linear(hidden_dim, n_gauss * (2 * latent_dim + 1))
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, z, action, hidden=None):
        """Sequence forward pass.

        Args:
            z:      (B, T, L)
            action: (B, T, A)
            hidden: optional (h0, c0) each (1, B, R)

        Returns:
            log_pi: (B, T, N)
            mu:     (B, T, N, L)
            sigma:  (B, T, N, L)
            reward: (B, T, 1)
            done:   (B, T, 1)
            hidden: (h_n, c_n)
        """
        B, T, _ = z.shape
        x = torch.cat([z, action], dim=-1)  # (B, T, L+A)
        out, hidden = self.lstm(x) if hidden is None else self.lstm(x, hidden)
        # out: (B, T, R)

        mdn_out = self.mdn_linear(out)  # (B, T, N*(2L+1))
        mdn_out = mdn_out.reshape(B, T, self.n_gauss, 2 * self.latent_dim + 1)

        mu = mdn_out[:, :, :, :self.latent_dim]                           # (B,T,N,L)
        sigma = torch.exp(mdn_out[:, :, :, self.latent_dim:2*self.latent_dim])  # (B,T,N,L)
        log_pi = F.log_softmax(mdn_out[:, :, :, -1], dim=-1)              # (B,T,N)

        return log_pi, mu, sigma, self.reward_head(out), self.done_head(out), hidden

    def initial_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device))

    def forward_single(self, z, action, hidden):
        """Single-step forward for controller rollout.

        Args: z (B,L), action (B,A), hidden (h,c)
        Returns: hidden, h_out (B,R)
        """
        x = torch.cat([z, action], dim=-1).unsqueeze(1)  # (B, 1, L+A)
        out, hidden = self.lstm(x, hidden)
        return hidden, out.squeeze(1)

def gmm_loss(z_next, log_pi, mu, sigma):
    """Negative log-likelihood of z_next under the Gaussian mixture.

    Args:
        z_next: (B, T, L) target
        log_pi: (B, T, N) log mixing coefficients
        mu:     (B, T, N, L) means
        sigma:  (B, T, N, L) stds

    Returns: scalar NLL
    """
    z_next = z_next.unsqueeze(2)  # (B, T, 1, L)

    # Log prob under each Gaussian, summed over latent dims
    log_probs = -0.5 * (
        ((z_next - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi)
    ).sum(dim=-1)  # (B, T, N)

    # Log-sum-exp over mixture components
    return -torch.logsumexp(log_pi + log_probs, dim=-1).mean()

# Controller (C): Linear + CMA-ES
class Controller(nn.Module):
    """Linear controller: a_t = tanh(W [z_t; h_t] + b).

    Deliberately tiny so model capacity lives in V and M.
    """

    def __init__(self, latent_dim, hidden_dim, action_dim=ASIZE):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        """z: (B, L), h: (B, R) -> action: (B, A) in [-1, 1]."""
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_params(self, flat_params):
        """Set from flat numpy array (for CMA-ES)."""
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(flat_params[idx:idx+n]).reshape(p.shape).float())
            idx += n

    def get_params(self):
        """Get as flat numpy array."""
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

# Datasets 
class FrameDataset(Dataset):
    """Dataset of individual frames for VAE training."""

    def __init__(self, data_dir):
        frames = []
        for ep_file in sorted(Path(data_dir).glob("episode_*.npz")):
            frames.append(np.load(ep_file)["obs"])
        self.frames = np.concatenate(frames, axis=0)  # (N, 64, 64, 3)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        f = self.frames[idx].astype(np.float32) / 255.0  # (64, 64, 3)
        return torch.from_numpy(f.transpose(2, 0, 1))     # (3, 64, 64)

class SequenceDataset(Dataset):
    """Dataset of VAE-encoded sequences for MDN-RNN training."""

    def __init__(self, data_dir, vae, device, seq_len):
        self.seq_len = seq_len
        self.episodes = []
        vae.eval()

        with torch.no_grad():
            for ep_file in sorted(Path(data_dir).glob("episode_*.npz")):
                data = np.load(ep_file)
                obs = torch.from_numpy(
                    data["obs"].astype(np.float32) / 255.0
                ).permute(0, 3, 1, 2).to(device)  # (T, 3, 64, 64)

                # Encode in chunks
                zs = []
                for i in range(0, len(obs), 256):
                    mu, _ = vae.encode(obs[i:i+256])
                    zs.append(mu.cpu().numpy())

                self.episodes.append({
                    "z": np.concatenate(zs, axis=0),
                    "action": data["action"].astype(np.float32),
                    "reward": data["reward"].astype(np.float32),
                    "done": data["done"].astype(np.float32),
                })

        # Build index of valid (episode, start) pairs
        self.index = []
        for ep_idx, ep in enumerate(self.episodes):
            for start in range(len(ep["z"]) - seq_len - 1):
                self.index.append((ep_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, start = self.index[idx]
        ep = self.episodes[ep_idx]
        s, s1 = slice(start, start + self.seq_len), slice(start+1, start+self.seq_len+1)
        return {
            "z": torch.from_numpy(ep["z"][s]),
            "action": torch.from_numpy(ep["action"][s]),
            "z_next": torch.from_numpy(ep["z"][s1]),
            "reward": torch.from_numpy(ep["reward"][s]),
            "done": torch.from_numpy(ep["done"][s]),
        }

# ═══════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def train_vae(args, device, logger):
    dataset = FrameDataset(os.path.join(args.logdir, "data"))
    loader = DataLoader(dataset, batch_size=args.vae_batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
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
    loader = DataLoader(dataset, batch_size=args.rnn_batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
    rnn = MDNRNN(latent_dim=args.vae_latent_size,
                 hidden_dim=args.rnn_hidden_size,
                 n_gauss=args.rnn_n_gauss).to(device)
    opt = optim.Adam(rnn.parameters(), lr=args.rnn_lr)

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

        print(f"  Epoch {epoch+1}/{args.rnn_epochs}  gmm={total_gmm/len(loader):.4f}  "
              f"reward={total_rew/len(loader):.4f}")
        logger.log({"rnn/gmm_loss": total_gmm/len(loader)}, step=step)

    torch.save(rnn.state_dict(), os.path.join(args.logdir, "rnn.pt"))
    return rnn

def rollout_agent(env_id, vae, rnn, controller, device, max_steps=1000, render=False):
    """Run a single episode with the full V+M+C agent. Returns total reward."""
    import cv2
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset()
    vae.eval(); rnn.eval(); controller.eval()

    hidden = rnn.initial_hidden(1, device)
    total_reward = 0.0
    prev_action = torch.zeros(1, ASIZE, device=device)

    with torch.no_grad():
        for _ in range(max_steps):
            frame = cv2.resize(obs, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            frame_t = torch.from_numpy(
                frame.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 64, 64)

            mu, _ = vae.encode(frame_t)                  # (1, L)
            h = hidden[0].squeeze(0)                      # (1, R)
            action = controller(mu, h)                    # (1, A)
            hidden, _ = rnn.forward_single(mu, prev_action, hidden)
            prev_action = action

            env_action = action.squeeze(0).cpu().numpy()
            env_action[1] = np.clip((env_action[1] + 1) / 2, 0, 1)
            env_action[2] = np.clip((env_action[2] + 1) / 2, 0, 1)

            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            if terminated or truncated:
                break
    env.close()
    return total_reward

def train_controller_cmaes(args, vae, rnn, device, logger):
    """Train controller via CMA-ES to maximize expected reward."""
    try:
        import cma
    except ImportError:
        print("ERROR: pip install cma")
        return None

    controller = Controller(latent_dim=args.vae_latent_size,
                            hidden_dim=args.rnn_hidden_size).to(device)
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
            rewards = [rollout_agent(args.env_id, vae, rnn, controller, device)
                       for _ in range(args.cma_n_rollouts)]
            fitnesses.append(-np.mean(rewards))  # CMA-ES minimizes

        es.tell(solutions, fitnesses)
        gen_best = -min(fitnesses)
        if gen_best > best_reward:
            best_reward = gen_best
            best_params = solutions[np.argmin(fitnesses)].copy()

        print(f"  Gen {gen+1}/{args.cma_generations}  best={gen_best:.1f}  "
              f"all-time={best_reward:.1f}")
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

if __name__ == "__main__":
    args = parse_args() 
    set_seed(args.seed)
    device = get_device()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    logger = Logger(project=args.wandb_project, name=f"wm_carracing_s{args.seed}",
                    config=vars(args), use_wandb=args.track)

    data_dir = os.path.join(args.logdir, "data")
    if not args.skip_data:
        print(f"\n{'='*60}\nPhase 1: Collecting {args.num_rollouts} rollouts\n{'='*60}")
        collect_data(args.env_id, args.num_rollouts, args.max_steps, data_dir, args.seed)
    
    if not args.skip_vae:
        print(f"\n{'='*60}\nPhase 2: Training VAE\n{'='*60}")
        vae = train_vae(args, device, logger)
    else:
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(os.path.join(args.logdir, "vae.pt"),
                                       map_location=device, weights_only=True))
        
    if not args.skip_rnn:
        print(f"\n{'='*60}\nPhase 3: Training MDN-RNN\n{'='*60}")
        rnn = train_rnn(args, vae, device, logger)
    else:
        rnn = MDNRNN(latent_dim=args.vae_latent_size,
                     hidden_dim=args.rnn_hidden_size,
                     n_gauss=args.rnn_n_gauss).to(device)
        rnn.load_state_dict(torch.load(os.path.join(args.logdir, "rnn.pt"),
                                       map_location=device, weights_only=True))

    if not args.skip_cma:
        print(f"\n{'='*60}\nPhase 4: CMA-ES Controller\n{'='*60}")
        controller = train_controller_cmaes(args, vae, rnn, device, logger)
    else:
        controller = Controller(latent_dim=args.vae_latent_size,
                                hidden_dim=args.rnn_hidden_size).to(device)
        controller.load_state_dict(torch.load(os.path.join(args.logdir, "controller.pt"),
                                              map_location=device, weights_only=True))

    if controller is not None:
        print(f"\n{'='*60}\nFinal Evaluation (100 rollouts)\n{'='*60}")
        rewards = []
        for i in range(100):
            r = rollout_agent(args.env_id, vae, rnn, controller, device)
            rewards.append(r)
            if (i+1) % 10 == 0:
                print(f"  {i+1}/100  mean={np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        print(f"\nFinal: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        logger.log({"eval/mean_return": float(np.mean(rewards))}, step=0)

    logger.close()
    print("Done.")