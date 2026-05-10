"""World Models: VAE + MDN-RNN + Controller"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ASIZE = 3
IMG_SIZE = 64

class VAE(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 6, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x).reshape(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).reshape(-1, 1024, 1, 1)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, target, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / target.size(0)

class MDNRNN(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_gauss: int, action_dim: int = ASIZE):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gauss = n_gauss
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.mdn_linear = nn.Linear(hidden_dim, n_gauss * (2 * latent_dim + 1))
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor, hidden=None):
        B, T, _ = z.shape
        x = torch.cat([z, action], dim=-1)
        out, hidden = self.lstm(x) if hidden is None else self.lstm(x, hidden)
        mdn_out = self.mdn_linear(out).reshape(B, T, self.n_gauss, 2 * self.latent_dim + 1)
        mu = mdn_out[:, :, :, :self.latent_dim]
        sigma = torch.exp(mdn_out[:, :, :, self.latent_dim:2*self.latent_dim])
        log_pi = F.log_softmax(mdn_out[:, :, :, -1], dim=-1)
        return log_pi, mu, sigma, self.reward_head(out), self.done_head(out), hidden

    def initial_hidden(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )

    def forward_single(self, z: torch.Tensor, action: torch.Tensor, hidden):
        x = torch.cat([z, action], dim=-1).unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        return hidden, out.squeeze(1)

def gmm_loss(z_next: torch.Tensor, log_pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z_next = z_next.unsqueeze(2)
    log_probs = -0.5 * (
        ((z_next - mu) / sigma) ** 2 + 2 * torch.log(sigma) + np.log(2 * np.pi)
    ).sum(dim=-1)
    return -torch.logsumexp(log_pi + log_probs, dim=-1).mean()

class Controller(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int = ASIZE):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_params(self, flat_params: np.ndarray):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(flat_params[idx:idx+n]).reshape(p.shape).float())
            idx += n

    def get_params(self) -> np.ndarray:
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])
