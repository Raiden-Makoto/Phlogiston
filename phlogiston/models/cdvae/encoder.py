"""CDVAE VAE encoder (see DESIGN.md §2).

Pools a crystal graph with the shared CrystalEncoder to an invariant graph
feature, then maps it to a Gaussian latent ``z`` (reparameterized). Separate
weights from the predictor model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from phlogiston.models.encoder import CrystalEncoder


@dataclass
class VAEOutput:
    z: torch.Tensor  # [B, latent_dim] sampled latent
    mu: torch.Tensor  # [B, latent_dim]
    logvar: torch.Tensor  # [B, latent_dim]


class CDVAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, mul: int = 128, **encoder_kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CrystalEncoder(mul=mul, **encoder_kwargs)
        self.to_mu = nn.Linear(mul, latent_dim)
        self.to_logvar = nn.Linear(mul, latent_dim)

    def forward(self, graph) -> VAEOutput:
        gf = self.encoder(graph).graph_feats  # [B, mul] invariant
        mu = self.to_mu(gf)
        logvar = self.to_logvar(gf)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # reparameterization
        return VAEOutput(z=z, mu=mu, logvar=logvar)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0, I)), averaged over the batch (>= 0)."""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
