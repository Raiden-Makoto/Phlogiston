"""Noise-level / timestep embedding for the CDVAE score decoder.

Sinusoidal (Fourier) embedding of a scalar diffusion noise level ``sigma`` (or
timestep) into invariant scalar features (`0e`) that are injected into node
features to condition the denoiser on the noise level.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class NoiseEmbedding(nn.Module):
    """Scalar noise level ``[B]`` -> embedding ``[B, dim]`` (invariant `0e`)."""

    def __init__(self, dim: int = 64, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        half = dim // 2
        # geometric frequencies, as in transformer positional / diffusion time embeddings
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1, 1).to(self.freqs)  # [B,1]
        args = sigma * self.freqs.reshape(1, -1)  # [B, half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
