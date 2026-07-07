"""Bessel radial basis + smooth polynomial cutoff + weight MLP.

See ``radial.md``. Maps each (invariant) edge length to the per-edge weights that
parameterize the interaction tensor product, decaying smoothly to zero at the
cutoff so messages are continuous as neighbors enter/leave the cutoff sphere.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class BesselBasis(nn.Module):
    """DimeNet Bessel radial basis: b_n(d) = sqrt(2/r_max)·sin(nπ d/r_max)/d."""

    def __init__(self, r_max: float = 6.0, n_bessel: int = 8):
        super().__init__()
        self.r_max = float(r_max)
        self.prefactor = math.sqrt(2.0 / self.r_max)
        # n·π for n = 1..n_bessel
        self.register_buffer("freqs", math.pi * torch.arange(1, n_bessel + 1, dtype=torch.float64))

    def forward(self, d: torch.Tensor) -> torch.Tensor:  # d [E]
        d = d.unsqueeze(-1)  # [E,1]
        freqs = self.freqs.to(d.dtype)
        return self.prefactor * torch.sin(freqs * d / self.r_max) / d  # [E, n_bessel]


class PolynomialCutoff(nn.Module):
    """C¹ polynomial envelope (order p): 1 at d=0, →0 with zero slope at r_max."""

    def __init__(self, r_max: float = 6.0, p: int = 6):
        super().__init__()
        self.r_max = float(r_max)
        self.p = int(p)

    def forward(self, d: torch.Tensor) -> torch.Tensor:  # d [E]
        p, x = self.p, d / self.r_max
        env = (
            1.0
            - (p + 1) * (p + 2) / 2.0 * x**p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2.0 * x ** (p + 2)
        )
        return env * (d < self.r_max)  # exactly 0 beyond cutoff


class RadialBasis(nn.Module):
    """edge_len [E] -> radial weights [E, n_out], envelope-damped at the cutoff.

    ``n_out`` is set by the consumer (an interaction TensorProduct's weight_numel).
    """

    def __init__(
        self, n_out: int, r_max: float = 6.0, n_bessel: int = 8, p: int = 6, hidden=(64, 64, 64)
    ):
        super().__init__()
        self.bessel = BesselBasis(r_max, n_bessel)
        self.cutoff = PolynomialCutoff(r_max, p)
        dims = [n_bessel, *hidden, n_out]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)
        self.n_out = n_out

    def forward(self, edge_len: torch.Tensor) -> torch.Tensor:  # [E]
        w = self.mlp(self.bessel(edge_len))  # [E, n_out]
        return w * self.cutoff(edge_len).unsqueeze(-1)  # damp to 0 at r_max
