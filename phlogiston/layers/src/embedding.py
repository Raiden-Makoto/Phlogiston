"""Atomic-number node embedding.

See ``embedding.md``. Produces the initial node scalars (``mul x 0e``) from
species; higher-ℓ features grow later through the interaction layers. Optional
fixed element-descriptor seeding is off by default (encoder DESIGN §8).
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn

Z_MAX = 118  # index directly by atomic number


class AtomEmbedding(nn.Module):
    """z [N] (int64 atomic numbers) -> h0 [N, mul] as irreps ``mul x 0e``."""

    def __init__(self, mul: int = 128, z_max: int = Z_MAX):
        super().__init__()
        self.mul = int(mul)
        self.embed = nn.Embedding(z_max + 1, self.mul)
        self.irreps_out = o3.Irreps(f"{self.mul}x0e")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.embed(z)

    def __repr__(self) -> str:
        return f"AtomEmbedding(mul={self.mul}, irreps_out={self.irreps_out})"
