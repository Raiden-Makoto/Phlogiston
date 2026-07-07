"""Equivariant vector readout.

Reads node features to a per-atom equivariant vector (`1o`) — e.g. the CDVAE
coord score (∇ log p over positions), which must rotate with the structure.
Distinct from ``ScalarReadout`` (invariant `0e`).
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn


class EquivariantVectorReadout(nn.Module):
    """Node features -> ``[N, 3*n_vectors]`` equivariant vectors (`n_vectors x 1o`)."""

    def __init__(self, irreps_in, n_vectors: int = 1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(f"{n_vectors}x1o")
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
