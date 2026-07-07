"""Equivariant linear layers.

See ``docs/linear.md``. `EquivariantLinear` wraps `o3.Linear` (per-irrep channel
mixing, no cross-ℓ leakage). `SpeciesLinear` is the MACE self-connection: a
separate equivariant linear per element, selected by atomic number.
"""

from __future__ import annotations

import torch
from torch import nn
from e3nn import o3

Z_MAX = 118


class EquivariantLinear(nn.Module):
    def __init__(self, irreps_in, irreps_out, biases: bool = True):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.linear = o3.Linear(self.irreps_in, self.irreps_out, biases=biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SpeciesLinear(nn.Module):
    """Per-element equivariant linear (self-connection / skip).

    Weights are looked up per node by atomic number `z`; the map is equivariant
    (weights are scalars) but element-specific.
    """

    def __init__(self, irreps_in, irreps_out, num_species: int = Z_MAX + 1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        # external, non-shared weights: one weight vector per species.
        self.linear = o3.Linear(self.irreps_in, self.irreps_out,
                                shared_weights=False, internal_weights=False, biases=False)
        self.weight = nn.Parameter(torch.randn(num_species, self.linear.weight_numel))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.weight[z])
