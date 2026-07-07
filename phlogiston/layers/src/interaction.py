"""MACE interaction block.

See ``docs/interaction.md``.

v1 (this file, ν=1): the **A-basis** 2-body message only — a NequIP-style
equivariant convolution: for each edge, a weight-parameterized CG tensor product
couples the neighbor's features with the edge spherical harmonics; messages are
summed over neighbors (neighbor-normalized) and combined with a species-dependent
skip. The ν≥2 **symmetric contraction** (higher body order) is added in v2.

Output is the pre-nonlinearity node update in ``irreps_out``; the caller applies
the gate (``EquivariantGate``).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from e3nn import o3

from phlogiston.layers.src.radial import RadialBasis
from phlogiston.layers.src.linear import SpeciesLinear


class Interaction(nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, l_feat: int = 2,
                 r_max: float = 6.0, n_bessel: int = 8, num_species: int = 119):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)

        # --- A-basis tensor product (uvu: keep neighbor multiplicities) ---
        irreps_A: list[tuple[int, o3.Irrep]] = []
        instructions: list[tuple] = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_sh) in enumerate(self.irreps_sh):
                for ir_out in ir_in * ir_sh:                 # CG selection rule
                    if ir_out.l <= l_feat:
                        k = len(irreps_A)
                        irreps_A.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        self.irreps_A = o3.Irreps(irreps_A)
        self.tp = o3.TensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_A, instructions,
            shared_weights=False, internal_weights=False,
        )
        # radial MLP supplies the per-edge tensor-product path weights
        self.radial = RadialBasis(self.tp.weight_numel, r_max=r_max, n_bessel=n_bessel)
        # message projection A -> output irreps
        self.msg_linear = o3.Linear(self.irreps_A, self.irreps_out)
        # species-dependent self-connection (residual)
        self.skip = SpeciesLinear(self.irreps_in, self.irreps_out, num_species)

    def forward(self, h, edge_index, edge_len, sh, z, avg_num_neighbors):
        """h [N, dim_in]; edge_index [2,E] (row0 center i, row1 neighbor j);
        edge_len [E]; sh [E, dim_sh]; z [N]; avg_num_neighbors: float."""
        from torch_geometric.utils import scatter          # pure-torch, ROCm-safe

        h_j = h[edge_index[1]]                              # neighbor features
        w = self.radial(edge_len)                          # [E, weight_numel]
        msg = self.tp(h_j, sh, w)                           # [E, dim_A]
        A = scatter(msg, edge_index[0], dim=0, dim_size=h.shape[0], reduce="sum")
        A = A / math.sqrt(avg_num_neighbors)               # neighbor normalization
        return self.msg_linear(A) + self.skip(h, z)
