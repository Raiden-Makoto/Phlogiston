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
from e3nn import o3
from torch import nn

from phlogiston.layers.src.linear import SpeciesLinear
from phlogiston.layers.src.radial import RadialBasis


def _uuu_product(irreps1, irreps2, l_feat: int):
    """Per-channel (uuu) equivariant product irreps1 x irreps2 -> irreps_out.

    Keeps multiplicity (no mul^2 blow-up); learnable per-path weights. Requires
    equal multiplicities (true throughout the encoder). Returns (tp, irreps_out).
    """
    out: list[tuple[int, object]] = []
    instr: list[tuple] = []
    for i, (mul1, ir1) in enumerate(irreps1):
        for j, (mul2, ir2) in enumerate(irreps2):
            if mul1 != mul2:
                continue
            for ir_out in ir1 * ir2:  # CG selection (parity too)
                if ir_out.l <= l_feat:
                    k = len(out)
                    out.append((mul1, ir_out))
                    instr.append((i, j, k, "uuu", True))
    irreps_out = o3.Irreps(out)
    tp = o3.TensorProduct(
        irreps1, irreps2, irreps_out, instr, shared_weights=True, internal_weights=True
    )
    return tp, irreps_out


class SymmetricContraction(nn.Module):
    """Raise the atomic basis A to body order ``correlation`` (the ACE step).

    order-1 = linear(A); order-k = per-channel product of the (k-1)-order feature
    with A. To keep it efficient, the running feature is projected back to the
    (bounded) input irreps after each order, so every product stays small
    instead of the irreps exploding combinatorially. The sum over orders 1..ν
    (each contracted to ``irreps_out``) is the product basis B. All ops
    equivariant (uuu tensor products + equivariant linears).
    """

    def __init__(self, irreps_in, irreps_out, correlation: int = 3, l_feat: int = 2):
        super().__init__()
        assert correlation >= 1
        self.correlation = correlation
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        self.lin1 = o3.Linear(irreps_in, irreps_out)
        self.prods = nn.ModuleList()  # order-k product TP (bounded x bounded)
        self.to_out = nn.ModuleList()  # project each order's product -> irreps_out
        self.to_next = nn.ModuleList()  # project product -> irreps_in for next order
        for order in range(2, correlation + 1):
            tp, ir = _uuu_product(irreps_in, irreps_in, l_feat)
            self.prods.append(tp)
            self.to_out.append(o3.Linear(ir, irreps_out))
            if order < correlation:
                self.to_next.append(o3.Linear(ir, irreps_in))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B = self.lin1(A)
        prev = A  # running feature (irreps_in)
        for i, (tp, out) in enumerate(zip(self.prods, self.to_out, strict=False)):
            prod = tp(prev, A)  # bounded product
            B = B + out(prod)
            if i < len(self.to_next):
                prev = self.to_next[i](prod)  # bound for next order
        return B


class Interaction(nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_sh,
        irreps_out,
        l_feat: int = 2,
        r_max: float = 6.0,
        n_bessel: int = 8,
        num_species: int = 119,
        correlation: int = 1,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)

        # --- A-basis tensor product (uvu: keep neighbor multiplicities) ---
        irreps_A: list[tuple[int, o3.Irrep]] = []
        instructions: list[tuple] = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_sh) in enumerate(self.irreps_sh):
                for ir_out in ir_in * ir_sh:  # CG selection rule
                    if ir_out.l <= l_feat:
                        k = len(irreps_A)
                        irreps_A.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        self.irreps_A = o3.Irreps(irreps_A)
        self.tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_A,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # radial MLP supplies the per-edge tensor-product path weights
        self.radial = RadialBasis(self.tp.weight_numel, r_max=r_max, n_bessel=n_bessel)
        # A -> message: order-1 linear (v1) or symmetric contraction (v2, ν≥2)
        self.correlation = correlation
        if correlation <= 1:
            self.msg = o3.Linear(self.irreps_A, self.irreps_out)
        else:
            self.msg = SymmetricContraction(
                self.irreps_A, self.irreps_out, correlation=correlation, l_feat=l_feat
            )
        # species-dependent self-connection (residual)
        self.skip = SpeciesLinear(self.irreps_in, self.irreps_out, num_species)

    def forward(self, h, edge_index, edge_len, sh, z, avg_num_neighbors):
        """h [N, dim_in]; edge_index [2,E] (row0 center i, row1 neighbor j);
        edge_len [E]; sh [E, dim_sh]; z [N]; avg_num_neighbors: float."""
        from torch_geometric.utils import scatter  # pure-torch, ROCm-safe

        h_j = h[edge_index[1]]  # neighbor features
        w = self.radial(edge_len)  # [E, weight_numel]
        msg = self.tp(h_j, sh, w)  # [E, dim_A]
        A = scatter(msg, edge_index[0], dim=0, dim_size=h.shape[0], reduce="sum")
        A = A / math.sqrt(avg_num_neighbors)  # neighbor normalization
        return self.msg(A) + self.skip(h, z)
