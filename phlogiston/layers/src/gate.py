"""Gated equivariant nonlinearity.

See ``docs/gate.md``. Given the desired hidden irreps, builds an `e3nn.nn.Gate`
that activates scalars (SiLU) and gates each higher-ℓ irrep by a sigmoid-
activated invariant scalar. Exposes `irreps_in` (what the preceding linear must
produce: scalars + gates + gated) and `irreps_out` (scalars + gated).
"""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn


class EquivariantGate(nn.Module):
    def __init__(self, irreps_hidden):
        super().__init__()
        irreps_hidden = o3.Irreps(irreps_hidden)
        scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l == 0])
        gated = o3.Irreps([(mul, ir) for mul, ir in irreps_hidden if ir.l > 0])
        n_gates = gated.num_irreps  # one gate scalar per gated irrep
        gates = o3.Irreps(f"{n_gates}x0e") if n_gates > 0 else o3.Irreps("")

        self.gate = Gate(
            irreps_scalars=scalars,
            act_scalars=[torch.nn.functional.silu],
            irreps_gates=gates,
            act_gates=[torch.sigmoid] if n_gates > 0 else [],
            irreps_gated=gated,
        )
        self.irreps_in = self.gate.irreps_in  # scalars + gates + gated
        self.irreps_out = self.gate.irreps_out  # scalars + gated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)
