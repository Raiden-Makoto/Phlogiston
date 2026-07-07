"""Scalar readout + graph pooling.

See ``docs/readout.md``. Reads only the invariant (`0e`) part of node features
through an MLP to per-atom scalars, then optionally pools to the graph level.
Output is invariant by construction (only scalars are read).
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import nn


class ScalarReadout(nn.Module):
    def __init__(self, irreps_in, n_out: int = 1, hidden=(), reduce: str = "sum"):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        # column slices of the 0e (scalar) channels
        self._scalar_slices = [
            sl
            for (mul, ir), sl in zip(self.irreps_in, self.irreps_in.slices(), strict=False)
            if ir.l == 0 and ir.p == 1
        ]
        scalar_dim = sum(sl.stop - sl.start for sl in self._scalar_slices)
        assert scalar_dim > 0, "readout needs at least one 0e irrep in the input"
        self.reduce = reduce

        dims = [scalar_dim, *hidden, n_out]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def _scalars(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, sl] for sl in self._scalar_slices], dim=-1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        r = self.mlp(self._scalars(x))  # [N, n_out] per-atom
        if batch is None:
            return r
        from torch_geometric.utils import scatter  # pure-torch, ROCm-safe

        n_graphs = int(batch.max()) + 1
        return scatter(r, batch, dim=0, dim_size=n_graphs, reduce=self.reduce)
