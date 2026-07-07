"""CrystalEncoder — the shared E(3)-equivariant encoder (see DESIGN.md).

Assembles the layer blocks: Spherical (once) + Embedding, then `n_layers`
interaction blocks (each with a Gate, except the last which collapses to
scalars). Returns invariant per-atom features and a mean-pooled graph feature
that downstream heads consume.

Uses interaction **v1** (ν=1, A-basis); the v2 symmetric contraction drops in
behind the same `Interaction` interface with no change here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from e3nn import o3

from phlogiston.layers import (
    AtomEmbedding, EquivariantGate, Interaction, SphericalHarmonics,
)


@dataclass
class EncoderOutput:
    node_feats: torch.Tensor    # [N, mul]  invariant per-atom features
    graph_feats: torch.Tensor   # [B, mul]  mean-pooled per-graph features


class CrystalEncoder(nn.Module):
    def __init__(self, mul: int = 128, l_feat: int = 2, l_sh: int = 3,
                 n_layers: int = 2, n_bessel: int = 8, r_max: float = 6.0,
                 num_species: int = 119, avg_num_neighbors: float = 50.0):
        super().__init__()
        self.mul = mul
        # Fixed dataset constant (NOT computed per batch) so a graph's features
        # never depend on what else is in the batch. ~50 for our corpus; set
        # precisely from data stats at train time.
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.sh = SphericalHarmonics(l_max=l_sh)
        self.embedding = AtomEmbedding(mul=mul, z_max=num_species - 1)
        hidden = o3.Irreps(f"{mul}x0e + {mul}x1o + {mul}x2e")

        self.interactions = nn.ModuleList()
        self.gates = nn.ModuleList()
        irreps_in = self.embedding.irreps_out               # mul x 0e
        for t in range(n_layers):
            last = t == n_layers - 1
            if last:
                # final block: scalars only (readout/heads use invariants)
                inter = Interaction(irreps_in, self.sh.irreps_out,
                                    o3.Irreps(f"{mul}x0e"), l_feat, r_max,
                                    n_bessel, num_species)
                self.interactions.append(inter)
                self.gates.append(nn.Identity())
                irreps_in = inter.irreps_out
            else:
                gate = EquivariantGate(hidden)
                inter = Interaction(irreps_in, self.sh.irreps_out,
                                    gate.irreps_in, l_feat, r_max, n_bessel,
                                    num_species)
                self.interactions.append(inter)
                self.gates.append(gate)
                irreps_in = gate.irreps_out                 # = hidden
        self.irreps_out = irreps_in                          # mul x 0e

    def forward(self, graph) -> EncoderOutput:
        from torch_geometric.utils import scatter

        sh = self.sh(graph.edge_vec)
        h = self.embedding(graph.z)
        for inter, gate in zip(self.interactions, self.gates):
            h = inter(h, graph.edge_index, graph.edge_len, sh, graph.z,
                      self.avg_num_neighbors)
            h = gate(h)

        n_graphs = int(graph.batch.max()) + 1
        graph_feats = scatter(h, graph.batch, dim=0, dim_size=n_graphs, reduce="mean")
        return EncoderOutput(node_feats=h, graph_feats=graph_feats)
