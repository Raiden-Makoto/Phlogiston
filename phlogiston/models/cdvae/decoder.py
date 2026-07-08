"""CDVAE score decoder / denoiser (see DESIGN.md §4).

A noise-conditioned equivariant network over a *noisy* crystal graph. Like the
encoder, but: (1) node scalars are conditioned on the latent ``z`` and the noise
level ``sigma``; (2) it keeps equivariant (ℓ>0) features to the last layer; and
(3) it has two heads — an equivariant **coord score** (`1o`, [N,3]) and invariant
per-atom **type logits** ([N, n_elements]).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from e3nn import o3
from torch import nn

from phlogiston.layers import (
    AtomEmbedding,
    EquivariantGate,
    EquivariantVectorReadout,
    Interaction,
    NoiseEmbedding,
    ScalarReadout,
    SphericalHarmonics,
)


@dataclass
class ScoreOutput:
    coord_score: torch.Tensor  # [N, 3]  equivariant (1o)
    type_logits: torch.Tensor  # [N, n_elements] invariant


class CDVAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        n_elements: int = 100,
        mul: int = 128,
        l_feat: int = 2,
        l_sh: int = 3,
        n_layers: int = 3,
        n_bessel: int = 8,
        r_max: float = 6.0,
        num_species: int = 119,
        avg_num_neighbors: float = 50.0,
        correlation: int = 3,
    ):
        super().__init__()
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.sh = SphericalHarmonics(l_max=l_sh)
        self.embedding = AtomEmbedding(mul=mul, z_max=num_species - 1)
        # conditioning injected into node scalars (all mul-dim, additive)
        self.z_proj = nn.Linear(latent_dim, mul)
        self.noise_emb = NoiseEmbedding(dim=mul)
        hidden = o3.Irreps(f"{mul}x0e + {mul}x1o + {mul}x2e")

        self.interactions = nn.ModuleList()
        self.gates = nn.ModuleList()
        irreps_in = self.embedding.irreps_out  # mul x 0e
        for _ in range(n_layers):
            gate = EquivariantGate(hidden)
            inter = Interaction(
                irreps_in,
                self.sh.irreps_out,
                gate.irreps_in,
                l_feat,
                r_max,
                n_bessel,
                num_species,
                correlation=correlation,
            )
            self.interactions.append(inter)
            self.gates.append(gate)
            irreps_in = gate.irreps_out  # keep full hidden (incl. 1o) to the end

        self.coord_head = EquivariantVectorReadout(irreps_in, n_vectors=1)
        self.type_head = ScalarReadout(irreps_in, n_out=n_elements)  # per-atom (no pooling)

    def forward(self, graph, z: torch.Tensor, sigma: torch.Tensor) -> ScoreOutput:
        sh = self.sh(graph.edge_vec)
        # condition: species embedding + per-graph latent + noise level (broadcast to nodes)
        h = self.embedding(graph.z)
        h = h + self.z_proj(z)[graph.batch] + self.noise_emb(sigma)[graph.batch]
        for inter, gate in zip(self.interactions, self.gates, strict=False):
            h = inter(h, graph.edge_index, graph.edge_len, sh, graph.z, self.avg_num_neighbors)
            h = gate(h)
        return ScoreOutput(coord_score=self.coord_head(h), type_logits=self.type_head(h))
