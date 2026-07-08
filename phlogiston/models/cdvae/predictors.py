"""CDVAE latent predictors (see DESIGN.md §3).

From the latent ``z`` predict the global structure descriptors that condition
generation: number of atoms, lattice, and composition. All are invariant
functions of the invariant latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))


@dataclass
class LatentPrediction:
    num_atoms_logits: torch.Tensor  # [B, n_max]   (class i -> i+1 atoms)
    lattice: torch.Tensor  # [B, 6]       (3 lengths + 3 angles, normalized)
    composition_logits: torch.Tensor  # [B, n_elements] (per-element propensity)


class LatentPredictors(nn.Module):
    def __init__(
        self, latent_dim: int = 256, n_max: int = 64, n_elements: int = 100, hidden: int = 256
    ):
        super().__init__()
        self.n_max = n_max
        self.n_elements = n_elements
        self.num_atoms = _mlp(latent_dim, hidden, n_max)
        self.lattice = _mlp(latent_dim, hidden, 6)
        self.composition = _mlp(latent_dim, hidden, n_elements)

    def forward(self, z: torch.Tensor) -> LatentPrediction:
        return LatentPrediction(
            num_atoms_logits=self.num_atoms(z),
            lattice=self.lattice(z),
            composition_logits=self.composition(z),
        )
