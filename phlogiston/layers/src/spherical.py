"""Real spherical harmonics of edge directions.

See ``spherical.md``. This is a thin, equivariant wrapper over the e3nn
primitive ``o3.spherical_harmonics`` (which carries the hard SH math); it emits
the angular signal `Y_ℓ^m(r̂)` for `ℓ = 0..L_sh` that the interaction block
couples with neighbor features.
"""

from __future__ import annotations

import torch
from e3nn import o3


class SphericalHarmonics(torch.nn.Module):
    """Map edge vectors ``[E, 3]`` -> spherical-harmonic features ``[E, dim]``.

    Parameters
    ----------
    l_max: highest harmonic degree (default 3).
    normalize: use the unit direction (length is handled by the radial layer).
    normalization: e3nn normalization convention (``component`` matches the
        tensor products in the interaction layer).
    """

    def __init__(self, l_max: int = 3, normalize: bool = True, normalization: str = "component"):
        super().__init__()
        self.l_max = int(l_max)
        # irreps 1x0e + 1x1o + 1x2e + ... with the correct parities.
        self.irreps_out = o3.Irreps.spherical_harmonics(self.l_max)
        self.normalize = normalize
        self.normalization = normalization

    def forward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            # o3.spherical_harmonics(normalize=True) divides by the vector norm
            # internally; a near-zero-length edge (e.g. from training-time
            # coordinate noise nearly cancelling out a short edge) gives a
            # near-0/0 division -> nan/inf that poisons the whole batch. Floor
            # the norm and rescale so direction stays defined (arbitrarily, for
            # this physically-meaningless case) instead of blowing up.
            norm = edge_vec.norm(dim=-1, keepdim=True)
            edge_vec = edge_vec * (norm.clamp(min=1e-2) / norm.clamp(min=1e-12))
        return o3.spherical_harmonics(
            self.irreps_out,
            edge_vec,
            normalize=self.normalize,
            normalization=self.normalization,
        )

    def __repr__(self) -> str:
        return f"SphericalHarmonics(l_max={self.l_max}, irreps_out={self.irreps_out})"
