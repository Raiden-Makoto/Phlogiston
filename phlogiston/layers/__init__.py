"""Reusable equivariant / neural building blocks.

Layout: implementations in ``src/``, specs in ``docs/`` (see ``README.md``).
Public classes are re-exported here, so import as e.g.
``from phlogiston.layers import SphericalHarmonics``.
"""

from phlogiston.layers.src.spherical import SphericalHarmonics
from phlogiston.layers.src.radial import BesselBasis, PolynomialCutoff, RadialBasis
from phlogiston.layers.src.embedding import AtomEmbedding
from phlogiston.layers.src.linear import EquivariantLinear, SpeciesLinear
from phlogiston.layers.src.gate import EquivariantGate
from phlogiston.layers.src.readout import ScalarReadout
from phlogiston.layers.src.interaction import Interaction

__all__ = [
    "SphericalHarmonics",
    "BesselBasis",
    "PolynomialCutoff",
    "RadialBasis",
    "AtomEmbedding",
    "EquivariantLinear",
    "SpeciesLinear",
    "EquivariantGate",
    "ScalarReadout",
    "Interaction",
]
