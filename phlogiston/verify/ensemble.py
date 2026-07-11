"""Stage 2c: ensemble cross-check (DESIGN.md §1, §4).

Any single uMLIP is least reliable off-distribution -- exactly where our more
exotic candidates live. So a second, independently-trained potential (MatterSim)
re-relaxes the candidate and places it on *its own* self-consistent hull; the
**disagreement** between the two members' hull distances is the confidence
signal. Concord => trustworthy verdict; divergence => flag the candidate
low-confidence / off-distribution for scrutiny. No external ground truth needed.

We compare hull distances (each in its member's self-consistent frame) rather
than raw energies, since absolute uMLIP energies carry per-model offsets that a
hull distance cancels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from phlogiston.verify.hull import CompetitorCache, build_competitor_entries, refined_hull_distance
from phlogiston.verify.relax import relax_structure

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


@dataclass
class EnsembleResult:
    """Cross-check of a candidate against a second uMLIP."""

    backend: str  # secondary member name
    e_above_hull_secondary: float  # eV/atom, secondary member's own-frame hull
    spread: float  # |e_hull_primary - e_hull_secondary| (eV/atom)
    confidence: str  # "high" (members concur) | "low" (off-distribution)


def ensemble_cross_check(
    relaxed: "Structure",
    primary_e_above_hull: float,
    calc_secondary: "Calculator",
    cache_secondary: CompetitorCache,
    backend_secondary: str,
    *,
    spread_max: float = 0.05,
    relax_steps: int = 300,
    competitor_relax_steps: int = 150,
    ehull_cutoff: float = 0.05,
    api_key: str | None = None,
    verbose: bool = True,
) -> EnsembleResult:
    """Re-relax the (primary-relaxed) candidate with the secondary potential,
    place it on the secondary's self-consistent hull, and score the disagreement.

    ``spread <= spread_max`` => ``high`` confidence; otherwise ``low``. Competitor
    energies are cached per-backend so the secondary hull amortizes across a batch.
    """
    rr = relax_structure(relaxed, calc_secondary, steps=relax_steps)
    elements = sorted({str(el) for el in rr.structure.composition.elements})
    competitors = build_competitor_entries(
        elements, calc_secondary, cache_secondary,
        ehull_cutoff=ehull_cutoff, relax_steps=competitor_relax_steps,
        api_key=api_key, verbose=verbose,
    )
    hull = refined_hull_distance(rr.structure, rr.energy, competitors)
    spread = abs(primary_e_above_hull - hull.e_above_hull_umlip)
    return EnsembleResult(
        backend=backend_secondary,
        e_above_hull_secondary=hull.e_above_hull_umlip,
        spread=float(spread),
        confidence="high" if spread <= spread_max else "low",
    )
