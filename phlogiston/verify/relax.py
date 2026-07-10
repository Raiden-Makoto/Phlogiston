"""uMLIP relaxation + drift metrics (DESIGN.md §4, stage 2a).

The CDVAE emits *approximate* geometry, so the physically meaningful structure
is the one at the bottom of the energy well. We relax each candidate (cell +
atomic positions) with the primary uMLIP and treat the **relaxed** structure as
canonical -- it is lower-energy and real, so it replaces the generated one
downstream. The generated structure is never mutated in place; callers keep it
for provenance.

Alongside the relaxed structure we record **drift diagnostics** -- how far the
generator's guess had to move to reach the minimum. Large drift is a red flag
that the generator produced an off-manifold cell:

  * ``rmsd``      -- per-atom Cartesian displacement RMSD (Angstrom, min-image).
  * ``dvol_frac`` -- |V_relaxed - V_init| / V_init.
  * ``de``        -- energy-per-atom change (relaxed - initial); expected <= 0.

Backends come from ``potential.py`` (CHGNet primary, MatterSim cross-check); this
module is calculator-agnostic and just drives the ASE relaxation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


@dataclass
class RelaxResult:
    """Outcome of relaxing one structure with a uMLIP."""

    structure: "Structure"  # relaxed (canonical); pymatgen
    energy: float  # total potential energy of the relaxed cell (eV)
    energy_per_atom: float  # eV/atom (relaxed)
    initial_energy_per_atom: float  # eV/atom (as-generated)
    n_steps: int
    converged: bool
    max_force: float  # eV/Angstrom on atoms after relaxation
    rmsd: float  # Angstrom, per-atom displacement (min-image)
    dvol_frac: float  # fractional volume change magnitude
    de: float  # energy_per_atom - initial_energy_per_atom (<= 0 expected)


def _drift(initial: "Structure", relaxed: "Structure") -> tuple[float, float]:
    """Per-atom displacement RMSD (Angstrom, min-image) and fractional |dV|.

    Atom order/identity is preserved by relaxation, so we compare like-for-like.
    Displacement is measured in fractional space (min-image wrapped) then mapped
    to Cartesian with the relaxed lattice -- a diagnostic, not an exact RMSD.
    """
    fi = np.asarray(initial.frac_coords)
    ff = np.asarray(relaxed.frac_coords)
    dfrac = ff - fi
    dfrac -= np.round(dfrac)  # min-image to [-0.5, 0.5)
    cart = dfrac @ relaxed.lattice.matrix
    rmsd = float(np.sqrt(np.mean(np.sum(cart**2, axis=1))))
    vi, vf = initial.volume, relaxed.volume
    dvol_frac = float(abs(vf - vi) / vi) if vi > 0 else float("nan")
    return rmsd, dvol_frac


def relax_structure(
    structure: "Structure",
    calc: "Calculator",
    *,
    fmax: float = 0.05,
    steps: int = 500,
    relax_cell: bool = True,
    optimizer: str = "fire",
) -> RelaxResult:
    """Relax ``structure`` with ASE + ``calc`` and return the relaxed structure
    plus drift diagnostics. Does not mutate the input structure.

    Parameters
    ----------
    fmax : force convergence threshold (eV/Angstrom).
    steps : max optimizer steps.
    relax_cell : also relax the lattice (FrechetCellFilter) vs positions only.
    optimizer : ``fire`` (robust default) or ``lbfgs``.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE, LBFGS
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    atoms.calc = calc

    n = len(atoms)
    e0 = atoms.get_potential_energy()
    e0_per_atom = e0 / n

    target = FrechetCellFilter(atoms) if relax_cell else atoms
    opt_cls = {"fire": FIRE, "lbfgs": LBFGS}[optimizer.lower()]
    opt = opt_cls(target, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    # Forces on the atoms themselves (not the cell filter's augmented array).
    forces = atoms.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max()) if n else 0.0
    n_steps = int(opt.get_number_of_steps())
    converged = max_force <= fmax

    e1 = atoms.get_potential_energy()
    e1_per_atom = e1 / n
    relaxed = adaptor.get_structure(atoms)
    rmsd, dvol_frac = _drift(structure, relaxed)

    return RelaxResult(
        structure=relaxed,
        energy=float(e1),
        energy_per_atom=float(e1_per_atom),
        initial_energy_per_atom=float(e0_per_atom),
        n_steps=n_steps,
        converged=converged,
        max_force=max_force,
        rmsd=rmsd,
        dvol_frac=dvol_frac,
        de=float(e1_per_atom - e0_per_atom),
    )


def relax_structures(
    structures: list["Structure"],
    calc: "Calculator",
    **kwargs,
) -> list[RelaxResult]:
    """Relax many structures reusing one loaded ``calc`` (avoids reloading the
    model per candidate). A structure that fails to relax yields ``None`` in the
    corresponding slot rather than aborting the batch."""
    results: list[RelaxResult] = []
    for s in structures:
        try:
            results.append(relax_structure(s, calc, **kwargs))
        except Exception:  # noqa: BLE001 -- one bad cell shouldn't kill the batch
            results.append(None)  # type: ignore[arg-type]
    return results
