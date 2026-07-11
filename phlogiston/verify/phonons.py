"""Stage 2d: dynamical stability via finite-displacement phonons (DESIGN.md §4).

A structure at a local energy *minimum* (thermodynamically near-hull) can still be
a saddle point on the potential energy surface -- it would spontaneously distort.
Phonons are the test: we displace atoms in a supercell, evaluate forces with the
primary uMLIP, build force constants (phonopy), and inspect the phonon spectrum.
**Imaginary (negative) frequencies = dynamically unstable.**

Only run on near-hull survivors (it's the expensive-ish step). A small tolerance
absorbs the numerical near-zero acoustic modes at Gamma so we don't fail genuinely
stable crystals on noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator
    from pymatgen.core import Structure


@dataclass
class PhononResult:
    """Dynamical-stability verdict from a finite-displacement phonon calc."""

    min_freq_thz: float  # most-negative phonon frequency over the mesh (THz)
    dynamically_stable: bool  # min_freq >= -tol_thz
    n_displacements: int  # supercells evaluated with the uMLIP
    supercell: tuple[int, int, int]


def _supercell_matrix(structure: "Structure", min_len: float, max_mult: int) -> list[int]:
    """Diagonal supercell repeats so each axis reaches ``min_len`` (Angstrom),
    capped at ``max_mult`` to bound cost."""
    abc = structure.lattice.abc
    return [int(min(max_mult, max(1, np.ceil(min_len / a)))) for a in abc]


def phonon_stability(
    structure: "Structure",
    calc: "Calculator",
    *,
    supercell_min_len: float = 8.0,
    max_multiplier: int = 2,
    displacement: float = 0.03,
    mesh: int = 8,
    tol_thz: float = 0.1,
) -> PhononResult:
    """Finite-displacement phonons for ``structure`` with ``calc`` forces.

    Returns the minimum phonon frequency over a ``mesh``^3 q-grid and a
    stability flag (``min_freq >= -tol_thz``). ``displacement`` is the atomic
    displacement (Angstrom) for the force calculations.
    """
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(
        symbols=[str(s.specie) for s in structure],
        scaled_positions=structure.frac_coords,
        cell=structure.lattice.matrix,
    )
    sc = _supercell_matrix(structure, supercell_min_len, max_multiplier)
    phonon = Phonopy(unitcell, supercell_matrix=[[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]])
    phonon.generate_displacements(distance=displacement)

    forces = []
    for cell in phonon.supercells_with_displacements:
        atoms = Atoms(
            symbols=cell.symbols,
            scaled_positions=cell.scaled_positions,
            cell=cell.cell,
            pbc=True,
        )
        atoms.calc = calc
        forces.append(atoms.get_forces())

    phonon.forces = np.array(forces)
    phonon.produce_force_constants()
    phonon.run_mesh([mesh, mesh, mesh])
    freqs = phonon.get_mesh_dict()["frequencies"]  # [n_qpoints, n_bands], THz
    min_freq = float(np.min(freqs))
    return PhononResult(
        min_freq_thz=min_freq,
        dynamically_stable=bool(min_freq >= -abs(tol_thz)),
        n_displacements=len(forces),
        supercell=(sc[0], sc[1], sc[2]),
    )
