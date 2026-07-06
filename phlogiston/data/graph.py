"""Crystal-graph featurization (Phase 3b).

Turns a periodic ``pymatgen`` Structure into a *lossless* graph-geometry record:
atomic numbers + a periodic neighbor graph with the exact Cartesian
displacement vectors. Everything a model needs geometrically is preserved;
learned features (element embeddings), radial basis expansion, and spherical
harmonics are computed *in the model* at train time, not baked in here. That
keeps the on-disk artifact minimal and means feature changes never require
re-preprocessing the corpus.

Conventions
-----------
* Edges are directed: ``edge_index[0]`` is the center/receiver atom ``i`` and
  ``edge_index[1]`` is the neighbor/sender atom ``j``.
* ``edge_vec[e] = r_j(image) - r_i`` in Angstrom (points from i to j), correctly
  accounting for periodic images. ``edge_len[e] = ||edge_vec[e]||``.
* A radius cutoff defines neighbors (standard for equivariant potentials);
  periodic images are included, and an atom may bond to its own images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CrystalGraph:
    z: torch.Tensor            # [N] int64 atomic numbers
    pos: torch.Tensor          # [N, 3] float Cartesian coords (Angstrom)
    lattice: torch.Tensor      # [3, 3] float lattice row-vectors (Angstrom)
    edge_index: torch.Tensor   # [2, E] int64  (row 0 = center i, row 1 = neighbor j)
    edge_vec: torch.Tensor     # [E, 3] float  r_j(image) - r_i  (Angstrom)
    edge_len: torch.Tensor     # [E] float
    num_nodes: int

    def __repr__(self) -> str:  # concise
        return (f"CrystalGraph(N={self.num_nodes}, E={self.edge_index.shape[1]}, "
                f"z={sorted(set(self.z.tolist()))})")


def structure_to_graph(
    structure,
    cutoff: float = 6.0,
    dtype: torch.dtype = torch.float32,
    numerical_tol: float = 1e-6,
) -> CrystalGraph:
    """Build a :class:`CrystalGraph` from a pymatgen ``Structure``.

    ``cutoff`` defaults to 6.0 A (the MACE-MP r_max); smaller values can isolate
    atoms in wide-spaced lattices (e.g. bcc Cs, NN ~5.24 A). Raises on
    disordered structures or atoms left with zero neighbors (which would
    silently drop information downstream).
    """
    if not structure.is_ordered:
        raise ValueError("structure_to_graph requires an ordered structure "
                         "(no partial occupancies).")

    n = len(structure)
    z = np.array([site.specie.Z for site in structure], dtype=np.int64)
    cart = np.asarray(structure.cart_coords, dtype=np.float64)      # [N, 3]
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float64)  # [3, 3]

    # Fast periodic neighbor list. images are integer lattice translations of j.
    center_idx, point_idx, images, dists = structure.get_neighbor_list(
        r=cutoff, numerical_tol=numerical_tol
    )
    if len(center_idx) == 0:
        raise ValueError(f"No neighbors within cutoff={cutoff} A; increase cutoff.")

    # Displacement vector i -> j(image), in Cartesian Angstrom.
    offset_cart = images @ lattice                       # [E, 3]
    edge_vec = cart[point_idx] + offset_cart - cart[center_idx]
    edge_len = np.linalg.norm(edge_vec, axis=1)

    # --- correctness guard: our vector norm must match pymatgen's distances ---
    max_err = float(np.max(np.abs(edge_len - dists))) if len(dists) else 0.0
    if max_err > 1e-4:
        raise AssertionError(
            f"edge length mismatch vs pymatgen (max {max_err:.2e} A) -- "
            "periodic-image handling is wrong."
        )

    # every atom must have at least one neighbor
    covered = np.unique(center_idx)
    if covered.size != n:
        missing = sorted(set(range(n)) - set(covered.tolist()))
        raise ValueError(f"atoms {missing} have no neighbors within {cutoff} A; "
                         "increase cutoff.")

    return CrystalGraph(
        z=torch.from_numpy(z),
        pos=torch.from_numpy(cart).to(dtype),
        lattice=torch.from_numpy(lattice).to(dtype),
        edge_index=torch.from_numpy(np.stack([center_idx, point_idx])).long(),
        edge_vec=torch.from_numpy(edge_vec).to(dtype),
        edge_len=torch.from_numpy(edge_len).to(dtype),
        num_nodes=n,
    )


def graph_from_cif(path: str, cutoff: float = 6.0, **kw) -> CrystalGraph:
    """Convenience: read a CIF and featurize it."""
    from pymatgen.core import Structure

    return structure_to_graph(Structure.from_file(path), cutoff=cutoff, **kw)
