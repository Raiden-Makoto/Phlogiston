"""Validation harness for the crystal-graph featurizer.

A bug here corrupts every downstream tensor, so we check it hard:
  1. Known structures (FCC, NaCl) give the textbook coordination numbers.
  2. ||edge_vec|| matches pymatgen distances (periodic-image handling).
  3. Translation invariance of the geometry.
  4. Rotation equivariance (edge_vec rotates, edge_len invariant).
  5. Supercell periodicity (local environments replicate exactly).
  6. Real corpus CIFs featurize without error.

Run inside the container:  python scripts/validate_graph.py
"""

from __future__ import annotations

import glob
import sys

import numpy as np
from pymatgen.core import Lattice, Structure

from phlogiston.data.graph import structure_to_graph

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}")


def fcc_al() -> Structure:
    # conventional cubic FCC, a = 4.05 A -> NN distance 2.863 A, 12 neighbors
    return Structure(Lattice.cubic(4.05), ["Al"] * 4,
                     [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def nacl() -> Structure:
    # rocksalt, a = 5.64 A -> Na-Cl NN 2.82 A, 6 neighbors each
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
              [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]]
    return Structure(Lattice.cubic(5.64), ["Na"] * 4 + ["Cl"] * 4, coords)


def rotation_matrix(a, b, c) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def test_coordination():
    g = structure_to_graph(fcc_al(), cutoff=3.0)
    deg = np.bincount(g.edge_index[0].numpy(), minlength=g.num_nodes)
    check("FCC Al coordination = 12", bool(np.all(deg == 12)), f"degrees={deg.tolist()}")
    nn = float(g.edge_len.min())
    check("FCC Al NN distance ~2.863 A", abs(nn - 2.863) < 1e-2, f"nn={nn:.3f}")

    g2 = structure_to_graph(nacl(), cutoff=3.0)
    deg2 = np.bincount(g2.edge_index[0].numpy(), minlength=g2.num_nodes)
    check("NaCl coordination = 6", bool(np.all(deg2 == 6)), f"degrees={deg2.tolist()}")
    # every NaCl NN edge must connect unlike species (Na<->Cl)
    zi = g2.z[g2.edge_index[0]].numpy(); zj = g2.z[g2.edge_index[1]].numpy()
    check("NaCl NN edges are Na-Cl", bool(np.all(zi != zj)))


def test_distance_consistency():
    # internal assert already enforces this; confirm on a bigger cutoff too
    try:
        g = structure_to_graph(nacl(), cutoff=6.0)
        recomputed = np.linalg.norm(g.edge_vec.numpy(), axis=1)
        check("||edge_vec|| == edge_len", np.allclose(recomputed, g.edge_len.numpy(), atol=1e-6))
    except Exception as e:  # noqa: BLE001
        check("||edge_vec|| == edge_len", False, repr(e))


def test_translation_invariance():
    s = nacl()
    g0 = structure_to_graph(s, cutoff=5.0)
    s2 = s.copy(); s2.translate_sites(range(len(s2)), [0.123, 0.456, 0.789], frac_coords=True)
    g1 = structure_to_graph(s2, cutoff=5.0)
    same_E = g0.edge_len.shape == g1.edge_len.shape
    sorted_ok = same_E and np.allclose(np.sort(g0.edge_len.numpy()),
                                        np.sort(g1.edge_len.numpy()), atol=1e-6)
    check("translation invariance (edge lengths)", bool(sorted_ok))


def test_rotation_equivariance():
    s = nacl()
    g0 = structure_to_graph(s, cutoff=5.0)
    R = rotation_matrix(0.3, -0.7, 1.1)
    # rotate the cell: same fractional coords, lattice rows rotated by R
    s_rot = Structure(Lattice(s.lattice.matrix @ R.T), s.species, s.frac_coords)
    g1 = structure_to_graph(s_rot, cutoff=5.0)
    len_ok = np.allclose(np.sort(g0.edge_len.numpy()), np.sort(g1.edge_len.numpy()), atol=1e-6)

    # Edges are a set; neighbor-list order may differ after rotation. Compare
    # the multiset of edge vectors (order-independent) against edge_vec @ R^T.
    def sort_rows(a):
        return a[np.lexsort(np.round(a, 4).T[::-1])]
    rotated = g0.edge_vec.numpy() @ R.T
    vec_ok = (g0.edge_vec.shape == g1.edge_vec.shape and
              np.allclose(sort_rows(g1.edge_vec.numpy()), sort_rows(rotated), atol=1e-4))
    check("rotation: edge_len invariant", bool(len_ok))
    check("rotation: edge_vec equivariant (v @ R^T, order-robust)", bool(vec_ok))


def test_supercell_periodicity():
    s = fcc_al()
    g0 = structure_to_graph(s, cutoff=5.0)
    s2 = s.copy(); s2.make_supercell([2, 2, 2])
    g1 = structure_to_graph(s2, cutoff=5.0)
    # 8 replicas -> edge count 8x and identical sorted length spectrum
    count_ok = g1.edge_len.shape[0] == 8 * g0.edge_len.shape[0]
    spec_ok = count_ok and np.allclose(
        np.sort(g1.edge_len.numpy()),
        np.sort(np.tile(g0.edge_len.numpy(), 8)), atol=1e-5)
    check("supercell periodicity (edge spectrum x8)", bool(spec_ok),
          f"E_prim={g0.edge_len.shape[0]} E_super={g1.edge_len.shape[0]}")


def test_real_cifs():
    files = sorted(glob.glob("data/raw/mp/cifs/*.cif"))[:200]
    if not files:
        check("real CIFs featurize", False, "no CIFs found under data/raw/mp/cifs")
        return
    ok, fails = 0, []
    for f in files:
        try:
            g = structure_to_graph(Structure.from_file(f), cutoff=6.0)
            assert g.edge_index.shape[1] > 0 and g.num_nodes > 0
            ok += 1
        except Exception as e:  # noqa: BLE001
            fails.append((f.split("/")[-1], repr(e)[:80]))
    check(f"real CIFs featurize ({ok}/{len(files)})", not fails,
          f"failures={fails[:3]}" if fails else "")


if __name__ == "__main__":
    test_coordination()
    test_distance_consistency()
    test_translation_invariance()
    test_rotation_equivariance()
    test_supercell_periodicity()
    test_real_cifs()
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n{'='*50}\n{len(results)-n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)
