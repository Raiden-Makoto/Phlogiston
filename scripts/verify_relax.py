"""Smoke test for phlogiston.verify.relax (DESIGN.md §10, step 2).

Two checks:
  1. Synthetic: perturb a known solid (rattle + strain), relax, and confirm the
     drift metrics + energy drop behave (energy decreases, rmsd/|dV| > 0).
  2. Real (optional): relax the candidates in a registry CIF dir if given, to
     see the drift on actual generator output.

Run inside the ROCm container, e.g.:
    python scripts/verify_relax.py --backend chgnet
    python scripts/verify_relax.py --backend chgnet --cif-dir data/runs/candidates/cifs
"""

from __future__ import annotations

import argparse
import glob
import os

from pymatgen.core import Structure

from phlogiston.verify import load_calculator, relax_structure, resolve_device


def _report(name: str, r) -> None:
    if r is None:
        print(f"{name:>28}: FAILED to relax")
        return
    flag = "OK" if (r.de <= 1e-6 and r.converged) else "CHECK"
    print(f"{name:>28}: E {r.initial_energy_per_atom:+.4f} -> {r.energy_per_atom:+.4f} "
          f"(de={r.de:+.4f}) eV/atom  rmsd={r.rmsd:.3f} A  |dV|={r.dvol_frac:.3f}  "
          f"steps={r.n_steps}  fmax={r.max_force:.3f}  conv={r.converged}  {flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="chgnet")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cif-dir", default=None, help="Relax real candidate CIFs from here")
    ap.add_argument("--max-cifs", type=int, default=10)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"backend={args.backend}  device={dev}\n")
    calc = load_calculator(args.backend, device=dev)

    # 1. synthetic: rattle + isotropic strain a known solid, then relax.
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor

    ok = True
    for elem, cryst, a in [("Si", "diamond", 5.43), ("Cu", "fcc", 3.61)]:
        atoms = bulk(elem, cryst, a=a)
        atoms.rattle(stdev=0.15, seed=1)
        atoms.set_cell(atoms.cell * 1.05, scale_atoms=True)  # 5% expansion
        s = AseAtomsAdaptor().get_structure(atoms)
        r = relax_structure(s, calc)
        _report(f"{elem} (rattled+strained)", r)
        ok = ok and r is not None and r.de <= 1e-6 and r.converged and r.rmsd > 0

    # 2. optional: real registry candidates.
    if args.cif_dir:
        cifs = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))[: args.max_cifs]
        print(f"\nRelaxing {len(cifs)} registry CIFs from {args.cif_dir}:")
        for path in cifs:
            s = Structure.from_file(path)
            try:
                r = relax_structure(s, calc)
            except Exception as e:  # noqa: BLE001
                print(f"{os.path.basename(path):>28}: ERROR {e!r}")
                continue
            _report(os.path.basename(path), r)

    print("\n" + ("PASS: relaxation + drift metrics work"
                  if ok else "FAIL: see CHECK rows"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
