"""Smoke test for phlogiston.verify.potential (DESIGN.md §10, step 1).

Loads a uMLIP backend through our adapter, scores a few known solids on the
GPU, then rattles and relaxes each to confirm the full ASE relaxation path
works and energy drops back toward the crystalline minimum.

Run inside the ROCm container, e.g.:
    python scripts/verify_potential.py --backend chgnet
"""

from __future__ import annotations

import argparse

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from phlogiston.verify import load_calculator, resolve_device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="chgnet")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    import torch

    dev = resolve_device(args.device)
    print(f"backend={args.backend}  device={dev}  torch={torch.__version__}  "
          f"gpus={torch.cuda.device_count()}")

    calc = load_calculator(args.backend, device=dev)
    print(f"loaded calculator: {type(calc).__module__}.{type(calc).__name__}\n")

    solids = [
        ("Si", bulk("Si", "diamond", a=5.43)),
        ("NaCl", bulk("NaCl", "rocksalt", a=5.64)),
        ("Cu", bulk("Cu", "fcc", a=3.61)),
    ]

    ok = True
    for name, atoms in solids:
        atoms.calc = calc
        e0 = atoms.get_potential_energy() / len(atoms)

        rattled = atoms.copy()
        rattled.calc = calc
        rattled.rattle(stdev=0.1, seed=0)
        e_rattled = rattled.get_potential_energy() / len(rattled)

        opt = FIRE(FrechetCellFilter(rattled), logfile=None)
        opt.run(fmax=args.fmax, steps=args.steps)
        e_relaxed = rattled.get_potential_energy() / len(rattled)
        nsteps = opt.get_number_of_steps()

        # A sane potential: rattling raises energy, relaxation recovers most of it.
        recovered = e_relaxed <= e_rattled + 1e-6
        near_ideal = e_relaxed <= e0 + 0.02  # eV/atom
        ok = ok and recovered and near_ideal
        print(f"{name:>5}: E0={e0:+.4f}  rattled={e_rattled:+.4f}  "
              f"relaxed={e_relaxed:+.4f} eV/atom  ({nsteps} steps)  "
              f"{'OK' if recovered and near_ideal else 'CHECK'}")

    print("\n" + ("PASS: adapter + ASE relaxation path work on this device"
                  if ok else "FAIL: see CHECK rows above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
