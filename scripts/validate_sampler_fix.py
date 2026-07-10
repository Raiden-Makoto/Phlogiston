"""Validate the batched-sampler coordinate fix end-to-end.

Generates a small batch from the CDVAE with the (now corrected) cart<->frac
transform and checks the structures are physically sane *before* any relaxation:

  * min interatomic distance (a real crystal has no atoms sitting on top of one
    another; the old transpose bug produced sub-Angstrom clashes),
  * volume per atom,
  * CHGNet initial energy/atom (should be negative for a plausible cell), and
  * relaxation drift (rmsd / dvol / de) -- should be far smaller than the
    8-11 eV/atom collapses we saw on the pre-fix registry.

Usage (inside the ROCm container):
  python scripts/validate_sampler_fix.py --ckpt data/runs/cdvae_long/cdvae_best.pt -n 8
"""

from __future__ import annotations

import argparse

import numpy as np


def min_pair_distance(structure) -> float:
    """Smallest interatomic distance (Angstrom) under PBC."""
    dm = structure.distance_matrix
    n = len(structure)
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(dm[iu].min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/runs/cdvae_long/cdvae_best.pt")
    ap.add_argument("-n", "--n", type=int, default=8)
    ap.add_argument("--steps-per-level", type=int, default=8)
    ap.add_argument("--relax", action="store_true", help="also CHGNet-relax each structure")
    args = ap.parse_args()

    from phlogiston.discovery.loop import load_generator, sample_candidates

    gen = load_generator(args.ckpt)
    print(f"[validate] generator loaded; sampling {args.n} structures ...", flush=True)
    structures = sample_candidates(gen, args.n, steps_per_level=args.steps_per_level)
    print(f"[validate] {len(structures)} structures decoded\n", flush=True)

    calc = None
    if args.relax:
        from phlogiston.verify.potential import load_calculator
        from phlogiston.verify.relax import relax_structure

        calc = load_calculator("chgnet")

    hdr = f"{'#':>2}  {'formula':<16}{'nat':>4}{'mind(A)':>9}{'V/atom':>9}"
    if calc is not None:
        hdr += f"{'e0/at':>9}{'e1/at':>9}{'de':>8}{'rmsd':>7}{'dV':>7}{'conv':>6}"
    print(hdr)
    print("-" * len(hdr))

    clashes = 0
    for i, s in enumerate(structures, 1):
        mind = min_pair_distance(s)
        if not np.isnan(mind) and mind < 0.7:
            clashes += 1
        row = (
            f"{i:>2}. {s.composition.reduced_formula:<15}"
            f"{len(s):>4}{mind:>9.3f}{s.volume / len(s):>9.2f}"
        )
        if calc is not None:
            try:
                r = relax_structure(s, calc, steps=300)
                row += (
                    f"{r.initial_energy_per_atom:>9.2f}{r.energy_per_atom:>9.2f}"
                    f"{r.de:>8.2f}{r.rmsd:>7.2f}{r.dvol_frac:>7.2f}"
                    f"{('yes' if r.converged else 'no'):>6}"
                )
            except Exception as e:  # noqa: BLE001
                row += f"  relax failed: {type(e).__name__}"
        print(row, flush=True)

    print(f"\n[validate] {clashes}/{len(structures)} structures have a sub-0.7A clash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
