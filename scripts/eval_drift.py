"""Physical drift eval: how far does a CDVAE generator's raw geometry sit from
the uMLIP energy minimum?

The generator emits *approximate* geometry. Relaxing each sample with a uMLIP and
measuring how far it moves (per-atom RMSD, cell volume change) and how hard it is
to converge (final max force, step count, energy drop) is the truest measure of
geometric quality -- and, unlike the stability predictor, it has no optimism
blind spot. Relaxation self-distillation should *reduce* this drift.

Samples N structures, drops clashed ones, relaxes every sane structure (not just
those that converge), and reports the full drift distribution. Run two
generators with the same --seed / --n-samples to compare head-to-head.

Run inside the phlogiston container (ROCm torch + CHGNet installed).
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch


def summarize(name: str, x: np.ndarray, thresholds: tuple[float, ...]) -> None:
    if len(x) == 0:
        print(f"  {name:<26} (empty)")
        return
    pct = np.percentile(x, [10, 25, 50, 75, 90])
    print(
        f"  {name:<26} n={len(x):<4} mean={x.mean():.3f} median={np.median(x):.3f}  "
        f"p10/25/50/75/90={pct[0]:.2f}/{pct[1]:.2f}/{pct[2]:.2f}/{pct[3]:.2f}/{pct[4]:.2f}"
    )
    for thr in thresholds:
        frac = float((x <= thr).mean())
        print(f"      frac <= {thr:<5}: {frac * 100:5.1f}%   ({int((x <= thr).sum())})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="data/runs/cdvae_long/cdvae_best.pt")
    ap.add_argument("--backend", default="chgnet")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--gen-batch-size", type=int, default=128)
    ap.add_argument("--steps-per-level", type=int, default=8)
    ap.add_argument("--relax-steps", type=int, default=200)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--min-dist", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from phlogiston.discovery.loop import drop_clashed, load_generator, sample_candidates
    from phlogiston.verify.potential import load_calculator
    from phlogiston.verify.relax import relax_structures

    print(f"[drift] device={device} | generator={args.generator}", flush=True)
    generator = load_generator(args.generator, device=device)

    print(f"[drift] sampling {args.n_samples} structures ...", flush=True)
    structures = sample_candidates(
        generator, args.n_samples, args.steps_per_level, args.gen_batch_size
    )
    sane, clashed = drop_clashed(structures, min_dist=args.min_dist)
    print(f"[drift] {len(sane)}/{len(structures)} geometrically sane ({clashed} clashed)", flush=True)

    calc = load_calculator(args.backend, device=device)
    print(f"[drift] relaxing {len(sane)} structures ({args.backend}, <= {args.relax_steps} steps) ...", flush=True)
    t0 = time.time()
    results = relax_structures(sane, calc, steps=args.relax_steps, fmax=args.fmax)

    rmsd, dvol, de, fmax_final, nsteps = [], [], [], [], []
    n_fail = 0
    for rr in results:
        if rr is None:
            n_fail += 1
            continue
        rmsd.append(rr.rmsd)
        dvol.append(rr.dvol_frac)
        de.append(rr.de)
        fmax_final.append(rr.max_force)
        nsteps.append(rr.n_steps)
    rmsd, dvol, de = np.array(rmsd), np.array(dvol), np.array(de)
    fmax_final, nsteps = np.array(fmax_final), np.array(nsteps)

    print(f"\n=== DRIFT UNDER uMLIP RELAXATION ({args.backend}) ===", flush=True)
    print(f"  generator      : {args.generator}")
    print(f"  sampled/sane   : {len(structures)}/{len(sane)}  ({clashed} clashed, {n_fail} relax-failed)")
    print(f"  relax wall     : {time.time() - t0:.0f}s\n")
    summarize("RMSD drift (A)", rmsd, (0.1, 0.25, 0.5, 1.0))
    summarize("|dV|/V (cell drift)", dvol, (0.05, 0.1, 0.25, 0.5))
    summarize("final max force (eV/A)", fmax_final, (0.05, 0.1, 0.2, 0.5))
    summarize("relax steps used", nsteps.astype(float), (50.0, 100.0, 150.0, 199.0))
    print("\n  energy drop de (eV/atom, more negative = more unrelaxed input):")
    if len(de):
        pct = np.percentile(de, [10, 25, 50, 75, 90])
        print(f"    mean={de.mean():+.3f} median={np.median(de):+.3f}  "
              f"p10/25/50/75/90={pct[0]:+.2f}/{pct[1]:+.2f}/{pct[2]:+.2f}/{pct[3]:+.2f}/{pct[4]:+.2f}")
    conv = float((fmax_final <= args.fmax).mean()) if len(fmax_final) else 0.0
    conv02 = float((fmax_final <= 0.2).mean()) if len(fmax_final) else 0.0
    print(f"\n  converged (force <= {args.fmax}): {conv * 100:.1f}%   |   force <= 0.2: {conv02 * 100:.1f}%")


if __name__ == "__main__":
    main()
