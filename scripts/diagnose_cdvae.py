"""CDVAE generator diagnostic: how far off the stable manifold does the
unconditional generative prior sit?

The CDVAE training loss is pure reconstruction (KL + num-atoms + lattice +
composition + coord score-matching + type); it carries no energy/stability
signal. This script quantifies the consequence by running the *same* stability
specialist over:

  (A) unconditional CDVAE samples, and
  (B) real held-out training structures (from a graph shard),

and comparing the predicted energy-above-hull distributions. Because both use
the identical predictor, the gap is a clean measure of how off-distribution the
generated structures are -- independent of the predictor's absolute calibration.

Also reports the generation yield funnel (sampled -> geometrically sane ->
featurizable -> Tier-0 feasible) and the predictor's calibration on real data.

Run inside the phlogiston container (ROCm torch + deps installed).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from phlogiston.data.dataset import TARGET_KEYS, ShardedCrystalDataset, collate
from phlogiston.discovery.feasibility import feasibility_filter
from phlogiston.discovery.loop import drop_clashed, load_generator, sample_candidates
from phlogiston.discovery.screen import load_predictor
from phlogiston.models.predictor import PREDICT_KEYS

EHULL_PRED = PREDICT_KEYS.index("energy_above_hull")
EHULL_TGT = TARGET_KEYS.index("energy_above_hull")


def summarize(name: str, x: np.ndarray) -> None:
    if len(x) == 0:
        print(f"  {name:<28} (empty)")
        return
    pct = np.percentile(x, [10, 25, 50, 75, 90])
    print(
        f"  {name:<28} n={len(x):<5} mean={x.mean():+.3f} median={np.median(x):+.3f}  "
        f"p10/25/50/75/90={pct[0]:+.2f}/{pct[1]:+.2f}/{pct[2]:+.2f}/{pct[3]:+.2f}/{pct[4]:+.2f}"
    )
    for thr in (0.0, 0.05, 0.1, 0.2):
        frac = float((x <= thr).mean())
        print(f"      frac <= {thr:<4} : {frac*100:5.1f}%   ({int((x <= thr).sum())})")


@torch.no_grad()
def score_graphs(stability, graphs, device, batch_size=64) -> np.ndarray:
    """Predicted energy_above_hull (eV/atom) for a list of CrystalGraphs."""
    out = []
    for i in range(0, len(graphs), batch_size):
        chunk = graphs[i : i + batch_size]
        buf = [(g, torch.zeros(1), torch.zeros(1, dtype=torch.bool)) for g in chunk]
        batch = collate(buf).to(device)
        preds = stability(batch).cpu()
        out.append(preds[:, EHULL_PRED].numpy())
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="data/runs/cdvae_long/cdvae_best.pt")
    ap.add_argument("--stability-ckpt", default="data/runs/ft_stage1/predictor_stage1_last.pt")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--gen-batch-size", type=int, default=128)
    ap.add_argument("--steps-per-level", type=int, default=8)
    ap.add_argument("--n-real", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[diag] device={device}")

    print(f"[diag] loading generator {args.generator}")
    generator = load_generator(args.generator, device=device)
    print(f"[diag] loading stability specialist {args.stability_ckpt}")
    stability = load_predictor(args.stability_ckpt, device=device)

    # ---- (A) unconditional CDVAE samples --------------------------------
    print(f"\n[diag] sampling {args.n_samples} unconditional structures ...")
    structures = sample_candidates(
        generator, args.n_samples, args.steps_per_level, args.gen_batch_size
    )
    n_gen = len(structures)
    sane, clashed = drop_clashed(structures)
    n_sane = len(sane)

    from phlogiston.data.graph import structure_to_graph

    gen_graphs, n_featfail = [], 0
    for s in sane:
        try:
            gen_graphs.append(structure_to_graph(s, cutoff=6.0))
        except Exception:  # noqa: BLE001
            n_featfail += 1
    gen_ehull = score_graphs(stability, gen_graphs, device)

    # Tier-0 feasibility on the sane set (reuse the discovery filter; needs the
    # ScoredCandidate wrapper -- but feasibility_filter works on objects with a
    # .structure; approximate here by counting via a lightweight wrapper).
    class _C:
        def __init__(self, s):
            self.structure = s
            self.formula = s.composition.reduced_formula
            self.properties = {}

    feasible, infeasible = feasibility_filter([_C(s) for s in sane])
    n_feasible = len(feasible)

    print("\n=== GENERATION YIELD FUNNEL (unconditional) ===")
    print(f"  sampled                : {n_gen}")
    print(f"  geometrically sane     : {n_sane}  ({clashed} clashed, <0.7A)")
    print(f"  featurizable           : {len(gen_graphs)}  ({n_featfail} failed)")
    print(f"  Tier-0 feasible        : {n_feasible}  ({len(infeasible)} rejected)")

    # ---- (B) real held-out structures -----------------------------------
    print(f"\n[diag] loading a shard for the real baseline ...")
    ds = ShardedCrystalDataset(args.data_root, max_shards=1)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    real_graphs, real_true = [], []
    for i in idxs:
        g, y, m = ds[i]
        if bool(m[EHULL_TGT]):
            real_graphs.append(g)
            real_true.append(float(y[EHULL_TGT]))
        if len(real_graphs) >= args.n_real:
            break
    real_true = np.array(real_true)
    real_pred = score_graphs(stability, real_graphs, device)

    # ---- report ---------------------------------------------------------
    print("\n=== PREDICTED energy_above_hull (same stability model) ===")
    summarize("GENERATED (CDVAE prior)", gen_ehull)
    summarize("REAL (held-out training)", real_pred)

    print("\n=== REAL ground-truth energy_above_hull (calibration check) ===")
    summarize("REAL true e_hull", real_true)
    if len(real_pred) == len(real_true) and len(real_true):
        resid = real_pred - real_true
        print(
            f"  predictor residual on real: mean={resid.mean():+.3f}  "
            f"MAE={np.abs(resid).mean():.3f} eV/atom"
        )

    print("\n=== VERDICT ===")
    if len(gen_ehull) and len(real_pred):
        gap = float(np.median(gen_ehull) - np.median(real_pred))
        gen_stable = float((gen_ehull <= 0.1).mean())
        real_stable = float((real_pred <= 0.1).mean())
        print(f"  median predicted e_hull gap (gen - real): {gap:+.3f} eV/atom")
        print(f"  fraction predicted stable (<=0.1): gen {gen_stable*100:.1f}%  vs  real {real_stable*100:.1f}%")


if __name__ == "__main__":
    main()
