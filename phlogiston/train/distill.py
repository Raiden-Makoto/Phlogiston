"""Relaxation self-distillation corpus for the CDVAE generator.

The CDVAE diagnostic showed the generator emits *plausible-looking but
physically unrelaxed* geometry (~1 A drift under a uMLIP) for its own novel
compositions -- a manifold the DFT-relaxed training data doesn't cover. Plain
retraining on the (already-stable) corpus can't fix that; the corrective signal
is the generator's *own outputs, relaxed*.

This module builds that signal: sample structures from the generator, relax them
with the primary uMLIP, and write the relaxed cells as graph shards in the format
``ShardedCrystalDataset`` consumes. Fine-tuning CDVAE on these (its loss is
self-supervised reconstruction -- labels are unused) pulls the decoder toward the
relaxed manifold for exactly the compositions it tends to produce.

Relaxation is the cost bottleneck, so runs are shardable: each process writes
``shard_<shard_start+i>.pt`` into a shared ``out_root``, letting several GPUs fill
one corpus in parallel (no manifest is written -- the dataset globs shards).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from phlogiston.data.dataset import TARGET_KEYS
from phlogiston.data.graph import structure_to_graph
from phlogiston.data.precompute import _vector_from_labels, shard_dir


def build_distill_corpus(
    generator_ckpt: str,
    out_root: str,
    *,
    n_samples: int = 2000,
    gen_batch_size: int | None = 384,
    steps_per_level: int = 8,
    backend: str = "chgnet",
    relax_steps: int = 150,
    fmax: float = 0.05,
    keep_fmax: float = 0.2,
    require_energy_drop: bool = True,
    min_dist: float = 0.7,
    cutoff: float = 6.0,
    shard_size: int = 2000,
    shard_start: int = 0,
    tag: str = "0",
    store_disp: bool = False,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Generate + relax ``n_samples`` structures and write relaxed graph shards.

    A relaxed structure is kept when it is converged-ish (max force <=
    ``keep_fmax``) and -- if ``require_energy_drop`` -- actually lowered in energy
    (a sane relaxation). Returns a summary dict.

    When ``store_disp`` is set, each record additionally carries ``relax_disp``:
    the per-atom Cartesian displacement ``cart_generated - cart_relaxed`` (in the
    relaxed cell's frame, min-image), which supervises the relaxation-consistency
    loss. Relaxation preserves atom order, so the pairing is exact.
    """
    import numpy as np
    from phlogiston.discovery.loop import drop_clashed, load_generator, sample_candidates
    from phlogiston.verify.potential import load_calculator
    from phlogiston.verify.relax import relax_structures

    def log(m):
        if verbose:
            print(m, flush=True)

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[distill] device={device} | loading generator {generator_ckpt}")
    generator = load_generator(generator_ckpt, device=device)

    log(f"[distill] sampling {n_samples} structures (steps_per_level={steps_per_level}) ...")
    structures = sample_candidates(generator, n_samples, steps_per_level, gen_batch_size)
    sane, clashed = drop_clashed(structures, min_dist=min_dist)
    log(f"[distill] {len(sane)} geometrically sane ({clashed} clashed); loading {backend} ...")

    calc = load_calculator(backend, device=device)
    log(f"[distill] relaxing {len(sane)} structures (uMLIP {backend}, <= {relax_steps} steps) ...")
    t0 = time.time()
    results = relax_structures(sane, calc, steps=relax_steps, fmax=fmax)

    records: list[dict] = []
    n_relax_fail = n_force = n_noe = 0
    rmsds: list[float] = []
    for j, rr in enumerate(results):
        if rr is None:
            n_relax_fail += 1
            continue
        if rr.max_force > keep_fmax:
            n_force += 1
            continue
        if require_energy_drop and rr.de > 1e-3:
            n_noe += 1
            continue
        try:
            g = structure_to_graph(rr.structure, cutoff=cutoff)
        except Exception:  # noqa: BLE001  degenerate relaxed cell -> skip
            continue
        vals, mask = _vector_from_labels({"density": float(rr.structure.density)})
        graph_np = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in g.__dict__.items()}
        rec = {
            "id": f"distill:{tag}:{j}",
            "source": "distill",
            "graph": graph_np,
            "y": vals,
            "mask": mask,
        }
        if store_disp:
            # cart_generated - cart_relaxed in the relaxed cell's frame (min-image),
            # matching relax._drift so d supervises the score toward the minimum.
            relaxed = rr.structure
            gen = sane[j]
            dfrac = np.asarray(gen.frac_coords) - np.asarray(relaxed.frac_coords)
            dfrac -= np.round(dfrac)
            disp = dfrac @ relaxed.lattice.matrix  # [N,3] Cartesian, gen - relaxed
            rec["relax_disp"] = disp.astype("float32")
        records.append(rec)
        rmsds.append(rr.rmsd)

    sd = shard_dir(out_root)
    sd.mkdir(parents=True, exist_ok=True)
    n_shards = 0
    for i in range(0, len(records), shard_size):
        torch.save(records[i : i + shard_size], sd / f"shard_{shard_start + n_shards:06d}.pt")
        n_shards += 1

    mean_rmsd = sum(rmsds) / len(rmsds) if rmsds else 0.0
    log(f"[distill] kept {len(records)}/{len(sane)} relaxed structures "
        f"({n_relax_fail} relax-failed, {n_force} force>{keep_fmax}, {n_noe} no-energy-drop) "
        f"in {n_shards} shards -> {sd}")
    log(f"[distill] mean relax RMSD of kept = {mean_rmsd:.3f} A  ({time.time() - t0:.0f}s relax)")
    return {
        "kept": len(records), "sane": len(sane), "clashed": clashed,
        "relax_failed": n_relax_fail, "force_rejected": n_force,
        "no_energy_drop": n_noe, "n_shards": n_shards,
        "mean_rmsd": mean_rmsd, "out_root": str(out_root),
        "n_targets": len(TARGET_KEYS),
    }
