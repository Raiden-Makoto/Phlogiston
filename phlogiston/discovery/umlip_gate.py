"""uMLIP in-the-loop stability gate (Tier-1.5).

The discovery predictor is a good stability model *on the training manifold*
(MAE ~0.01 eV/atom on held-out MP), but it is **blind off-manifold**: it rates
~90% of raw CDVAE samples as stable while a real potential relaxes them ~+0.25
eV/atom off the hull. So the cheap predictor gate cannot filter generation
quality -- the truth only appears after physical relaxation.

This module closes that blind spot by pulling the Tier-2 relax + self-consistent
hull (``verify.relax`` / ``verify.hull``) *into the discovery loop*, applied to
the small set of candidates that survive the cheap gates. Each candidate is:

  1. relaxed with the primary uMLIP -> the relaxed cell becomes canonical, and
     drift diagnostics (rmsd / |dV| / de) flag off-manifold guesses;
  2. optionally placed on a self-consistent uMLIP hull -> ``e_above_hull_umlip``,
     an MP-comparable stability estimate independent of the predictor.

The candidate's ``energy_above_hull`` is then **overwritten with the uMLIP hull
distance** (the authoritative value), the predicted one is kept as
``energy_above_hull_pred`` for the calibration residual, and only candidates at
or below ``e_hull_max`` survive. To stay tractable, relaxation runs on at most
``max_candidates`` of the best (lowest predicted-e_hull) survivors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UmlipGateStats:
    considered: int = 0          # candidates handed to the gate
    relaxed: int = 0             # successfully relaxed
    relax_failed: int = 0
    drift_rejected: int = 0      # off-manifold (non-converged / excessive drift)
    hull_failed: int = 0         # MP/hull error (kept or dropped per policy)
    passed: int = 0              # survived the uMLIP stability gate
    residuals: list = field(default_factory=list)  # umlip - predicted, eV/atom


def umlip_relax_gate(
    candidates: list,
    *,
    backend: str = "chgnet",
    e_hull_max: float = 0.1,
    relax_steps: int = 300,
    relax_cell: bool = True,
    with_hull: bool = True,
    max_candidates: int | None = None,
    max_rmsd: float | None = None,
    max_dvol: float | None = None,
    ehull_cutoff: float = 0.05,
    competitor_relax_steps: int = 200,
    cache_dir: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
    verbose: bool = True,
) -> tuple[list, UmlipGateStats]:
    """Relax + (optionally) hull-check ``candidates``; return survivors + stats.

    Survivors have their ``structure`` replaced by the relaxed cell and their
    ``energy_above_hull`` overwritten with ``e_above_hull_umlip`` (when the hull
    is built). ``energy_above_hull_pred`` preserves the original predicted value.

    ``max_candidates`` caps the (expensive) relaxations to the most-promising
    candidates by predicted hull distance. ``with_hull=False`` runs a fast
    relax+drift-only pass (no Materials Project round-trip): it cannot compute a
    true hull distance, so it gates purely on convergence + drift.
    """
    from phlogiston.verify.hull import CompetitorCache, build_competitor_entries, refined_hull_distance
    from phlogiston.verify.potential import load_calculator
    from phlogiston.verify.relax import relax_structure

    def log(m):
        if verbose:
            print(m, flush=True)

    stats = UmlipGateStats()
    if not candidates:
        return [], stats

    # Prioritize by predicted hull distance so a capped budget spends on the
    # candidates most likely to verify.
    ordered = sorted(
        candidates,
        key=lambda c: c.properties.get("energy_above_hull", float("inf")),
    )
    targets = ordered[:max_candidates] if max_candidates else ordered
    stats.considered = len(targets)
    log(f"[umlip-gate] loading {backend} calculator on {device or 'auto'} ...")
    calc = load_calculator(backend, device=device)
    cache = None
    if with_hull:
        cache_path = Path(cache_dir or "umlip_gate_cache") / f"umlip_{backend}.json"
        cache = CompetitorCache(cache_path)
    log(f"[umlip-gate] relaxing {len(targets)} candidates "
        f"({'relax+hull' if with_hull else 'relax+drift only'}, gate e_hull<={e_hull_max}) ...")

    survivors: list = []
    for c in targets:
        try:
            rr = relax_structure(c.structure, calc, steps=relax_steps, relax_cell=relax_cell)
        except Exception as exc:  # noqa: BLE001  one bad cell shouldn't abort the batch
            stats.relax_failed += 1
            log(f"[umlip-gate] {c.formula:<16} relax FAILED ({exc})")
            continue
        stats.relaxed += 1

        # Relaxed cell is canonical downstream.
        c.structure = rr.structure
        c.formula = rr.structure.composition.reduced_formula
        c.properties["relax_rmsd"] = round(rr.rmsd, 4)
        c.properties["relax_dvol"] = round(rr.dvol_frac, 4)
        c.properties["relax_de"] = round(rr.de, 4)
        c.properties["relax_converged"] = bool(rr.converged)

        # Drift prefilter: a huge move to the minimum means the generator's guess
        # was off-manifold -- exactly the case the predictor can't see.
        drift_bad = (not rr.converged) \
            or (max_rmsd is not None and rr.rmsd > max_rmsd) \
            or (max_dvol is not None and rr.dvol_frac > max_dvol)
        if drift_bad:
            stats.drift_rejected += 1
            continue

        if not with_hull:
            # No hull -> can't gate on absolute stability; keep drift-clean cells.
            survivors.append(c)
            stats.passed += 1
            continue

        elements = sorted({str(el) for el in rr.structure.composition.elements})
        try:
            competitors = build_competitor_entries(
                elements, calc, cache,
                ehull_cutoff=ehull_cutoff, relax_steps=competitor_relax_steps,
                api_key=api_key, verbose=False,
            )
            hull = refined_hull_distance(rr.structure, rr.energy, competitors)
        except Exception as exc:  # noqa: BLE001  MP/hull failure -> drop (can't verify)
            stats.hull_failed += 1
            log(f"[umlip-gate] {c.formula:<16} hull FAILED ({exc}) -> dropped")
            continue

        e_pred = c.properties.get("energy_above_hull")
        e_umlip = hull.e_above_hull_umlip
        if e_pred is not None:
            stats.residuals.append(e_umlip - e_pred)
        c.properties["energy_above_hull_pred"] = e_pred
        c.properties["energy_above_hull"] = round(e_umlip, 4)   # authoritative
        c.properties["e_above_hull_umlip"] = round(e_umlip, 4)
        c.properties["formation_energy_umlip"] = round(hull.formation_energy_umlip, 4)

        keep = e_umlip <= e_hull_max
        log(f"[umlip-gate] {c.formula:<16} e_hull_umlip={e_umlip:+.3f} "
            f"(pred {('%+.3f' % e_pred) if e_pred is not None else '  --'})  "
            f"rmsd={rr.rmsd:.2f} de={rr.de:+.2f}  "
            f"[{'PASS' if keep else 'screened'}]  ({hull.n_competitors} competitors)")
        if keep:
            survivors.append(c)
            stats.passed += 1

    if stats.residuals:
        import statistics
        mean = statistics.fmean(stats.residuals)
        med = statistics.median(stats.residuals)
        log(f"[umlip-gate] predictor residual (umlip - pred): n={len(stats.residuals)} "
            f"mean={mean:+.4f} median={med:+.4f} eV/atom "
            f"(positive => predictor optimistic; use as --stability-bias)")
    log(f"[umlip-gate] {stats.passed}/{stats.considered} passed "
        f"({stats.drift_rejected} drift-rejected, {stats.relax_failed} relax-failed, "
        f"{stats.hull_failed} hull-failed)")
    return survivors, stats
