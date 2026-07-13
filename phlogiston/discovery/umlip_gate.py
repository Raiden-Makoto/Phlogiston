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
    hull_screened: int = 0       # relaxed + hulled but above the stability gate
    ensemble_checked: int = 0    # ran 2c cross-check
    ensemble_low_conf: int = 0   # members disagreed (off-distribution)
    phonon_checked: int = 0      # ran 2d phonons
    phonon_unstable: int = 0     # confirmed imaginary modes (dropped if required)
    passed: int = 0              # survived the full in-loop gate
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
    cross_backend: str | None = None,
    ensemble_spread_max: float = 0.05,
    do_phonons: bool = False,
    require_phonon_stable: bool = True,
    phonon_e_hull_max: float = 0.05,
    phonon_supercell_min_len: float = 8.0,
    phonon_displacement: float = 0.03,
    phonon_mesh: int = 8,
    phonon_tol: float = 0.1,
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

    On the candidates that clear the hull gate, two further in-loop checks run
    (mirroring Tier-2 post-processing, so the saved queue is fully verified):
    ``cross_backend`` (2c) re-relaxes with a second, independent uMLIP and flags
    members' disagreement; ``do_phonons`` (2d) runs finite-displacement phonons
    on near-hull survivors and -- when ``require_phonon_stable`` -- drops any with
    confirmed imaginary modes (a phonon *calc failure* keeps the candidate,
    annotated as unchecked).
    """
    from phlogiston.verify.ensemble import ensemble_cross_check
    from phlogiston.verify.hull import CompetitorCache, build_competitor_entries, refined_hull_distance
    from phlogiston.verify.phonons import phonon_stability
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
    calc2 = cache2 = None
    if with_hull:
        base = Path(cache_dir or "umlip_gate_cache")
        cache = CompetitorCache(base / f"umlip_{backend}.json")
        if cross_backend:
            calc2 = load_calculator(cross_backend, device=device)
            cache2 = CompetitorCache(base / f"umlip_{cross_backend}.json")
    extras = []
    if with_hull and cross_backend:
        extras.append(f"2c={cross_backend}")
    if with_hull and do_phonons:
        extras.append("2d=phonons")
    mode = "relax+hull" if with_hull else "relax+drift only"
    log(f"[umlip-gate] relaxing {len(targets)} candidates "
        f"({mode}{(', ' + ', '.join(extras)) if extras else ''}, gate e_hull<={e_hull_max}) ...")

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

        if e_umlip > e_hull_max:
            stats.hull_screened += 1
            log(f"[umlip-gate] {c.formula:<16} e_hull_umlip={e_umlip:+.3f} "
                f"(pred {('%+.3f' % e_pred) if e_pred is not None else '  --'})  "
                f"rmsd={rr.rmsd:.2f} de={rr.de:+.2f}  [screened]  ({hull.n_competitors} competitors)")
            continue
        log(f"[umlip-gate] {c.formula:<16} e_hull_umlip={e_umlip:+.3f} "
            f"(pred {('%+.3f' % e_pred) if e_pred is not None else '  --'})  "
            f"rmsd={rr.rmsd:.2f} de={rr.de:+.2f}  [hull PASS]  ({hull.n_competitors} competitors)")

        # 2c ensemble cross-check on the hull-passers.
        if calc2 is not None:
            try:
                ens = ensemble_cross_check(
                    rr.structure, e_umlip, calc2, cache2, cross_backend,
                    spread_max=ensemble_spread_max, relax_steps=relax_steps,
                    competitor_relax_steps=competitor_relax_steps, ehull_cutoff=ehull_cutoff,
                    api_key=api_key, verbose=False,
                )
                stats.ensemble_checked += 1
                if ens.confidence == "low":
                    stats.ensemble_low_conf += 1
                c.properties["e_above_hull_umlip_secondary"] = round(ens.e_above_hull_secondary, 4)
                c.properties["ensemble_spread"] = round(ens.spread, 4)
                c.properties["ensemble_confidence"] = ens.confidence
                log(f"[umlip-gate]     2c {cross_backend}: e_hull={ens.e_above_hull_secondary:+.3f} "
                    f"spread={ens.spread:.3f} -> {ens.confidence}-confidence")
            except Exception as exc:  # noqa: BLE001  cross-check failure is non-fatal
                log(f"[umlip-gate]     2c cross-check failed: {exc}")

        # 2d phonons on near-hull survivors; drop confirmed-imaginary if required.
        if do_phonons and e_umlip <= phonon_e_hull_max:
            try:
                ph = phonon_stability(
                    rr.structure, calc,
                    supercell_min_len=phonon_supercell_min_len,
                    displacement=phonon_displacement, mesh=phonon_mesh, tol_thz=phonon_tol,
                )
                stats.phonon_checked += 1
                c.properties["min_phonon_freq"] = round(ph.min_freq_thz, 4)
                c.properties["dynamically_stable"] = bool(ph.dynamically_stable)
                log(f"[umlip-gate]     2d phonons: min_freq={ph.min_freq_thz:+.2f} THz "
                    f"({'stable' if ph.dynamically_stable else 'IMAGINARY'})")
                if not ph.dynamically_stable:
                    stats.phonon_unstable += 1
                    if require_phonon_stable:
                        continue  # dynamically unstable -> drop from the queue
            except Exception as exc:  # noqa: BLE001  phonon failure keeps the candidate (unchecked)
                log(f"[umlip-gate]     2d phonons failed: {exc}")

        survivors.append(c)
        stats.passed += 1

    if stats.residuals:
        import statistics
        mean = statistics.fmean(stats.residuals)
        med = statistics.median(stats.residuals)
        log(f"[umlip-gate] predictor residual (umlip - pred): n={len(stats.residuals)} "
            f"mean={mean:+.4f} median={med:+.4f} eV/atom "
            f"(positive => predictor optimistic; use as --stability-bias)")
    extra = ""
    if stats.phonon_checked:
        extra += f", {stats.phonon_unstable}/{stats.phonon_checked} phonon-unstable"
    if stats.ensemble_checked:
        extra += f", {stats.ensemble_low_conf}/{stats.ensemble_checked} low-confidence"
    log(f"[umlip-gate] {stats.passed}/{stats.considered} passed "
        f"({stats.hull_screened} hull-screened, {stats.drift_rejected} drift-rejected, "
        f"{stats.relax_failed} relax-failed, {stats.hull_failed} hull-failed{extra})")
    return survivors, stats
