"""End-to-end discovery loop (pipeline §7): sample from the CDVAE generator,
filter for novelty, gate on predicted stability, screen properties with the
Predictor, and rank by the multi-objective goal.
"""

from __future__ import annotations

import torch

from phlogiston.discovery.feasibility import feasibility_filter
from phlogiston.discovery.novelty import dedup, load_reference_formulas, novelty_filter
from phlogiston.discovery.rank import rank_candidates
from phlogiston.discovery.screen import PropertyScreen, load_predictor, load_synth_model
from phlogiston.models.cdvae import CDVAE


def load_generator(ckpt_path: str, device: str | None = None, use_ema: bool = True) -> CDVAE:
    """Rebuild a CDVAE from a checkpoint; load the EMA weights by default (what
    the trainer selected on and what samples best)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt.get("hparams", {})
    model = CDVAE(
        latent_dim=hp.get("latent_dim", 256),
        mul=hp.get("mul", 128),
        n_layers=hp.get("n_layers", 3),
        correlation=hp.get("correlation", 2),
        n_max=hp.get("n_max", 64),
        beta=hp.get("beta", 0.01),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    if use_ema and ckpt.get("ema"):
        shadow = ckpt["ema"]["shadow"]
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in shadow:
                    p.data.copy_(shadow[name].to(device))
    model.eval()
    return model


@torch.no_grad()
def sample_candidates(generator: CDVAE, n: int, steps_per_level: int = 8) -> list:
    """Draw ``n`` ab-initio structures via the batched GPU sampler."""
    try:
        return generator.sample_batch(n=n, steps_per_level=steps_per_level)
    except Exception:  # noqa: BLE001  fall back to per-structure sampling
        out = []
        for _ in range(n):
            try:
                out.append(generator.sample(steps_per_level=steps_per_level))
            except Exception:  # noqa: BLE001
                continue
        return out


def load_latent_head(head_ckpt: str, latent_dim: int, device: str):
    """Load a fitted LatentPropertyHead from a checkpoint."""
    from phlogiston.models.cdvae import LatentPropertyHead

    ckpt = torch.load(head_ckpt, map_location=device)
    head = LatentPropertyHead(latent_dim, hidden=ckpt.get("hidden", 256)).to(device)
    head.load_state_dict(ckpt["model"])
    return head.eval()


def discover(
    generator_ckpt: str,
    predictor_ckpt: str,
    data_root: str = "data",
    *,
    stability_ckpt: str | None = None,
    synth_ckpt: str | None = None,
    synth_min: float = 0.3,
    latent_head_ckpt: str | None = None,
    profile: dict[str, float] | None = None,
    cond_steps: int = 100,
    cond_trust_radius: float = 8.0,
    n_samples: int = 128,
    steps_per_level: int = 4,
    e_hull_max: float = 0.1,
    rho_max: float | None = None,
    weights: dict[str, float] | None = None,
    do_dedup: bool = True,
    check_novelty: bool = True,
    check_feasibility: bool = True,
    save_dir: str | None = None,
    max_elements: int = 5,
    max_reduced_atoms: int = 40,
    allow_radioactive: bool = False,
    device: str | None = None,
    verbose: bool = True,
):
    """Run the full loop and return ranked, novel, stable candidates.

    ``stability_ckpt`` (recommended) is a separate stability-specialist model
    used for the gate; ``predictor_ckpt`` scores the mechanical/thermal
    properties. If omitted, the property model gates too.

    ``latent_head_ckpt`` (optional) enables **property-conditioned** generation:
    latents are gradient-ascended toward ``profile`` before decoding, instead of
    sampled unconditionally.
    """

    def log(m):
        if verbose:
            print(m, flush=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[discover] loading generator + predictor on {device} ...")
    generator = load_generator(generator_ckpt, device)
    predictor = load_predictor(predictor_ckpt, device)
    stability_model = load_predictor(stability_ckpt, device) if stability_ckpt else None
    if stability_model is not None:
        log("[discover] decoupled gate: stability from a separate specialist model")
    synth_model = load_synth_model(synth_ckpt, device) if synth_ckpt else None
    if synth_model is not None:
        log("[discover] Tier-1 synthesizability model loaded (learned synthesis prior)")
    screen = PropertyScreen(
        predictor, stability_model=stability_model, synth_model=synth_model, device=device
    )

    if latent_head_ckpt is not None:
        from phlogiston.models.cdvae import generate_conditioned

        head = load_latent_head(latent_head_ckpt, generator.latent_dim, device)
        log(f"[discover] property-conditioned generation of {n_samples} candidates ...")
        structures = generate_conditioned(
            generator, head, n_samples, profile=profile, steps=cond_steps,
            trust_radius=cond_trust_radius, steps_per_level=steps_per_level, device=device,
        )
    else:
        log(f"[discover] sampling {n_samples} candidates (unconditional) ...")
        structures = sample_candidates(generator, n_samples, steps_per_level)
    log(f"[discover] {len(structures)} valid structures generated")

    scored = screen.score(structures)
    log(f"[discover] {len(scored)} featurized + scored")

    if do_dedup:
        scored = dedup(scored)
        log(f"[discover] {len(scored)} unique after dedup")

    if check_novelty:
        ref = load_reference_formulas(data_root)
        if ref:
            scored, known = novelty_filter(scored, ref)
            log(f"[discover] {len(scored)} novel formulas ({len(known)} already in GNoME/MP)")
        else:
            log("[discover] no reference formulas found; skipping novelty filter")

    if check_feasibility:  # Tier-0 composition sanity (synthesizability first pass)
        scored, infeasible = feasibility_filter(
            scored,
            max_elements=max_elements,
            max_reduced_atoms=max_reduced_atoms,
            allow_radioactive=allow_radioactive,
        )
        log(
            f"[discover] {len(scored)} pass Tier-0 feasibility "
            f"({len(infeasible)} rejected: too many elements / radioactive / implausible stoichiometry)"
        )

    if synth_model is not None and synth_min > 0:  # Tier-1 learned synthesizability gate
        # Deliberately loose: the model reflects *today's* synthesis record, so a
        # low bar admits borderline candidates that near-future methods could
        # reach, while still culling the clearly-implausible. The synth score is
        # kept on every card so the ranking still favors the more-synthesizable.
        before = len(scored)
        scored = [c for c in scored if c.properties.get("synthesizability", 0.0) >= synth_min]
        log(f"[discover] {len(scored)} pass Tier-1 synthesizability >= {synth_min} ({before - len(scored)} rejected)")

    ranked = rank_candidates(scored, rho_max=rho_max, e_hull_max=e_hull_max, weights=weights)
    log(f"[discover] {len(ranked)} candidates pass stability gate (e_hull<={e_hull_max})")
    if save_dir and ranked:
        save_candidates(ranked, save_dir, verbose=verbose)
    return ranked


# Persisted columns for the candidate registry (fixed order).
_CANDIDATE_COLUMNS = [
    "id", "formula", "run_id", "timestamp", "score", "is_pareto",
    "feasibility", "synthesizability", "energy_above_hull", "density",
    "bulk_modulus_vrh", "shear_modulus_vrh", "vickers_hardness",
    "fracture_toughness", "debye_temperature", "slack_thermal_conductivity",
    "formation_energy_per_atom",
]


def save_candidates(candidates, save_dir: str, *, run_id: str | None = None,
                    dedup_existing: bool = True, verbose: bool = True) -> int:
    """Persist ranked candidates to ``save_dir``: write one CIF per survivor and
    append to an accumulating ``candidates.csv`` (all scores + properties).

    Deduplicates against previously-saved structures (StructureMatcher) so
    repeated runs build one growing, unique shortlist -- exactly the durable set
    to feed Tier-2 DFT. Returns the number of newly-added candidates.
    """
    import csv
    import time
    from pathlib import Path

    save = Path(save_dir)
    cifs = save / "cifs"
    cifs.mkdir(parents=True, exist_ok=True)
    csv_path = save / "candidates.csv"
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")

    next_id = 0
    if csv_path.exists():
        import pandas as pd

        prev = pd.read_csv(csv_path)
        next_id = int(prev["id"].max()) + 1 if len(prev) else 0

    existing = []
    matcher = None
    if dedup_existing:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.core import Structure

        matcher = StructureMatcher()
        for p in sorted(cifs.glob("*.cif")):
            try:
                existing.append(Structure.from_file(str(p)))
            except Exception:  # noqa: BLE001
                pass

    added = 0
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CANDIDATE_COLUMNS)
        if write_header:
            w.writeheader()
        for c in candidates:
            s = c.structure
            if matcher is not None and any(matcher.fit(s, e) for e in existing):
                continue  # already in the registry
            cid = next_id
            next_id += 1
            fname = f"{cid:05d}_{c.formula}.cif".replace("/", "_")
            try:
                s.to(filename=str(cifs / fname), fmt="cif")
            except Exception:  # noqa: BLE001  (skip CIFs pymatgen can't serialize)
                pass
            p = c.properties
            row = {
                "id": cid, "formula": c.formula, "run_id": run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "score": round(float(c.score), 4), "is_pareto": bool(c.is_pareto),
            }
            for k in _CANDIDATE_COLUMNS[6:]:
                v = p.get(k)
                row[k] = round(float(v), 4) if v is not None else ""
            w.writerow(row)
            existing.append(s)
            added += 1
    if verbose:
        print(f"[discover] saved {added} new candidates -> {csv_path} ({next_id} total in registry)",
              flush=True)
    return added


def format_report(candidates, top_k: int = 10) -> str:
    """Human-readable property card for the top candidates."""
    if not candidates:
        return "No candidates survived the stability gate."
    lines = [f"Top {min(top_k, len(candidates))} candidates:\n"]
    for rank, c in enumerate(candidates[:top_k], start=1):
        p = c.properties
        star = " [Pareto]" if c.is_pareto else ""
        feas = p.get("feasibility")
        feas_str = f"  feas={feas:.2f}" if feas is not None else ""
        synth = p.get("synthesizability")
        synth_str = f"  synth={synth:.2f}" if synth is not None else ""
        lines.append(
            f"{rank:>2}. {c.formula:<14} score={c.score:.3f}{star}{feas_str}{synth_str}\n"
            f"      Ehull={p.get('energy_above_hull', float('nan')):+.3f} eV/atom  "
            f"rho={p.get('density', float('nan')):.2f} g/cm^3\n"
            f"      K={p.get('bulk_modulus_vrh', float('nan')):.0f}  "
            f"G={p.get('shear_modulus_vrh', float('nan')):.0f} GPa  "
            f"Hv={p.get('vickers_hardness', float('nan')):.1f} GPa  "
            f"Kic={p.get('fracture_toughness', float('nan')):.2f}\n"
            f"      Debye={p.get('debye_temperature', float('nan')):.0f} K  "
            f"kappa={p.get('slack_thermal_conductivity', float('nan')):.1f} W/m/K"
        )
    return "\n".join(lines)
