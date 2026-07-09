"""End-to-end discovery loop (pipeline §7): sample from the CDVAE generator,
filter for novelty, gate on predicted stability, screen properties with the
Predictor, and rank by the multi-objective goal.
"""

from __future__ import annotations

import torch

from phlogiston.discovery.novelty import dedup, load_reference_formulas, novelty_filter
from phlogiston.discovery.rank import rank_candidates
from phlogiston.discovery.screen import PropertyScreen, load_predictor
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
    latent_head_ckpt: str | None = None,
    profile: dict[str, float] | None = None,
    cond_steps: int = 100,
    cond_trust_radius: float = 6.0,
    n_samples: int = 128,
    steps_per_level: int = 4,
    e_hull_max: float = 0.1,
    rho_max: float | None = None,
    weights: dict[str, float] | None = None,
    do_dedup: bool = True,
    check_novelty: bool = True,
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
    screen = PropertyScreen(predictor, stability_model=stability_model, device=device)

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

    ranked = rank_candidates(scored, rho_max=rho_max, e_hull_max=e_hull_max, weights=weights)
    log(f"[discover] {len(ranked)} candidates pass stability gate (e_hull<={e_hull_max})")
    return ranked


def format_report(candidates, top_k: int = 10) -> str:
    """Human-readable property card for the top candidates."""
    if not candidates:
        return "No candidates survived the stability gate."
    lines = [f"Top {min(top_k, len(candidates))} candidates:\n"]
    for rank, c in enumerate(candidates[:top_k], start=1):
        p = c.properties
        star = " [Pareto]" if c.is_pareto else ""
        lines.append(
            f"{rank:>2}. {c.formula:<14} score={c.score:.3f}{star}\n"
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
