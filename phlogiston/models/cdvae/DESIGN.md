# CDVAE — DESIGN

Ab-initio crystal generator (composition + lattice + structure). Chosen over
DiffCSP because the target compound's formula is unknown. See `pipeline.md` §6.

> Status: **skeleton** — module-level architecture to be filled next.

## Sub-modules (planned)
- `encoder`: VAE head on the shared `encoder` block → `z`.
- `predictors`: `num_atoms`, `lattice`, `composition` (+ optional property) from `z`.
- `decoder`: noise-conditioned score net → coord score (`1o`) + atom-type logits.
- `diffusion`: noise schedules, score-matching loss, annealed Langevin sampling.
- `cdvae`: assembly + training losses (KL / num / lattice / composition / coord / type).

## To specify
- Latent dim `d`, `N_max`, lattice parameterization, fractional vs Cartesian
  coordinate diffusion, atom-type generation scheme, conditioning method.

## Open decisions
- See `pipeline.md` §11 (CDVAE items).
