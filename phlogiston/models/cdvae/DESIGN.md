# CDVAE — DESIGN

Ab-initio crystal **generator** (composition + lattice + structure), chosen over
DiffCSP because the target compound's formula is unknown. A **separate model**
from the predictor: its own weights and training objective; it reuses only the
equivariant *blocks* in `phlogiston/layers` (not the trained predictor). At
discovery time it feeds candidates to the predictor (generate → screen). See
`pipeline.md` §6.

---

## 0. Data & I/O

- Trains on the same precomputed graphs (`ShardedCrystalDataset`) — the stable
  structures (MP + GNoME); labels are optional (only for property conditioning).
- Generation output: a `pymatgen`-ready `(lattice, frac_coords, atom_types)`.

## 1. Sub-modules

| Module (`models/cdvae/`) | Role |
|---|---|
| `encoder.py` | VAE encoder: `CrystalEncoder` (graph pooling) → `μ, logσ²` → latent `z ∈ R^d` |
| `predictors.py` | from `z`: `num_atoms`, `lattice`, `composition` (+ optional property) |
| `decoder.py` | noise-conditioned score net: per-atom **coord score** (`1o`) + **type logits** |
| `diffusion.py` | noise schedules, score-matching loss, annealed Langevin sampling |
| `cdvae.py` | assembly + the composite training loss |

## 2. Latent encoder
```
graph ─► CrystalEncoder ─► graph_feats [B, mul] ─► Linear→ μ [B,d], logσ² [B,d]
z = μ + σ·ε         (reparameterization; ε ~ N(0,I))
```
`d` = latent dim (open decision, e.g. 256). Uses the shared encoder (separate
weights from the predictor).

## 3. Latent predictors (global, from `z`)
- `num_atoms`: classification over `1..N_max` (or Poisson-ish regression).
- `lattice`: 6 params — **lengths (3) + angles (3)**, predicted in a normalized
  space (open: full matrix vs Niggli-reduced).
- `composition`: per-element logits → expected element fractions → integer
  counts (consistent with `num_atoms`).
- *(optional)* `property`: our target vector from `z`, for conditioning (§6).

## 4. Score decoder (the denoiser)
Given a **noisy** structure (frac coords + atom types) in the predicted lattice,
`z`, and noise level `σ_t`:
- Build a crystal graph from the *noisy* coords (nearest-image periodicity),
  featurize with the shared blocks, and add a **noise-level embedding**
  (`layers/noise`, sinusoidal σ/t embedding) into node features.
- Two heads:
  - **coord score** `s_θ [N,3]` — an **equivariant `1o` vector** per atom
    (∇ log p over Cartesian positions); read out via an equivariant `1o` linear
    on node features (not the invariant scalar readout).
  - **type logits** `[N, n_elements]` — invariant, via `ScalarReadout`-style head.

## 5. Diffusion processes
- **Coordinates**: denoising score matching with **wrapped-Gaussian** noise on
  fractional coords (periodic); geometric σ schedule `σ_max→σ_min`; sampling by
  **annealed Langevin dynamics**.
- **Atom types**: denoise toward the composition-consistent types (CDVAE-style
  type prediction + update; alt: discrete diffusion / D3PM — open).
- **Lattice / N / composition**: predicted once from `z` (global), coords/types
  denoised within that fixed cell.

## 6. Training loss
```
L = β·L_KL(q(z|x)‖N(0,I))          # VAE latent regularization
  + L_num + L_lattice + L_composition   # latent predictors
  + L_coord   (denoising score matching, periodic)
  + L_type    (atom-type denoising cross-entropy)
  + λ·L_prop  (optional: property prediction from z, for conditioning)
```
Uses **EMA** of weights (diffusion models rely on it) — deferred here from the
predictor, added for CDVAE.

## 7. Generation (ab-initio)
```
z ~ N(0,I)                          # or property-optimized z (§8)
N, lattice, composition = predictors(z)
init N atoms: types ~ composition, random fractional coords, in `lattice`
for σ in anneal(σ_max → σ_min):     # annealed Langevin
    rebuild noisy graph; coords += step·s_θ(coords, types, lattice, z, σ) + noise
    (periodically) types = update(type_logits(·))
return Structure(lattice, coords, types)  → screen with the predictor
```

## 8. Property-conditioned generation
- Train latent property heads `f_p(z)` on the labels (low ρ, high K/G/hardness/
  toughness, high Debye/κ).
- Generate by **gradient-ascending `z`** toward the target multi-objective score,
  then decode (CDVAE's latent-optimization route). Alt: classifier-free guidance.
- Density is analytic from the decoded structure; all properties re-verified by
  the independent predictor (the screen).

## 9. Reuse & new layers
- **Reused** from `phlogiston/layers`: embedding, radial, spherical, linear,
  interaction, gate (the encoder/decoder backbones).
- **New**: `layers/noise` (σ/t embedding), an **equivariant `1o` vector readout**
  for the coord score (the existing `ScalarReadout` is invariant/`0e` only).

## 10. Build plan (incremental, validated like the encoder)
1. `layers/noise` + equivariant vector readout (+ equivariance tests).
2. VAE encoder + latent predictors (shape/round-trip tests).
3. Score decoder (coord-score **equivariance** test: rotate → score rotates as `1o`).
4. `diffusion` (forward noising + a 1-step denoise sanity; loss decreases).
5. Assembly + a tiny end-to-end generate → produces a valid `pymatgen` Structure.

## 11. Open decisions
- Latent dim `d`; `N_max`.
- Lattice parameterization (lengths+angles vs matrix vs Niggli).
- Coordinate diffusion in fractional vs Cartesian (nearest-image handling).
- Atom-type generation: predict-and-update vs discrete diffusion (D3PM).
- Conditioning: latent gradient optimization vs classifier-free guidance.
