# Phlogiston — Pipeline Design

End-to-end design for discovering a novel structural material that is
**light** (flight), **strong/tough** (survive ~300 km/h impacts), and
**heat-resistant** (survive lava). This document is the plan; it is intentionally
code-free.

---

## 0. Design goal → learnable targets

| Requirement | Property target | Source |
|---|---|---|
| Light (flight) | density ρ | analytic from structure |
| Withstand impact | bulk `K`, shear `G`, Young's `E` | MP elasticity |
| Resistance to breaking | Vickers hardness, fracture toughness `K_IC`, Pugh ratio | derived (K, G, V) |
| Stability (is it real?) | formation energy, energy above hull | GNoME + MP |
| Survive lava (thermal) | Debye temperature, Slack κ | MP + derived |

Fixed target vector (`TARGET_KEYS`), assembled per material with a **mask**
(no material has all labels):

```
[ formation_energy_per_atom, energy_above_hull, density,
  bulk_modulus_vrh, shear_modulus_vrh, vickers_hardness,
  fracture_toughness, debye_temperature, slack_thermal_conductivity ]
```

---

## 1. Pipeline at a glance

```
                 ┌─────────────────────── DATA (done) ───────────────────────┐
  GNoME 554k ────┤ summary + by_id CIFs                                       │
  MP 71,939  ────┤ near-stable structures + stability labels                  │
  MP 12,246  ────┤ elasticity structures + mechanical/thermal labels          │
                 └───────────────────────────┬───────────────────────────────┘
                                              ▼
                        FEATURIZE (done): 629,376 lossless crystal graphs
                                              ▼
        ┌──────────────────── PREDICTOR MODEL (Phase 4) ───────────────────┐
        │  shared equivariant encoder                                       │
        │      ├─ stage-1 heads:  formation energy, energy_above_hull       │
        │      └─ stage-2 heads:  K, G, hardness, toughness, Debye, κ       │
        └───────────────────────────┬──────────────────────────────────────┘
                                     │  (density is analytic, no head)
                                     ▼
        ┌──────────────────── GENERATOR MODEL (Phase 5) ───────────────────┐
        │  CDVAE (ab-initio): latent z → composition + lattice, then        │
        │  score-based denoising of coordinates + atom types                │
        └───────────────────────────┬──────────────────────────────────────┘
                                     ▼
        ┌──────────────────── DISCOVERY LOOP (Phase 6) ────────────────────┐
        │  sample → screen → dedup/novelty → Tier-0/1 feasibility → rank     │
        └───────────────────────────┬──────────────────────────────────────┘
                                     ▼
        ┌──────────────────── VERIFY / Tier 2 (Phase 7) ───────────────────┐
        │  ensemble uMLIP: relax → hull → confidence → phonon gate → write  │
        │  (independent physics check; see phlogiston/verify/DESIGN.md)     │
        └───────────────────────────────────────────────────────────────────┘
```

Two trained models: **(a) predictor** = shared encoder + heads (stability +
properties), **(b) generator** = separate diffusion model. They connect at
discovery time by usage, not weights.

---

## 2. Data foundation — DONE

- Sources fetched, labeled, and inspectable via `phlogiston datasets-summary`.
- **629,376 precomputed graphs** in `data/processed/` (155 shards, ~30 GB),
  loaded by `ShardedCrystalDataset`.
- Label coverage is uneven by design; handled by masked multi-task losses.

---

## 3. Featurization — DONE

Lossless periodic crystal graph per structure (`phlogiston/data/graph.py`):

- **Nodes**: atomic numbers `z` (int), Cartesian positions.
- **Edges** (radius cutoff 6.0 Å, periodic images): `edge_index` (i←j),
  `edge_vec` = r_j(image) − r_i, `edge_len`.
- Validated: coordination numbers, `‖edge_vec‖`=distance, translation
  invariance, rotation equivariance, supercell periodicity, batching/masks.
- Learned features are computed **in the model**, not stored (so feature changes
  never require re-preprocessing; 6.0 Å is the max usable `r_max`).

---

## 4. Predictor model (Phase 4) — from scratch

MACE-style E(3)-equivariant message-passing network. We implement the blocks;
only low-level equivariant math (`e3nn`: spherical harmonics, Clebsch–Gordan
tensor products) and native `torch` scatter are reused.

### 4.1 Inputs (per batched graph)
`z [N]`, `edge_index [2,E]`, `edge_vec [E,3]`, `edge_len [E]`, `batch [N]`.

### 4.2 Embeddings
- **Node**: learnable embedding table indexed by `z` → scalar (`0e`) features,
  optionally seeded with element descriptors (electronegativity, radius, group,
  period, valence, mass).
- **Edge radial**: Bessel radial basis of `edge_len` × smooth polynomial cutoff
  envelope (→ 0 at `r_max`), giving continuous, differentiable edge weights.
- **Edge angular**: real spherical harmonics `Y_l^m(edge_vec)` up to `L_max`
  (e.g. 2–3) → the equivariant geometric signal.

### 4.3 Interaction blocks (× N_layers, e.g. 3–4)
Per layer, per edge i←j:
1. Message = Clebsch–Gordan tensor product of neighbor features `h_j` with the
   edge spherical harmonics, weighted by an MLP of the radial basis.
2. Aggregate messages into node i via native `torch` scatter (sum).
3. Equivariant linear + gated nonlinearity (scalars gate higher-`l` channels);
   residual update of node features.
Node features carry mixed irreps (`0e + 1o + 2e + …`); equivariance is exact.

### 4.4 Readouts (heads)
- **Per-atom scalar** (`0e`) → head MLPs. Graph-level = scatter-mean/sum over
  `batch`.
- **Stability heads** (stage 1): `formation_energy_per_atom`, `energy_above_hull`.
- **Property heads** (stage 2): `K`, `G`, hardness, toughness, Debye, κ.
- **Density**: analytic (mass/volume) — not a head.

> Note on relaxation: our labels are per-structure scalar energies (no forces),
> so the stability model is an **energy/e_above_hull regressor**, not a force
> field. It gates candidates by *predicting* hull distance directly — a fast,
> in-loop screen. Force-based relaxation is done later and independently by the
> Tier-2 ensemble uMLIP layer (§7.5), not by this predictor.

### 4.5 Targets & normalization
- Each target standardized (z-score) using train-set statistics; predictions
  de-normalized at inference.
- Extensive quantities (energy) handled per-atom; intensive (moduli, ρ) as-is.

---

## 5. Training (Phase 4) — schedule B (pretrain → fine-tune)

### Stage 1 — pretrain encoder + stability heads (all 629k)
- Objective: masked regression on `formation_energy_per_atom` and
  `energy_above_hull` (both present for ~all materials).
- Split: train/val/test by material (no leakage); consider holding out whole
  chemistries for OOD checks.
- Output: a strong general encoder + a stability predictor.

### Stage 2 — attach property heads, fine-tune (12,246 labeled)
- Freeze nothing hard: fine-tune encoder at a **low LR**, property heads at a
  normal LR. Masked multi-task loss over the 6 mechanical/thermal targets.
- Guards against overfitting the small set: weight decay, early stopping on a
  held-out property val split, optional light encoder freezing if needed.

### Common
- Loss: per-target Huber/MSE, summed over masked-in targets, per-target weights.
- Optimizer: AdamW + cosine schedule + warmup; EMA of weights.
- **Parallelism: target TP2 (2 GPUs), max TP4** — never spread across >4 GPUs
  (shared box). Data-parallel within the TP group; large batches (288 GB HBM).
- Metrics: MAE per property, energy-above-hull classification (stable vs not) at
  thresholds; parity plots; per-chemistry error.
- Checkpoints + configs versioned; runs reproducible.

---

## 6. Generator (Phase 5) — CDVAE, from scratch

**Why CDVAE over DiffCSP.** DiffCSP is crystal *structure prediction* — it needs
the composition (formula) as input and only generates the arrangement. CDVAE is
**ab-initio**: it samples composition + lattice + structure from scratch. Since
we are inventing an unknown fictional compound (no formula given), CDVAE is the
logical choice. We implement it ourselves, reusing the equivariant blocks from
§4 as backbones.

### 6.1 Components

1. **Encoder `E(x) → z`** — periodic equivariant GNN (our §4 encoder) pools a
   crystal graph to `μ, logσ²`; sample latent `z ∈ R^d` (VAE reparameterization).
2. **Latent predictors** (MLPs on `z`, aggregate/global properties):
   - `num_atoms`: distribution over `1..N_max`.
   - `lattice`: the cell (see 6.4 for parameterization).
   - `composition`: expected per-element proportions (softmax over the element
     set) → element counts.
   - *(optional)* `property head(s)`: our target properties from `z`, used for
     conditioning (6.5).
3. **Decoder / score network `D`** — noise-conditioned equivariant GNN. Given a
   *noisy* structure (coords + types) in the predicted lattice, `z`, and noise
   level `σ_t`, it outputs:
   - **coordinate score** `s_θ` — a per-atom **equivariant vector** (`1o`,
     shape `[N,3]`) = ∇ log p over positions (uses nearest-image periodicity).
   - **atom-type logits** — per-atom invariant distribution over elements.

### 6.2 Diffusion processes
- **Coordinates**: denoising score matching with wrapped-Gaussian noise on
  (fractional→Cartesian) positions; annealed Langevin sampling `σ_max → σ_min`.
- **Atom types**: denoising toward the composition-consistent types (CDVAE-style
  type prediction + update; alternative: discrete diffusion / D3PM — see §11).
- **Lattice, N, composition**: predicted from `z` (global), then coords/types
  are denoised *within* that fixed cell.

### 6.3 Training losses
```
L = L_KL(q(z|x)‖N(0,I))·β          # VAE latent regularization
  + L_num                          # num_atoms
  + L_lattice                      # cell parameters
  + L_composition                  # element distribution
  + L_coord   (denoising score matching, periodic)
  + L_type    (atom-type denoising CE)
  + λ·L_prop  (optional: property prediction from z, for conditioning)
```
Trained on the stable-structure corpus (MP + GNoME).

### 6.4 Ab-initio generation
```
z ~ N(0, I)                         # or property-optimized z (6.5)
N, lattice, composition = predictors(z)
init N atoms: types ~ composition, random fractional coords, in `lattice`
for σ in anneal(σ_max → σ_min):     # annealed Langevin
    coords += step·s_θ(coords, types, lattice, z, σ) + noise
    (periodically) types = update(type_logits(·, z, σ))
return Structure(lattice, coords, types)
```

### 6.5 Property-conditioned generation (our targets)
- Train latent property predictors `f_p(z)` on our labels (low ρ, high
  K/G/hardness/toughness, high Debye/κ).
- At generation, **optimize `z` by gradient ascent** toward the desired
  (multi-objective) property score, then decode — CDVAE's latent-optimization
  route. (Alternative: classifier-free guidance — §11.)
- Density is analytic from the decoded structure; all other properties are
  re-verified by the independent Phase-4 predictor as the screen (§7).

### 6.6 Reuse of §4 machinery
- Encoder = §4 equivariant encoder + VAE head.
- Decoder = §4 interaction blocks + **σ (noise-level) embedding** + an
  equivariant `1o` vector head (coord score) + invariant type-logit head.
- Same primitives (`e3nn` SH/CG, native scatter); separate weights.

---

## 7. Discovery loop (Phase 6)

```
for target profile P (light + strong + tough + heat-resistant):
    candidates = generator.sample(condition=P, n=large)
    candidates = dedup / novelty-filter vs GNoME + MP
    keep = stability_gate(candidates)          # predicted e_above_hull ≤ τ
    scored = property_model(keep)              # K, G, hardness, toughness, Debye, κ
    rank by multi-objective score:
        maximize (specific strength = strength/ρ, toughness, Debye/κ)
        subject to ρ ≤ ρ_max and predicted stability
    shortlist top candidates → report structures + predicted property card
```

- Multi-objective ranking (Pareto front) over the competing goals; density is a
  hard-ish constraint (flight), stability a gate, mechanics+thermal the score.
- Feasibility tiers screen synthesizability before physics: **Tier 0** rule-based
  composition checks (`discovery/feasibility.py`), **Tier 1** a learned
  synthesizability classifier (`models/synth/`). Survivors are persisted to a
  durable, deduplicated registry (`candidates.csv` + CIFs).

---

## 7.5 Verification / Tier 2 (Phase 7) — independent physics gate

Everything up to here is a *learned* judgement, and the discovery stability score
is **inside the generation loop** — the latent optimizer ascends the predictor's
own hull estimate, so candidates cluster where that predictor is optimistic
(grading our own homework). Tier 2 brings in models that never saw the loop: an
**ensemble of pretrained universal ML interatomic potentials (uMLIPs)** —
CHGNet (primary) and MatterSim (independent cross-check), both wired on ROCm.

- **Relax** each candidate to its true local minimum (the predictor scores the
  as-generated, unrelaxed cell — a structure that doesn't physically exist); the
  relaxed structure replaces the generated one, drift is recorded.
- **Refined hull**: uMLIPs are `MPtrj`-trained, so their energies are MP-frame
  comparable → a directly-usable `energy_above_hull`. The residual vs the
  predictor is a **bias meter** feeding back into the conditioning trust radius.
- **Ensemble confidence**: several independent potentials re-score the relaxed
  cell; their **disagreement** flags off-distribution (exotic) candidates as
  low-confidence for manual review — the built-in "don't trust me here" detector.
- **Dynamical stability**: finite-displacement phonons (near-hull survivors only)
  reject imaginary-mode structures — something a scalar-energy predictor *cannot*
  compute (it needs forces).

The ensemble is the whole verification method: it runs entirely on our ROCm GPUs,
is hull-comparable by construction, is cheap enough to apply to the full registry,
and turns foundation potentials' one real weakness — unreliability
off-distribution — into an explicit per-candidate confidence estimate.

Full design: `phlogiston/verify/DESIGN.md`.

---

## 8. Evaluation & trust

- **Predictor**: held-out MAE per property vs MP; stability AUC/threshold; OOD
  chemistry holdout; sanity anchors (Pd, diamond, etc.).
- **Generator**: validity (physical distances, charge/symmetry), novelty vs
  training sets, uniqueness, and — crucially — **stability rate** of samples per
  the predictor.
- **End-to-end**: do shortlisted candidates actually hit the target profile
  under the predictor, and are they novel + stable?

---

## 9. Module layout (planned)

```
phlogiston/
  data/          # DONE: gnome, materials_project, properties, graph, dataset, precompute
  layers/        # reusable blocks — see layers/README.md
                 #   radial, spherical, linear, interaction, gate, readout, embedding, noise
  models/        # assembled models — see models/README.md
    encoder/     #   shared E(3)-equivariant encoder        (DESIGN.md)
    predictor/   #   encoder + stability & property heads   (DESIGN.md, schedule B)
    cdvae/       #   ab-initio generator (VAE + predictors + score decoder) (DESIGN.md)
    synth/       #   Tier-1 synthesizability classifier          (models/synth)
  train/         # stage1 (pretrain) / stage2 (fine-tune) / synth / cdvae
  discovery/     # screen + dedup/novelty + Tier-0/1 feasibility + rank + registry
  verify/        # Tier-2 uMLIP relax + hull + phonon gate         (DESIGN.md)
```

Detailed, per-component architecture lives in `layers/README.md` and
`models/*/DESIGN.md` — this file stays a high-level map.

---

## 10. Milestones

- [x] Data foundation + labels
- [x] Featurizer + validation
- [x] Full precompute (629k graphs)
- [x] Phase 4: predictor model + schedule-B training
- [x] Phase 5: CDVAE generator (+ latent-optimization conditioning)
- [x] Phase 6: discovery loop + Tier-0/1 feasibility + registry + reporting
- [ ] **Phase 7**: Tier-2 uMLIP verification (relax + hull + phonons) — this is next

---

## 11. Open decisions

- Encoder size: `L_max` (2 vs 3), hidden irreps, N_layers (accuracy vs cost).
- Stage-2: low-LR fine-tune vs partial encoder freeze (decide from val curves).
- CDVAE latent dim `d`; `N_max` (max atoms to generate).
- Lattice parameterization: lengths+angles vs full matrix vs Niggli-reduced.
- Coordinate diffusion in fractional vs Cartesian (nearest-image periodicity).
- Atom-type generation: CDVAE type-prediction+Langevin vs discrete diffusion (D3PM).
- Conditioning: latent gradient optimization vs classifier-free guidance.
- Discovery: exact multi-objective scoring / density ceiling for "flight".
- Tier-2 verification: uMLIP backend, self-consistent hull, phonon rigor,
  bias-feedback — see `phlogiston/verify/DESIGN.md` §11.
