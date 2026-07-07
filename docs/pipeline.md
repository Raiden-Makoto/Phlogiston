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
        │  property-conditioned diffusion over lattice + coords + species   │
        └───────────────────────────┬──────────────────────────────────────┘
                                     ▼
        ┌──────────────────── DISCOVERY LOOP (Phase 6) ────────────────────┐
        │  sample candidates → stability gate → property screen → rank      │
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

> Note on relaxation: our labels are per-structure scalar energies (no DFT
> forces), so the stability model is an **energy/e_above_hull regressor**, not a
> force field. It gates candidates by *predicting* hull distance directly.
> Force-based relaxation (optional, later) would require force labels or
> initializing from a pretrained potential — out of scope for the from-scratch v1.

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

## 6. Generator (Phase 5) — separate, from scratch

Property-conditioned diffusion over crystal structure (CDVAE / DiffCSP /
MatterGen family, implemented ourselves):

- **State**: lattice (6 params), fractional coordinates (periodic/wrapped), atom
  types (categorical), variable N.
- **Forward process**: noise coords (wrapped Gaussian), lattice, and types.
- **Denoiser**: an equivariant network (reuses our encoder-style blocks) that
  predicts scores/noise, **conditioned** on a target property vector
  (low ρ, high K/G/hardness/toughness, high Debye/κ) via embeddings.
- **Training**: denoising objective on the stable-structure corpus (MP + GNoME).
- Composition/charge/symmetry sanity handled via sampling constraints.

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
  data/            # DONE: gnome, materials_project, properties, graph, dataset, precompute
  models/
    encoder.py     # shared equivariant encoder (embeddings, interaction blocks)
    heads.py       # stability + property readout heads
    predictor.py   # encoder + heads assembly, masked multi-task loss
  train/
    stage1.py      # pretrain encoder + stability
    stage2.py      # attach + fine-tune property heads
    parallel.py    # TP2/TP4 setup
  generator/
    diffusion.py   # forward/reverse process
    denoiser.py    # conditioned equivariant denoiser
  discovery/
    screen.py      # stability gate + property scoring
    rank.py        # multi-objective ranking
```

---

## 10. Milestones

- [x] Data foundation + labels
- [x] Featurizer + validation
- [x] Full precompute (629k graphs)
- [ ] **Phase 4**: predictor model + schedule-B training (this is next)
- [ ] Phase 5: generator
- [ ] Phase 6: discovery loop + reporting

---

## 11. Open decisions

- Encoder size: `L_max` (2 vs 3), hidden irreps, N_layers (accuracy vs cost).
- Stage-2: low-LR fine-tune vs partial encoder freeze (decide from val curves).
- Generator family: DiffCSP-style vs MatterGen-style conditioning.
- Discovery: exact multi-objective scoring / density ceiling for "flight".
