# Phlogiston

A crystal-structure ML framework for materials discovery. It pairs the **GNoME**
stable-material candidates with structural records from the **Materials Project**
into one combined training corpus, learns to *generate* new crystals, and screens
them against a target property profile — down to a durable, deduplicated shortlist
of feasible candidates.

## Design goal

The driving use case is a (fictional) structural material that must be **light
enough to enable flight**, **strong/tough enough to survive ~300 km/h impacts**,
and **heat-resistant enough to survive lava**. Those requirements map onto
concrete, learnable materials properties:

| Requirement | Property target | Source / derivation |
|---|---|---|
| Light (flight) | density ρ | analytic from structure |
| Survive high-speed impact | bulk `K`, shear `G`, Young's `E` | MP elasticity |
| Resistance to breaking | Vickers hardness (Chen), fracture toughness `K_IC` (Niu), Pugh ratio G/K | derived from K, G, V |
| Stability (is it real?) | formation energy, energy above hull | GNoME + MP labels |
| Survive lava (thermal) | Debye temperature, Slack κ, sound velocities | MP + derived |

Derived quantities live in `phlogiston/data/properties.py`.

## Pipeline at a glance

```
fetch (GNoME + MP)  ->  featurize  ->  train (predictor, synth, CDVAE)  ->  discover
                                                                              |
   generate (CDVAE, optionally property-conditioned)                         |
     -> screen (property + stability predictors)                             |
     -> dedup + novelty (vs GNoME/MP)                                        |
     -> Tier-0 feasibility (composition rules)                               |
     -> Tier-1 synthesizability (learned classifier)                         |
     -> multi-objective rank + Pareto front                                  |
     -> persist to candidates.csv + CIFs  <-------------------------------- ─┘
```

## Modeling stack

**Philosophy: implemented from scratch.** The architectures are written in this
repo — equivariant interaction blocks, message passing, readout heads, and the
generator. We reuse only *low-level* primitives (equivariant math, graph/dataset
utilities), never a prebuilt model or high-level layer. The from-scratch
equivariant primitives live in `phlogiston/layers/` (embedding, spherical
harmonics, radial bases, equivariant linears, interaction/gate blocks, scalar
and vector readouts, diffusion noise embedding).

Four model roles, all sharing the same equivariant `CrystalEncoder`:

1. **Property predictor** (`phlogiston/models/predictor/`) — a shared equivariant
   encoder with multi-task readout heads for the targets above (moduli, hardness,
   toughness, Debye, κ). Trained under masked multi-task losses on the
   unevenly-labeled corpus. Equivariance matters because elastic response is
   tensorial.
2. **Stability gate** — the predictor's stability heads (`energy_above_hull`,
   formation energy). In practice a dedicated Stage-1 checkpoint is used for the
   gate, since property fine-tuning can erode stability accuracy; pass it via
   `discover --stability-ckpt`.
3. **Generator** (`phlogiston/models/cdvae/`) — a **CDVAE**: a VAE encoder over
   lattice/coords/types plus a noise-conditioned equivariant score decoder,
   sampled with GPU-batched annealed Langevin dynamics. Trained *unconditionally*;
   property conditioning is applied post-hoc (see below).
4. **Synthesizability classifier** (`phlogiston/models/synth/`) — the Tier-1
   feasibility model: the shared encoder + a single sigmoid head estimating
   P(experimentally synthesizable), trained positive-unlabeled.

### Property-conditioned generation

Rather than retraining the generator per target, conditioning is done in latent
space (`phlogiston/models/cdvae/conditioning.py`):

1. `fit-latent-head` fits a small property head `f_p(z)` on the frozen CDVAE
   latents.
2. `optimize_latent` gradient-ascends `z` toward a signed property profile
   (maximize the mechanical/thermal targets, minimize `energy_above_hull`).
3. A **trust region** clips `‖z − z₀‖` to `--cond-trust-radius` (default 8) each
   step, keeping the latent on the learned manifold, and a **saturating
   (`tanh`) reward** caps each objective so the optimizer can't chase adversarial
   extrapolations of the head.
4. The optimized latents are decoded, and every candidate is still independently
   re-verified by the property/stability screen.

### Runtime / primitives (verified on AMD ROCm GPUs)

| Layer | Choice |
|---|---|
| Tensor engine | PyTorch (ROCm build) |
| Equivariant math | `e3nn` (spherical harmonics, Clebsch–Gordan tensor products) |
| Graph containers / scatter | `torch-geometric` (pure-torch parts only; no `pyg-lib`) |
| Structures / graph construction | `pymatgen` |

> PyG's compiled ops (`radius_graph`, `pyg-lib`, `torch-scatter`) are
> intentionally avoided — they have no ROCm build. Graphs are constructed with
> pymatgen (`phlogiston/data/graph.py`), featurization is precomputed on CPU into
> sharded caches, and aggregation uses native `torch` scatter, so nothing in the
> model path depends on a CUDA-only extension.

## Feasibility: is the fictional material actually makeable?

Stability and hitting the property profile don't mean a structure can be
synthesized. Feasibility is screened in cheap-to-expensive tiers so that costly
checks only ever see survivors:

| Tier | What | Where | Cost |
|---|---|---|---|
| **Tier 0** | Rule-based composition sanity: rejects radioactive elements, noble gases, too many distinct elements, oversized stoichiometries; soft-scores charge-balanceability (pymatgen oxidation guesses) + optional SMACT validity | `phlogiston/discovery/feasibility.py` | cheap |
| **Tier 1** | Learned synthesizability classifier — P(makeable) trained on MP experimental provenance (ICSD / non-theoretical) as positives vs GNoME/theoretical as unlabeled | `phlogiston/models/synth/` | ML inference |
| **Tier 2** | DFT verification (relaxation, phonons) | *deferred* — the candidate registry is the durable shortlist to feed it | expensive |

The Tier-1 gate (`--synth-min`, default 0.3) is deliberately loose: the model
reflects *today's* synthesis record, so a low bar admits borderline candidates
that near-future methods could reach, while the score is always kept for ranking.

## Ranking & the candidate registry

`phlogiston/discovery/rank.py` gates survivors on stability
(`e_hull <= --e-hull-max`) and an optional density ceiling (`--rho-max`), then
scores them on competing higher-is-better objectives — specific stiffness
`(K+G)/2 / ρ`, fracture toughness, Vickers hardness, Debye temperature, and
Slack κ — as a min-max-normalized weighted sum (`score`, 0–1 within the pool).
It also flags the non-dominated **Pareto front**.

With `discover --save-dir DIR`, survivors are persisted durably:

- `DIR/cifs/*.cif` — one CIF per candidate
- `DIR/candidates.csv` — an accumulating, `StructureMatcher`-deduplicated
  registry with formula, run id, timestamp, score, Pareto flag, feasibility,
  synthesizability, and all key properties

`show-candidates --save-dir DIR` pretty-prints that registry as an aligned table.

## Datasets

Phlogiston trains on two sources of crystal structures + energy labels:

- **GNoME** (Graph Networks for Materials Exploration) — ~381k novel, predicted
  stable structures. Hosted in the public Google Cloud bucket
  `gs://gdm_materials_discovery` and pulled over plain HTTPS (no auth required).
- **Materials Project (MP)** — trusted, known-material structures + properties,
  fetched via the `mp-api` client. Requires an API key (get one at
  https://materialsproject.org, Dashboard -> API), read from the `MP54AC`,
  `MP_API_KEY`, or `MP_API_TOKEN` environment variable.

Both are stored under `data/raw/` (git-ignored):

```
data/raw/
  gnome/
    gnome_data/stable_materials_summary.csv   # ~554k rows: compositions + energies
    gnome_data/by_id.zip                       # CIFs keyed by MaterialId
    external_data/mp_snapshot_summary.csv
  mp/
    mp_metadata.csv                            # labels: formation energy, e_above_hull, ...
    mp_synth.csv                               # experimental provenance (theoretical / has_icsd)
    cifs/<material_id>.cif                     # one CIF per structure
```

Label coverage is intentionally uneven — every structure has stability info,
but only a subset (~12k MP materials) has measured elastic moduli. The models
handle this with a shared encoder + per-property heads trained under masked
multi-task losses. `fetch-mp` runs in two phases and is **resumable**: it writes
`mp_metadata.csv` first, then downloads structures in batches, skipping CIFs
already on disk and retrying transient API errors.

## Installation

```bash
pip install -r requirements.txt   # includes e3nn + torch-geometric (not in pyproject deps)
pip install -e .                  # installs the `phlogiston` CLI (Python >= 3.10)
```

For AMD ROCm boxes, `docker/deploy_gbt.sh` syncs the repo to a GPU host and
builds the ROCm image there (`sync` / `build` / `run` / `all`).

## Usage

The `phlogiston` CLI (or `python -m phlogiston.cli`) exposes the full pipeline.
A `--data-root` global flag sets the dataset root (default `./data`).

```bash
# 1. Fetch data
phlogiston fetch-gnome                                   # GNoME summary + MP snapshot
phlogiston fetch-gnome --keys structures_by_id           # GNoME CIFs by MaterialId
phlogiston gnome-info                                    # load summary, print stats
phlogiston fetch-mp --max-energy-above-hull 0.05 --exclude-radioactive
phlogiston fetch-mp-elasticity                           # elastic + derived labels
phlogiston fetch-mp-synth                                # Tier-1 provenance flags
phlogiston datasets-summary                              # label coverage across sets

# 2. Featurize the corpus (CPU, sharded cache)
phlogiston featurize

# 3. Train the models
phlogiston train --stage 1                               # predictor: stability
phlogiston train --stage 2 --init-ckpt runs/predictor_stage1_best.pt   # properties
phlogiston train-synth --init-ckpt runs/predictor_stage1_best.pt       # Tier-1
phlogiston train-cdvae                                   # generator (EMA + composite loss)
phlogiston evaluate --ckpt runs/predictor_stage2_best.pt # MAE + R2 + stability AUC/AP
phlogiston fit-latent-head --generator runs/cdvae_best.pt --out runs/latent_head.pt

# 4. Discover, screen, and persist candidates
phlogiston discover \
  --generator runs/cdvae_best.pt \
  --predictor runs/predictor_stage2_best.pt \
  --stability-ckpt runs/predictor_stage1_best.pt \
  --synth-ckpt runs/synth_best.pt \
  --latent-head runs/latent_head.pt \
  --n-samples 256 --e-hull-max 0.1 --save-dir runs/candidates

# 5. View the standing shortlist any time
phlogiston show-candidates --save-dir runs/candidates
```

Multi-GPU training uses `torchrun`, e.g.
`torchrun --nproc_per_node=4 -m phlogiston.cli train --stage 2 ...`.

### CLI commands

| Command | Purpose |
|---|---|
| `fetch-gnome` | Download GNoME dataset files |
| `gnome-info` | Load the summary and print statistics |
| `fetch-mp` | Download Materials Project structures + labels |
| `fetch-mp-elasticity` | Fetch MP elastic constants + derive mechanical/thermal labels |
| `fetch-mp-synth` | Fetch MP experimental-provenance flags (theoretical/ICSD) for Tier-1 |
| `featurize` | Precompute crystal graphs for the whole corpus (CPU, sharded) |
| `datasets-summary` | Print label coverage across GNoME + MP datasets |
| `train` | Train the predictor (schedule B: stage 1 stability, stage 2 properties) |
| `train-synth` | Train the Tier-1 synthesizability classifier |
| `evaluate` | Score a checkpoint (MAE + R2 + stability AUC/AP) |
| `train-cdvae` | Train the CDVAE generator (EMA + composite loss) |
| `discover` | Generate -> screen -> rank novel stable candidates |
| `show-candidates` | Pretty-print the saved candidate registry |
| `fit-latent-head` | Fit f_p(z) on a CDVAE for property conditioning |

### Loading in Python

```python
from phlogiston.data import gnome, materials_project as mp

df = gnome.load_summary("data")                 # GNoME summary DataFrame
stable = gnome.filter_stable(df, 0.0)           # rows on/below the convex hull
cif = gnome.read_structure_cif("data", material_id="000006a8c4")

meta = mp.load_metadata("data")                 # MP labels
structure = mp.load_structure("data", "mp-862690")   # pymatgen Structure
```

## Project layout

```
phlogiston/
  cli.py                    # `phlogiston` command-line interface
  config.py                 # dataclass configs (YAML-serializable)
  data/
    gnome.py                # GNoME acquisition + loading
    materials_project.py    # MP structures, elasticity, synth provenance
    properties.py           # derived mechanical/thermal targets
    graph.py                # pymatgen Structure -> lossless CrystalGraph
    dataset.py              # sharded dataset + masked multi-task collation
    precompute.py           # CPU-parallel corpus featurization -> shards
    synth.py                # Tier-1 synthesizability labels + dataset
  layers/                   # from-scratch equivariant primitives (see layers/README.md)
  models/
    encoder/                # shared E(3)-equivariant CrystalEncoder
    predictor/              # multi-task property + stability heads
    synth/                  # Tier-1 synthesizability classifier
    cdvae/                  # generator: VAE + score decoder + sampler + conditioning
  train/                    # predictor / synth / cdvae trainers + EMA
  discovery/
    loop.py                 # end-to-end discover() orchestrator + registry export
    screen.py               # featurize + score with predictor/stability/synth
    feasibility.py          # Tier-0 composition rules
    novelty.py              # dedup + novelty vs GNoME/MP
    rank.py                 # multi-objective score + Pareto front
scripts/                    # cond_compare, sweep, validate_graph diagnostics
tests/                      # layer / model / discovery unit tests
docker/deploy_gbt.sh        # rsync + ROCm Docker build on a GPU box
```
