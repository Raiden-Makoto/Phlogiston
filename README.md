# Phlogiston

A crystal-structure ML framework for materials discovery. It pairs the **GNoME**
stable-material candidates with structural records from the **Materials Project**
into one combined training corpus, then learns to propose *new* crystals that
hit a target property profile.

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
| Stability (is it real?) | formation energy, energy above hull | GNoME + MP + uMLIP checker |
| Survive lava (thermal) | Debye temperature, Slack κ, sound velocities | MP + derived |

Derived quantities live in `phlogiston/data/properties.py`.

## Modeling stack

**Philosophy: implemented from scratch.** The architectures are written in this
repo — interaction blocks, message passing, readout heads, and the generator.
We reuse only *low-level* primitives (equivariant math, graph/dataset utilities),
never a prebuilt model or high-level layer.

Three model roles:

1. **Stability checker** — a MACE-style equivariant universal interatomic
   potential (uMLIP). Relaxes a candidate and estimates energy above the convex
   hull; the gate for physical realizability.
2. **Property predictor** — a shared equivariant encoder with multi-task readout
   heads for the targets above (moduli, hardness, toughness, Debye, κ). Equivariance
   matters because the elastic response is tensorial.
3. **Generator** — a property-conditioned diffusion model over lattice, fractional
   coordinates, and atom types, to sample new candidates given a target profile.

**Featurization** (built offline, no CUDA-only ops):

- Periodic crystal graph via **pymatgen** (CrystalNN bonding / radius cutoff).
- Node features: element descriptors (electronegativity, radii, valence, group/period, mass).
- Edge features: relative position vectors + radial basis expansion + spherical harmonics.
- Message aggregation with native `torch` scatter.

**Runtime / primitives** (verified on AMD ROCm GPUs):

| Layer | Choice |
|---|---|
| Tensor engine | PyTorch (ROCm build) |
| Equivariant math | `e3nn` (spherical harmonics, Clebsch–Gordan tensor products) |
| Graph containers / scatter | `torch-geometric` (pure-torch parts only; no `pyg-lib`) |
| Structures / graph construction | `pymatgen` |

> Note: PyG's compiled ops (`radius_graph`, `pyg-lib`, `torch-scatter`) are
> intentionally avoided — they have no ROCm build. Graphs are constructed with
> pymatgen and aggregation uses native `torch` scatter, so nothing in the model
> path depends on a CUDA-only extension.

## Datasets

Phlogiston trains on two sources of crystal structures + energy labels:

- **GNoME** (Graph Networks for Materials Exploration) — ~381k novel, predicted
  stable structures. Hosted in the public Google Cloud bucket
  `gs://gdm_materials_discovery` and pulled over plain HTTPS (no auth required).
- **Materials Project (MP)** — trusted, known-material structures + properties,
  fetched via the `mp-api` client. Requires an API key (get one at
  https://materialsproject.org, Dashboard -> API), read from the `MP54AC` or
  `MP_API_KEY` environment variable.

Both are stored under `data/raw/` (git-ignored):

```
data/raw/
  gnome/
    gnome_data/stable_materials_summary.csv   # ~554k rows: compositions + energies
    gnome_data/by_id.zip                       # CIFs keyed by MaterialId
    external_data/mp_snapshot_summary.csv
  mp/
    mp_metadata.csv                            # labels: formation energy, e_above_hull, ...
    cifs/<material_id>.cif                     # one CIF per structure
```

### Fetching the data

```bash
# GNoME: summary + MP snapshot (default), or structure CIFs, or everything
phlogiston fetch-gnome                                   # default files
phlogiston fetch-gnome --keys structures_by_id           # CIFs keyed by MaterialId
phlogiston gnome-info                                    # load summary, print stats

# Materials Project: stable + near-stable, non-radioactive structures
phlogiston fetch-mp --max-energy-above-hull 0.05 --exclude-radioactive
```

`fetch-mp` runs in two phases and is **resumable**: it writes `mp_metadata.csv`
first, then downloads structures in batches, skipping CIFs already on disk and
retrying transient API errors.

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
  config.py                 # dataclass configs (YAML-serializable)
  data/
    gnome.py                # GNoME acquisition + loading
    materials_project.py    # Materials Project structure fetcher
    properties.py           # derived mechanical/thermal targets
  cli.py                    # `phlogiston` command-line interface
```

Planned modules (from scratch): `data/graph.py` (featurizer),
`models/` (equivariant encoder + stability/property heads), `generator/`
(property-conditioned diffusion), and training/discovery drivers.
