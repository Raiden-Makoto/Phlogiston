# Phlogiston

A crystal-structure ML framework for materials discovery, pairing the **GNoME**
stable-material candidates with structural records from the **Materials Project**
as a combined training corpus.

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
  cli.py                    # `phlogiston` command-line interface
```
