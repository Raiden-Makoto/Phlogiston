"""Materials Project structural records (Phase 2).

Fetches crystal structures + energy labels from the Materials Project via the
official ``mp-api`` client. These trusted, known-material structures form the
reference half of the combined GNoME + MP training corpus.

The API key is read from the environment (``MP54AC`` or ``MP_API_KEY``). Get a
key at https://materialsproject.org (Dashboard -> API).

Storage layout (mirrors the GNoME raw layout)::

    data/raw/mp/
      mp_metadata.csv          # one row per material (ids, energies, symmetry)
      cifs/<material_id>.cif    # one CIF per structure
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm import tqdm

API_KEY_ENV_VARS = ("MP54AC", "MP_API_KEY", "MP_API_TOKEN")

# Radioactive elements to screen out of the training corpus: those with no
# stable isotopes (Tc, Pm) plus Po..Pu (Z 84-94). Transplutonium/superheavy
# elements are omitted -- they never appear in Materials Project and their
# symbols are rejected by the API's element validation.
RADIOACTIVE_ELEMENTS: tuple[str, ...] = (
    "Tc", "Pm",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
)

# Summary fields we pull. `structure` carries the pymatgen Structure; the rest
# are training targets / metadata.
DEFAULT_FIELDS: tuple[str, ...] = (
    "material_id",
    "formula_pretty",
    "structure",
    "nsites",
    "elements",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "band_gap",
    "symmetry",
)

# Columns persisted to the metadata CSV (everything except the heavy structure).
_METADATA_COLUMNS: tuple[str, ...] = (
    "material_id",
    "formula_pretty",
    "nsites",
    "elements",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "band_gap",
    "spacegroup_number",
    "crystal_system",
)


def get_api_key(explicit: str | None = None) -> str:
    """Resolve the MP API key from an explicit value or known env vars."""
    if explicit:
        return explicit
    for var in API_KEY_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    raise RuntimeError(
        "No Materials Project API key found. Set one of "
        f"{API_KEY_ENV_VARS} (e.g. `export MP54AC=...`)."
    )


def raw_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "raw" / "mp"


def cif_dir(data_root: str | Path) -> Path:
    return raw_dir(data_root) / "cifs"


def metadata_path(data_root: str | Path) -> Path:
    return raw_dir(data_root) / "mp_metadata.csv"


def _resolve_chunking(limit: int | None, chunk_size: int) -> tuple[int | None, int]:
    """Translate a total ``limit`` into (num_chunks, chunk_size) for mp-api."""
    if limit is None:
        return None, chunk_size
    chunk_size = min(chunk_size, limit)
    num_chunks = max(1, -(-limit // chunk_size))  # ceil div
    return num_chunks, chunk_size


def fetch_structures(
    data_root: str | Path = "data",
    *,
    api_key: str | None = None,
    elements: Sequence[str] | None = None,
    exclude_elements: Sequence[str] | None = None,
    num_elements: tuple[int, int] | None = None,
    num_sites_max: int | None = None,
    is_stable: bool | None = None,
    max_energy_above_hull: float | None = None,
    limit: int | None = None,
    chunk_size: int = 1000,
    fields: Iterable[str] = DEFAULT_FIELDS,
    save_cif: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Download MP structures + labels and persist CIFs + a metadata table.

    Parameters
    ----------
    elements: restrict to materials containing these elements (e.g. ["Li","O"]).
    exclude_elements: drop materials containing any of these elements (e.g.
        RADIOACTIVE_ELEMENTS to skip radioactive chemistries).
    num_elements: (min, max) number of distinct elements.
    num_sites_max: cap on sites per unit cell (keeps graphs tractable).
    is_stable: filter to stable (True) / unstable (False) materials.
    max_energy_above_hull: keep materials with 0 <= e_above_hull <= this value
        (eV/atom); e.g. 0.05 selects stable + near-stable (metastable) materials.
    limit: max number of materials to fetch (None = all matching).

    Returns the metadata DataFrame (also written to ``mp_metadata.csv``).
    """
    # Imported lazily so the package imports without mp-api present.
    from mp_api.client import MPRester

    key = get_api_key(api_key)
    cifs = cif_dir(data_root)
    cifs.mkdir(parents=True, exist_ok=True)

    search_kwargs: dict = {"fields": list(fields)}
    if elements is not None:
        search_kwargs["elements"] = list(elements)
    if exclude_elements is not None:
        search_kwargs["exclude_elements"] = list(exclude_elements)
    if num_elements is not None:
        search_kwargs["num_elements"] = tuple(num_elements)
    if num_sites_max is not None:
        search_kwargs["num_sites"] = (1, num_sites_max)
    if is_stable is not None:
        search_kwargs["is_stable"] = is_stable
    if max_energy_above_hull is not None:
        search_kwargs["energy_above_hull"] = (0.0, max_energy_above_hull)
    num_chunks, chunk = _resolve_chunking(limit, chunk_size)
    if num_chunks is not None:
        search_kwargs["num_chunks"] = num_chunks
    search_kwargs["chunk_size"] = chunk

    print(f"[mp] querying Materials Project (limit={limit}) ...")
    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(**search_kwargs)

    rows: list[dict] = []
    for doc in tqdm(docs, desc="[mp] structures"):
        mid = str(doc.material_id)
        symmetry = getattr(doc, "symmetry", None)
        rows.append(
            {
                "material_id": mid,
                "formula_pretty": getattr(doc, "formula_pretty", None),
                "nsites": getattr(doc, "nsites", None),
                "elements": [str(e) for e in (getattr(doc, "elements", None) or [])],
                "formation_energy_per_atom": getattr(doc, "formation_energy_per_atom", None),
                "energy_above_hull": getattr(doc, "energy_above_hull", None),
                "is_stable": getattr(doc, "is_stable", None),
                "band_gap": getattr(doc, "band_gap", None),
                "spacegroup_number": getattr(symmetry, "number", None) if symmetry else None,
                "crystal_system": str(getattr(symmetry, "crystal_system", "")) or None,
            }
        )
        if save_cif and getattr(doc, "structure", None) is not None:
            dest = cifs / f"{mid}.cif"
            if force or not dest.exists():
                doc.structure.to(filename=str(dest), fmt="cif")

    df = pd.DataFrame(rows, columns=list(_METADATA_COLUMNS))
    meta = metadata_path(data_root)
    df.to_csv(meta, index=False)
    print(f"[mp] wrote {len(df):,} rows -> {meta}")
    if save_cif:
        print(f"[mp] wrote {len(list(cifs.glob('*.cif'))):,} CIFs -> {cifs}")
    return df


def load_metadata(data_root: str | Path = "data") -> pd.DataFrame:
    """Load the persisted MP metadata table."""
    meta = metadata_path(data_root)
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found; run fetch_structures() first.")
    return pd.read_csv(meta)


def load_structure(data_root: str | Path, material_id: str):
    """Load a single MP structure as a pymatgen ``Structure`` from its CIF."""
    from pymatgen.core import Structure

    path = cif_dir(data_root) / f"{material_id}.cif"
    if not path.exists():
        raise FileNotFoundError(f"No CIF for {material_id} at {path}")
    return Structure.from_file(str(path))
