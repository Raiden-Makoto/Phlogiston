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
import time
from pathlib import Path
from typing import Callable, Sequence

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


def _with_retries(fn: Callable, *, tries: int = 4, backoff: float = 3.0, what: str = "request"):
    """Call ``fn`` retrying on transient MP API / network errors."""
    from mp_api.client.core.exceptions import MPRestError

    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except (MPRestError, Exception) as exc:  # noqa: BLE001 - want broad transient catch
            last = exc
            if attempt == tries:
                break
            wait = backoff * attempt
            print(f"[mp] {what} failed (attempt {attempt}/{tries}): {exc}. Retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"[mp] {what} failed after {tries} attempts") from last


def _batches(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


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
    structure_batch: int = 500,
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
    structure_batch: how many structures to request per API call in phase 2.

    The fetch runs in two phases and is resumable: (1) pull lightweight metadata
    for all matching materials and write ``mp_metadata.csv`` immediately, then
    (2) download structures in small batches, writing CIFs incrementally and
    skipping any already on disk. Re-running resumes where it left off.

    Returns the metadata DataFrame.
    """
    # Imported lazily so the package imports without mp-api present.
    from mp_api.client import MPRester

    key = get_api_key(api_key)
    cifs = cif_dir(data_root)
    cifs.mkdir(parents=True, exist_ok=True)

    filters: dict = {}
    if elements is not None:
        filters["elements"] = list(elements)
    if exclude_elements is not None:
        filters["exclude_elements"] = list(exclude_elements)
    if num_elements is not None:
        filters["num_elements"] = tuple(num_elements)
    if num_sites_max is not None:
        filters["num_sites"] = (1, num_sites_max)
    if is_stable is not None:
        filters["is_stable"] = is_stable
    if max_energy_above_hull is not None:
        filters["energy_above_hull"] = (0.0, max_energy_above_hull)

    meta_fields = [f for f in DEFAULT_FIELDS if f != "structure"]

    with MPRester(key) as mpr:
        # --- Phase 1: metadata (fast, no structures) --------------------
        print(f"[mp] phase 1/2: querying metadata ({filters}) ...")
        docs = _with_retries(
            lambda: mpr.materials.summary.search(fields=meta_fields, **filters),
            what="metadata search",
        )
        rows: list[dict] = []
        for doc in docs:
            symmetry = getattr(doc, "symmetry", None)
            rows.append(
                {
                    "material_id": str(doc.material_id),
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

        df = pd.DataFrame(rows, columns=list(_METADATA_COLUMNS))
        if limit is not None:
            df = df.head(limit)
        meta = metadata_path(data_root)
        df.to_csv(meta, index=False)
        print(f"[mp] phase 1 done: {len(df):,} rows -> {meta}")

        if not save_cif:
            return df

        # --- Phase 2: structures (batched, resumable) -------------------
        material_ids = [m for m in df["material_id"].tolist()
                        if force or not (cifs / f"{m}.cif").exists()]
        have = len(df) - len(material_ids)
        print(f"[mp] phase 2/2: {len(material_ids):,} structures to fetch "
              f"({have:,} already on disk), batch={structure_batch}")

        written = 0
        with tqdm(total=len(material_ids), desc="[mp] structures") as bar:
            for batch in _batches(material_ids, structure_batch):
                sdocs = _with_retries(
                    lambda b=batch: mpr.materials.summary.search(
                        material_ids=b, fields=["material_id", "structure"]
                    ),
                    what=f"structure batch ({len(batch)})",
                )
                for d in sdocs:
                    struct = getattr(d, "structure", None)
                    if struct is not None:
                        struct.to(filename=str(cifs / f"{d.material_id}.cif"), fmt="cif")
                        written += 1
                bar.update(len(batch))

    print(f"[mp] wrote {written:,} new CIFs (total on disk: "
          f"{len(list(cifs.glob('*.cif'))):,}) -> {cifs}")
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
