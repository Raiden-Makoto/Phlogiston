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

import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from tqdm import tqdm

API_KEY_ENV_VARS = ("MP54AC", "MP_API_KEY", "MP_API_TOKEN")

# Radioactive elements to screen out of the training corpus: those with no
# stable isotopes (Tc, Pm) plus Po..Pu (Z 84-94). Transplutonium/superheavy
# elements are omitted -- they never appear in Materials Project and their
# symbols are rejected by the API's element validation.
RADIOACTIVE_ELEMENTS: tuple[str, ...] = (
    "Tc",
    "Pm",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
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


def synth_path(data_root: str | Path) -> Path:
    """Synthesizability provenance table (material_id, theoretical, has_icsd)."""
    return raw_dir(data_root) / "mp_synth.csv"


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


def _download_structures(
    mpr, material_ids, cifs: Path, structure_batch: int = 500, force: bool = False
) -> int:
    """Download structures for ``material_ids`` in batches, writing one CIF each
    (skipping existing unless ``force``). Resumable. Returns count written."""
    todo = [m for m in material_ids if force or not (cifs / f"{m}.cif").exists()]
    have = len(material_ids) - len(todo)
    print(
        f"[mp] structures: {len(todo):,} to fetch ({have:,} already on disk), "
        f"batch={structure_batch}"
    )
    written = 0
    with tqdm(total=len(todo), desc="[mp] structures") as bar:
        for batch in _batches(todo, structure_batch):
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
    return written


def _vrh(modulus) -> float | None:
    """Extract the Voigt-Reuss-Hill average from an MP modulus field (dict or obj)."""
    if modulus is None:
        return None
    if isinstance(modulus, dict):
        return modulus.get("vrh")
    return getattr(modulus, "vrh", None)


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
        print("[mp] phase 2/2: downloading structures")
        written = _download_structures(
            mpr, df["material_id"].tolist(), cifs, structure_batch, force
        )

    print(
        f"[mp] wrote {written:,} new CIFs (total on disk: "
        f"{len(list(cifs.glob('*.cif'))):,}) -> {cifs}"
    )
    return df


def load_metadata(data_root: str | Path = "data") -> pd.DataFrame:
    """Load the persisted MP metadata table."""
    meta = metadata_path(data_root)
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found; run fetch_structures() first.")
    return pd.read_csv(meta)


def fetch_synthesizability(
    data_root: str | Path = "data",
    *,
    api_key: str | None = None,
    material_ids: Sequence[str] | None = None,
    chunk: int = 1000,
) -> pd.DataFrame:
    """Fetch the experimental-provenance flags used for the Tier-1
    synthesizability label, for the material_ids already in ``mp_metadata.csv``
    (or an explicit list). Writes ``mp_synth.csv`` and returns it.

    Columns: ``material_id``, ``theoretical`` (True = DFT-only/never observed),
    ``has_icsd`` (present in the ICSD, i.e. experimentally reported). A material
    is a synthesizability *positive* iff it has been experimentally observed
    (``theoretical == False`` or ``has_icsd``); everything else (theoretical MP +
    all of GNoME) is treated as unlabeled/negative for PU training.
    """
    from mp_api.client import MPRester

    key = get_api_key(api_key)
    if material_ids is None:
        material_ids = load_metadata(data_root)["material_id"].astype(str).tolist()
    ids = [str(m) for m in material_ids]

    rows: list[dict] = []
    with MPRester(key) as mpr:
        with tqdm(total=len(ids), desc="[mp] synth flags") as bar:
            for batch in _batches(ids, chunk):
                docs = _with_retries(
                    lambda b=batch: mpr.materials.summary.search(
                        material_ids=b, fields=["material_id", "theoretical", "database_IDs"]
                    ),
                    what=f"synth batch ({len(batch)})",
                )
                for d in docs:
                    dbids = getattr(d, "database_IDs", None) or {}
                    # database_IDs maps db name -> list of ids; ICSD presence => observed
                    has_icsd = False
                    if isinstance(dbids, dict):
                        has_icsd = bool(dbids.get("icsd"))
                    rows.append(
                        {
                            "material_id": str(d.material_id),
                            "theoretical": bool(getattr(d, "theoretical", True)),
                            "has_icsd": has_icsd,
                        }
                    )
                bar.update(len(batch))

    df = pd.DataFrame(rows, columns=["material_id", "theoretical", "has_icsd"])
    out = synth_path(data_root)
    df.to_csv(out, index=False)
    n_obs = int((~df["theoretical"] | df["has_icsd"]).sum())
    print(f"[mp] synth flags: {len(df):,} rows, {n_obs:,} experimentally observed -> {out}")
    return df


def load_structure(data_root: str | Path, material_id: str):
    """Load a single MP structure as a pymatgen ``Structure`` from its CIF."""
    from pymatgen.core import Structure

    path = cif_dir(data_root) / f"{material_id}.cif"
    if not path.exists():
        raise FileNotFoundError(f"No CIF for {material_id} at {path}")
    return Structure.from_file(str(path))


def elasticity_path(data_root: str | Path) -> Path:
    return raw_dir(data_root) / "mp_elasticity.csv"


# Columns of mp_elasticity.csv: raw MP fields + derived mechanical/thermal targets.
_ELASTIC_COLUMNS: tuple[str, ...] = (
    "material_id",
    "formula_pretty",
    "nsites",
    "density",
    "volume",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "poisson_mp",
    "universal_anisotropy",
    "debye_temperature_mp",
    # derived (phlogiston.data.properties)
    "youngs_modulus",
    "poisson_ratio",
    "pugh_ratio",
    "vickers_hardness",
    "fracture_toughness",
    "debye_temperature",
    "sound_velocity_mean",
    "gruneisen",
    "slack_thermal_conductivity",
)


def fetch_elasticity(
    data_root: str | Path = "data",
    *,
    api_key: str | None = None,
    limit: int | None = None,
    save_cif: bool = True,
    structure_batch: int = 500,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch MP elastic constants (~13k materials), derive the mechanical/thermal
    target set, and write ``mp_elasticity.csv``.

    Uses the bulk/shear VRH moduli + cell geometry to compute Young's modulus,
    Poisson ratio, Pugh ratio, Vickers hardness, fracture toughness, Debye
    temperature, and Slack thermal conductivity (see ``properties.derive_all``).
    When ``save_cif`` is set, also downloads any missing structures for these
    materials so every label has matching geometry.
    """
    from mp_api.client import MPRester
    from pymatgen.core import Composition

    from phlogiston.data import properties as props

    key = get_api_key(api_key)
    cifs = cif_dir(data_root)
    cifs.mkdir(parents=True, exist_ok=True)

    # NB: young_modulus is not a queryable elasticity field; we derive it.
    fields = [
        "material_id",
        "formula_pretty",
        "nsites",
        "elements",
        "bulk_modulus",
        "shear_modulus",
        "homogeneous_poisson",
        "universal_anisotropy",
        "debye_temperature",
        "density",
        "volume",
    ]

    with MPRester(key) as mpr:
        # Always chunk explicitly: the fully-unchunked elasticity search returns
        # docs with the nested moduli unpopulated (None), whereas an explicit
        # num_chunks/chunk_size retrieval fills them in correctly.
        chunk_size = 1000
        if limit is not None:
            num_chunks, chunk_size = 1, min(limit, 1000)
        else:
            total = _with_retries(lambda: mpr.materials.elasticity.count(), what="elasticity count")
            num_chunks = max(1, math.ceil(total / chunk_size))
            print(f"[mp] elasticity records available: {total:,}")

        print(f"[mp] querying elasticity (num_chunks={num_chunks}, chunk_size={chunk_size}) ...")
        docs = _with_retries(
            lambda: mpr.materials.elasticity.search(
                fields=fields, num_chunks=num_chunks, chunk_size=chunk_size
            ),
            what="elasticity search",
        )

        rows: list[dict] = []
        for doc in tqdm(docs, desc="[mp] elasticity"):
            K = _vrh(getattr(doc, "bulk_modulus", None))
            G = _vrh(getattr(doc, "shear_modulus", None))
            if K is None or G is None or K <= 0 or G <= 0:
                continue  # skip degenerate/soft entries the models can't use

            density = getattr(doc, "density", None) or float("nan")
            volume = getattr(doc, "volume", None) or float("nan")
            nsites = getattr(doc, "nsites", None) or 0
            formula = getattr(doc, "formula_pretty", None)
            try:
                comp = Composition(formula)
                mean_mass = comp.weight / comp.num_atoms
            except Exception:
                mean_mass = float("nan")

            dp = props.derive_all(K, G, density, volume, nsites, mean_mass)
            rows.append(
                {
                    "material_id": str(doc.material_id),
                    "formula_pretty": formula,
                    "nsites": nsites,
                    "density": density,
                    "volume": volume,
                    "bulk_modulus_vrh": K,
                    "shear_modulus_vrh": G,
                    "poisson_mp": getattr(doc, "homogeneous_poisson", None),
                    "universal_anisotropy": getattr(doc, "universal_anisotropy", None),
                    "debye_temperature_mp": getattr(doc, "debye_temperature", None),
                    "youngs_modulus": dp.youngs_modulus,
                    "poisson_ratio": dp.poisson_ratio,
                    "pugh_ratio": dp.pugh_ratio,
                    "vickers_hardness": dp.vickers_hardness,
                    "fracture_toughness": dp.fracture_toughness,
                    "debye_temperature": dp.debye_temperature,
                    "sound_velocity_mean": dp.sound_velocity_mean,
                    "gruneisen": dp.gruneisen,
                    "slack_thermal_conductivity": dp.slack_thermal_conductivity,
                }
            )
            if limit is not None and len(rows) >= limit:
                break

        df = pd.DataFrame(rows, columns=list(_ELASTIC_COLUMNS))
        out = elasticity_path(data_root)
        df.to_csv(out, index=False)
        print(f"[mp] wrote {len(df):,} elasticity rows -> {out}")

        if save_cif and len(df):
            print("[mp] fetching structures for elasticity materials")
            written = _download_structures(
                mpr, df["material_id"].tolist(), cifs, structure_batch, force
            )
            print(
                f"[mp] fetched {written:,} new structures "
                f"(total on disk: {len(list(cifs.glob('*.cif'))):,})"
            )

    return df


def load_elasticity(data_root: str | Path = "data") -> pd.DataFrame:
    """Load the persisted elasticity + derived-property table."""
    path = elasticity_path(data_root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run fetch_elasticity() first.")
    return pd.read_csv(path)
