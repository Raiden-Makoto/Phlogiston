"""GNoME dataset acquisition and loading (Phase 1).

GNoME ("Graph Networks for Materials Exploration", Merchant et al., Nature 2023)
released ~381k novel stable structures. The data lives in the *public* Google
Cloud bucket ``gs://gdm_materials_discovery`` and is fully readable over plain
HTTPS, so we do not need ``gcloud`` or any credentials.

Bucket layout (subset we care about)::

    gdm_materials_discovery/
      gnome_data/
        stable_materials_summary.csv     # PBE summary (~151 MB)
        stable_materials_r2scan.csv      # r2SCAN summary
        by_id.zip                        # CIFs keyed by MaterialId
        by_reduced_formula.zip           # CIFs keyed by reduced formula
        by_composition.zip               # CIFs keyed by composition
      external_data/
        mp_snapshot_summary.csv          # Materials Project snapshot (pairing)
        external_materials_summary.csv

The summary CSV columns (as released) are::

    "", Composition, MaterialId, Reduced Formula, Elements, NSites, Volume,
    Density, Point Group, Space Group, Space Group Number, Crystal System,
    Uncorrected Energy, Corrected Energy, Formation Energy Per Atom,
    Decomposition Energy Per Atom, Dimensionality Cheon, Bandgap, Is Train,
    Decomposition Energy Per Atom All, Decomposition Energy Per Atom Relative,
    Decomposition Energy Per Atom MP, Decomposition Energy Per Atom MP OQMD,
    Data Directory
"""

from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

GCS_BASE = "https://storage.googleapis.com/gdm_materials_discovery"

# Logical name -> path within the bucket.
GNOME_FILES: dict[str, str] = {
    "summary_pbe": "gnome_data/stable_materials_summary.csv",
    "summary_r2scan": "gnome_data/stable_materials_r2scan.csv",
    "structures_by_id": "gnome_data/by_id.zip",
    "structures_by_reduced_formula": "gnome_data/by_reduced_formula.zip",
    "structures_by_composition": "gnome_data/by_composition.zip",
    "mp_snapshot": "external_data/mp_snapshot_summary.csv",
    "external_summary": "external_data/external_materials_summary.csv",
}

# The three summary/structure keys most useful for the discovery pipeline.
DEFAULT_KEYS: tuple[str, ...] = ("summary_pbe", "mp_snapshot")

_CHUNK = 1 << 20  # 1 MiB


def _ssl_verify() -> bool:
    """Whether to verify TLS. Disable via PHLOGISTON_SSL_NO_VERIFY=1 (e.g. behind
    a TLS-inspecting corporate proxy)."""
    return os.environ.get("PHLOGISTON_SSL_NO_VERIFY", "0") not in ("1", "true", "True")


def raw_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "raw" / "gnome"


def local_path(data_root: str | Path, key: str) -> Path:
    """Local destination for a registry ``key`` (mirrors the bucket sub-path)."""
    if key not in GNOME_FILES:
        raise KeyError(f"Unknown GNoME file key {key!r}. Known: {sorted(GNOME_FILES)}")
    return raw_dir(data_root) / GNOME_FILES[key]


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Stream ``url`` to ``dest`` with a progress bar; skip if already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    verify = _ssl_verify()

    # Determine remote size for skip / progress.
    remote_size: int | None = None
    try:
        head = requests.head(url, timeout=30, verify=verify, allow_redirects=True)
        if head.ok and "content-length" in head.headers:
            remote_size = int(head.headers["content-length"])
    except requests.RequestException:
        pass

    if dest.exists() and not force:
        if remote_size is None or dest.stat().st_size == remote_size:
            print(f"[gnome] up to date: {dest}")
            return dest

    with requests.get(url, stream=True, timeout=60, verify=verify) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", remote_size or 0)) or None
        tmp = dest.with_suffix(dest.suffix + ".part")
        with (
            open(tmp, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"[gnome] {dest.name}",
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
    return dest


def download(
    data_root: str | Path = "data",
    keys: Iterable[str] = DEFAULT_KEYS,
    force: bool = False,
) -> dict[str, Path]:
    """Download one or more registry ``keys`` into ``data_root``.

    Returns a mapping of key -> local path.
    """
    out: dict[str, Path] = {}
    for key in keys:
        url = f"{GCS_BASE}/{GNOME_FILES[key]}"
        dest = local_path(data_root, key)
        out[key] = download_file(url, dest, force=force)
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case the released column names (e.g. 'Formation Energy Per Atom'
    -> 'formation_energy_per_atom')."""

    def norm(c: str) -> str:
        c = c.strip()
        c = re.sub(r"\s+", "_", c)
        return c.lower()

    return df.rename(columns={c: norm(c) for c in df.columns})


def load_summary(
    data_root: str | Path = "data",
    functional: str = "pbe",
    normalize_columns: bool = True,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the GNoME summary table as a DataFrame.

    Parameters
    ----------
    functional: "pbe" (default) or "r2scan".
    """
    key = {"pbe": "summary_pbe", "r2scan": "summary_r2scan"}.get(functional)
    if key is None:
        raise ValueError("functional must be 'pbe' or 'r2scan'")

    path = local_path(data_root, key)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"{path} not found; run download() first.")
        download(data_root, keys=[key])

    # First column is an unnamed integer index in the released file.
    df = pd.read_csv(path, index_col=0)
    if normalize_columns:
        df = _normalize_columns(df)
    return df


def filter_stable(
    df: pd.DataFrame,
    max_decomposition_energy: float = 0.0,
    column: str = "decomposition_energy_per_atom",
) -> pd.DataFrame:
    """Keep rows on/below the convex hull (decomposition energy <= threshold).

    A threshold of 0.0 selects thermodynamically stable materials; a small
    positive value (e.g. 0.05 eV/atom) selects metastable candidates too.
    """
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found. Available: {list(df.columns)}")
    return df[df[column] <= max_decomposition_energy].copy()


def read_structure_cif(
    data_root: str | Path,
    material_id: str | None = None,
    reduced_formula: str | None = None,
) -> str:
    """Return the raw CIF text for a GNoME structure from the downloaded zips.

    Provide exactly one of ``material_id`` (uses ``by_id.zip``) or
    ``reduced_formula`` (uses ``by_reduced_formula.zip``). The relevant zip must
    have been downloaded first (see ``download``).
    """
    if (material_id is None) == (reduced_formula is None):
        raise ValueError("Provide exactly one of material_id or reduced_formula")

    if material_id is not None:
        zip_path = local_path(data_root, "structures_by_id")
        needle = str(material_id)
    else:
        zip_path = local_path(data_root, "structures_by_reduced_formula")
        needle = str(reduced_formula)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Download it first, e.g. "
            f"download(keys=['{'structures_by_id' if material_id else 'structures_by_reduced_formula'}'])"
        )

    with zipfile.ZipFile(zip_path) as zf:
        # Member names look like "by_id/<MaterialId>.CIF"; match on the stem.
        candidates = [n for n in zf.namelist() if not n.endswith("/") and Path(n).stem == needle]
        if not candidates:
            # Fall back to a substring match for robustness.
            candidates = [n for n in zf.namelist() if needle in n and n.lower().endswith(".cif")]
        if not candidates:
            raise KeyError(f"No CIF for {needle!r} found in {zip_path.name}")
        with zf.open(candidates[0]) as fh:
            return fh.read().decode("utf-8")
