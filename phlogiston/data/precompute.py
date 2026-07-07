"""Full-corpus precompute: featurize every structure once into a sharded cache.

Featurization (pymatgen neighbor search) is CPU-bound, so this runs on CPU
workers only -- it never touches a GPU. The output is a set of shard files plus
a manifest, consumed at train time by :class:`ShardedCrystalDataset` so the GPUs
are never blocked on featurization.

Label assembly (merged into TARGET_KEYS):
  * MP:    formation_energy_per_atom, energy_above_hull  (mp_metadata.csv)
           bulk/shear moduli, hardness, toughness, Debye, kappa (mp_elasticity.csv)
  * GNoME: formation_energy_per_atom, and decomposition_energy_per_atom mapped to
           the energy_above_hull slot (GNoME's decomposition energy IS its
           distance to the convex hull).
  * density: computed analytically from the structure for EVERY material.
"""

from __future__ import annotations

import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from phlogiston.data import gnome
from phlogiston.data import materials_project as mp
from phlogiston.data.dataset import TARGET_KEYS
from phlogiston.data.graph import structure_to_graph

DENSITY_IDX = TARGET_KEYS.index("density")


def processed_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "processed"


def shard_dir(data_root: str | Path) -> Path:
    return processed_dir(data_root) / "shards"


def manifest_path(data_root: str | Path) -> Path:
    return processed_dir(data_root) / "manifest.jsonl"


# --------------------------------------------------------------------------
# Label assembly
# --------------------------------------------------------------------------
def _vector_from_labels(labels: dict) -> tuple[list[float], list[bool]]:
    vals = [float("nan")] * len(TARGET_KEYS)
    for i, k in enumerate(TARGET_KEYS):
        v = labels.get(k)
        if v is not None:
            try:
                vals[i] = float(v)
            except (TypeError, ValueError):
                pass
    mask = [bool(np.isfinite(v)) for v in vals]
    vals = [v if m else 0.0 for v, m in zip(vals, mask)]
    return vals, mask


def build_tasks(data_root: str | Path, sources=("mp", "gnome"),
                limit: int | None = None) -> list[dict]:
    """Assemble the work list: one task per structure with its label dict."""
    tasks: list[dict] = []

    if "mp" in sources:
        meta = pd.read_csv(mp.metadata_path(data_root))
        elas_path = mp.elasticity_path(data_root)
        mech_keys = ["bulk_modulus_vrh", "shear_modulus_vrh", "vickers_hardness",
                     "fracture_toughness", "debye_temperature",
                     "slack_thermal_conductivity"]
        # dict lookups (fast) instead of per-row .loc
        meta_map = meta.set_index("material_id")[
            ["formation_energy_per_atom", "energy_above_hull"]].to_dict("index")
        elas_map: dict = {}
        if elas_path.exists():
            elas = pd.read_csv(elas_path)
            elas_map = elas.set_index("material_id")[mech_keys].to_dict("index")
        cifs = mp.cif_dir(data_root)
        for mid in set(meta_map) | set(elas_map):
            cif = cifs / f"{mid}.cif"
            if not cif.exists():
                continue
            labels = dict(meta_map.get(mid, {}))
            labels.update(elas_map.get(mid, {}))
            tasks.append({"id": f"mp:{mid}", "source": "mp",
                          "ref": str(cif), "labels": labels})

    if "gnome" in sources:
        summ = gnome.load_summary(data_root, functional="pbe")
        mids = summ["materialid"].to_numpy()
        fe = summ["formation_energy_per_atom"].to_numpy()
        de = summ["decomposition_energy_per_atom"].to_numpy()
        for mid, f, d in zip(mids, fe, de):
            tasks.append({
                "id": f"gnome:{mid}", "source": "gnome", "ref": str(mid),
                # GNoME decomposition energy == distance to convex hull
                "labels": {"formation_energy_per_atom": f, "energy_above_hull": d},
            })

    if limit is not None:
        tasks = tasks[:limit]
    return tasks


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
_ZIP_HANDLE: dict = {}   # per-process cache of the GNoME zip + member map


def _gnome_zip(data_root: str):
    if "zf" not in _ZIP_HANDLE:
        zp = gnome.local_path(data_root, "structures_by_id")
        zf = zipfile.ZipFile(zp)
        members = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".cif")}
        _ZIP_HANDLE["zf"] = zf
        _ZIP_HANDLE["members"] = members
    return _ZIP_HANDLE["zf"], _ZIP_HANDLE["members"]


def _featurize_one(task: dict, data_root: str, cutoff: float) -> dict | None:
    from pymatgen.core import Structure
    try:
        if task["source"] == "mp":
            struct = Structure.from_file(task["ref"])
        else:
            zf, members = _gnome_zip(data_root)
            member = members.get(task["ref"])
            if member is None:
                return {"id": task["id"], "error": "cif-not-in-zip"}
            struct = Structure.from_str(zf.read(member).decode(), fmt="cif")

        labels = dict(task["labels"])
        labels["density"] = float(struct.density)   # analytic, always present
        g = structure_to_graph(struct, cutoff=cutoff)
        vals, mask = _vector_from_labels(labels)
        return {
            "id": task["id"], "source": task["source"],
            "graph": {k: (v if isinstance(v, int) else v) for k, v in g.__dict__.items()},
            "y": vals, "mask": mask,
        }
    except Exception as e:  # noqa: BLE001
        return {"id": task["id"], "error": repr(e)[:120]}


def _init_worker():
    # keep each worker single-threaded so N workers ~ N cores (polite on a
    # shared box) and BLAS doesn't oversubscribe.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def featurize_all(data_root: str | Path = "data", *, sources=("mp", "gnome"),
                  cutoff: float = 6.0, workers: int = 8, shard_size: int = 4096,
                  limit: int | None = None) -> dict:
    """Featurize the whole corpus into shards + a manifest. Resumable."""
    data_root = str(data_root)
    shard_dir(data_root).mkdir(parents=True, exist_ok=True)
    mpath = manifest_path(data_root)

    done_ids: set[str] = set()
    if mpath.exists():
        with open(mpath) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass

    tasks = [t for t in build_tasks(data_root, sources, limit) if t["id"] not in done_ids]
    print(f"[featurize] {len(tasks):,} to do ({len(done_ids):,} already done); "
          f"workers={workers}, shard_size={shard_size}")

    existing_shards = list(shard_dir(data_root).glob("shard_*.pt"))
    shard_idx = len(existing_shards)
    buf: list[dict] = []
    n_ok = n_err = 0

    def flush(idx: int):
        if not buf:
            return
        torch.save(buf, shard_dir(data_root) / f"shard_{idx:06d}.pt")

    worker = partial(_featurize_one, data_root=data_root, cutoff=cutoff)
    with open(mpath, "a") as mf, ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker) as ex:
        # map with chunksize streams results without materializing millions of
        # Future objects (matters at ~629k tasks).
        for res in tqdm(ex.map(worker, tasks, chunksize=64),
                        total=len(tasks), desc="[featurize]"):
            if res is None or "error" in res:
                n_err += 1
                continue
            buf.append(res)
            mf.write(json.dumps({"id": res["id"], "source": res["source"],
                                 "shard": shard_idx, "y": res["y"],
                                 "mask": res["mask"]}) + "\n")
            n_ok += 1
            if len(buf) >= shard_size:
                flush(shard_idx)
                buf = []
                shard_idx += 1
        flush(shard_idx)

    print(f"[featurize] done: {n_ok:,} graphs, {n_err:,} errors -> {shard_dir(data_root)}")
    return {"ok": n_ok, "errors": n_err, "shards": shard_idx + 1}
