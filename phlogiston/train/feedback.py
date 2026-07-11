"""Harvest Tier-2 verdicts into feedback training shards (active-learning loop).

Reads one or more discovery registries (``candidates.csv`` + ``relaxed/`` CIFs),
keeps the rows that carry an outside uMLIP stability label
(``e_above_hull_umlip``), featurizes the **relaxed** structure, and writes records
in the same shard format the trainer consumes (``{id, source, graph, y, mask}``)
to a *separate* feedback root. Keeping it out of the main corpus lets the
fine-tune up-weight the scarce hard negatives and keeps them out of val/test.

See ``docs/active_learning.md``. Label policy (v1): inject the uMLIP hull distance
directly as ``energy_above_hull`` (Stage-1 is AUC-selected, so ordering is what
matters and the failures sit far above threshold where the uMLIP-vs-DFT offset is
not the dominant signal).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from phlogiston.data.dataset import TARGET_KEYS
from phlogiston.data.graph import structure_to_graph
from phlogiston.data.precompute import _vector_from_labels, manifest_path, shard_dir


def _find_cif(registry: Path, row) -> Path | None:
    """Prefer the relaxed cell (the real local minimum); fall back to the
    as-generated CIF under ``cifs/``."""
    rel = str(row.get("relaxed_cif") or "").strip()
    if rel:
        p = registry / rel
        if p.exists():
            return p
    cid = int(row["id"])
    hits = sorted((registry / "cifs").glob(f"{cid:05d}_*.cif"))
    return hits[0] if hits else None


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def harvest_verified(
    registries: list[str],
    out_root: str,
    *,
    cutoff: float = 6.0,
    dedup: bool = True,
    verbose: bool = True,
) -> dict:
    """Build feedback shards from verified candidates across ``registries``.

    Returns a summary dict (counts + label stats). Idempotent per output root:
    rewrites a single ``shard_000000.pt`` + ``manifest.jsonl`` each call.
    """
    import pandas as pd
    from pymatgen.core import Structure

    def log(m):
        if verbose:
            print(m, flush=True)

    matcher = None
    if dedup:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher()

    records: list[dict] = []
    kept_structs: list = []
    ehull_vals: list[float] = []
    n_rows = n_no_label = n_no_cif = n_dup = 0

    for reg in registries:
        registry = Path(reg)
        csv_path = registry / "candidates.csv"
        if not csv_path.exists():
            log(f"[harvest] skip {reg}: no candidates.csv")
            continue
        df = pd.read_csv(csv_path)
        if "e_above_hull_umlip" not in df.columns:
            log(f"[harvest] skip {reg}: not verified (no e_above_hull_umlip column)")
            continue
        for _, row in df.iterrows():
            n_rows += 1
            umlip = _to_float(row.get("e_above_hull_umlip"))
            if umlip is None:
                n_no_label += 1
                continue
            cif = _find_cif(registry, row)
            if cif is None:
                n_no_cif += 1
                continue
            try:
                struct = Structure.from_file(str(cif))
            except Exception:  # noqa: BLE001
                n_no_cif += 1
                continue
            if matcher is not None and any(matcher.fit(struct, s) for s in kept_structs):
                n_dup += 1
                continue

            labels = {
                "energy_above_hull": umlip,
                "density": float(struct.density),
            }
            fe = _to_float(row.get("formation_energy_umlip"))
            if fe is not None:
                labels["formation_energy_per_atom"] = fe
            try:
                g = structure_to_graph(struct, cutoff=cutoff)
            except Exception:  # noqa: BLE001
                n_no_cif += 1
                continue
            vals, mask = _vector_from_labels(labels)
            graph_np = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in g.__dict__.items()}
            records.append(
                {
                    "id": f"feedback:{registry.name}:{int(row['id'])}",
                    "source": "feedback",
                    "graph": graph_np,
                    "y": vals,
                    "mask": mask,
                }
            )
            kept_structs.append(struct)
            ehull_vals.append(umlip)

    out = Path(out_root)
    sd = shard_dir(out)
    sd.mkdir(parents=True, exist_ok=True)
    # single rewritten shard (the feedback set is small by construction)
    for old in sd.glob("shard_*.pt"):
        old.unlink()
    torch.save(records, sd / "shard_000000.pt")
    with open(manifest_path(out), "w") as mf:
        for r in records:
            mf.writelines(
                json.dumps({"id": r["id"], "source": r["source"], "shard": 0,
                            "y": r["y"], "mask": r["mask"]}) + "\n"
            )

    ehull_idx = TARGET_KEYS.index("energy_above_hull")  # noqa: F841 (documents intent)
    n_pos = sum(1 for v in ehull_vals if v <= 0.1)
    summary = {
        "registries": registries,
        "rows_scanned": n_rows,
        "written": len(records),
        "no_label": n_no_label,
        "no_cif": n_no_cif,
        "dedup_dropped": n_dup,
        "near_hull_le_0.1": n_pos,
        "ehull_min": min(ehull_vals) if ehull_vals else None,
        "ehull_max": max(ehull_vals) if ehull_vals else None,
        "out_root": str(out),
    }
    if ehull_vals:
        mean = sum(ehull_vals) / len(ehull_vals)
        log(f"[harvest] wrote {len(records)} feedback records -> {sd}")
        log(f"[harvest]   e_above_hull_umlip: n={len(ehull_vals)} mean={mean:+.3f} "
            f"min={min(ehull_vals):+.3f} max={max(ehull_vals):+.3f} | "
            f"{n_pos} near-hull (<=0.1), {len(ehull_vals) - n_pos} unstable")
        log(f"[harvest]   scanned {n_rows} rows: {n_no_label} unverified, "
            f"{n_no_cif} no-CIF, {n_dup} duplicates dropped")
    else:
        log("[harvest] no verified candidates found across the given registries")
    return summary
