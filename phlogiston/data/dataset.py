"""Batched crystal-graph dataset with masked multi-task targets (Phase 3b).

Combines the partially-labeled sources into one training interface:
each item is (CrystalGraph, target_vector, target_mask). Because no material
has every label, ``target_mask`` marks which entries of ``target_vector`` are
real; the training loss only counts masked-in targets.

Batching concatenates per-structure graphs into one big disjoint graph (native
torch, no pyg-lib): node tensors are stacked, ``edge_index`` is offset by the
running node count, and a ``batch`` vector maps each node to its graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from phlogiston.data.graph import CrystalGraph, structure_to_graph

# Canonical, fixed order of prediction targets. Density is analytic (always
# present); the rest come from GNoME/MP and may be missing per material.
TARGET_KEYS: tuple[str, ...] = (
    "formation_energy_per_atom",  # stability (GNoME + MP)
    "energy_above_hull",  # stability (MP; GNoME has decomposition E)
    "density",  # light-for-flight (analytic)
    "bulk_modulus_vrh",  # strength
    "shear_modulus_vrh",  # strength
    "vickers_hardness",  # hardness
    "fracture_toughness",  # resistance to breaking
    "debye_temperature",  # thermal
    "slack_thermal_conductivity",  # thermal
)


# Per-target physically-plausible ranges (inclusive). Labels outside these are
# treated as *absent* (mask False), not just NaN. Rationale: the Materials
# Project elasticity table contains unphysical outliers (negative or
# thousands-of-GPa moduli), and the nonlinear derived formulas (Vickers ~
# G^0.585, Slack ~ theta_D^3 / gamma^2) amplify those into astronomically large
# values (1e16+). A single such label poisons the per-target mean/std, which in
# turn inflates physical-unit MAE (= standardized error x std) to nonsense.
TARGET_BOUNDS: dict[str, tuple[float, float]] = {
    "formation_energy_per_atom": (-10.0, 10.0),  # eV/atom
    # For MP this is e_above_hull (>=0); for GNoME the column holds the
    # *decomposition energy*, which is negative for novel below-hull materials
    # (the discovery signal, observed down to ~-2.5). Lower bound must stay
    # generous so we don't mask those out; only absurd values are dropped.
    "energy_above_hull": (-6.0, 10.0),  # eV/atom
    "density": (0.1, 30.0),  # g/cm^3
    "bulk_modulus_vrh": (1.0, 1000.0),  # GPa (diamond ~440)
    "shear_modulus_vrh": (1.0, 800.0),  # GPa
    "vickers_hardness": (0.0, 200.0),  # GPa (diamond ~90-100)
    "fracture_toughness": (0.0, 100.0),  # MPa*m^0.5
    "debye_temperature": (1.0, 3000.0),  # K (diamond ~2200)
    "slack_thermal_conductivity": (0.0, 5000.0),  # W/m/K (diamond ~2000-3000)
}
_BOUND_LO = torch.tensor([TARGET_BOUNDS[k][0] for k in TARGET_KEYS])
_BOUND_HI = torch.tensor([TARGET_BOUNDS[k][1] for k in TARGET_KEYS])


def physical_mask(vals: torch.Tensor) -> torch.Tensor:
    """Per-target validity: finite AND within the physical bounds. Shape [..., T]."""
    lo = _BOUND_LO.to(vals)
    hi = _BOUND_HI.to(vals)
    return torch.isfinite(vals) & (vals >= lo) & (vals <= hi)


def targets_to_vector(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (target_vector, mask) for TARGET_KEYS from a label dict.

    Missing / NaN / physically-implausible entries get value 0 and mask False.
    """
    vals = np.full(len(TARGET_KEYS), np.nan, dtype=np.float64)
    for i, k in enumerate(TARGET_KEYS):
        v = row.get(k, None)
        if v is not None:
            try:
                vals[i] = float(v)
            except (TypeError, ValueError):
                pass
    t = torch.from_numpy(vals).float()
    mask = physical_mask(t)
    t = torch.where(mask, t, torch.zeros_like(t))
    return t, mask


@dataclass
class BatchedGraph:
    z: torch.Tensor  # [Ntot] int64
    pos: torch.Tensor  # [Ntot, 3]
    edge_index: torch.Tensor  # [2, Etot] (global node indices)
    edge_vec: torch.Tensor  # [Etot, 3]
    edge_len: torch.Tensor  # [Etot]
    batch: torch.Tensor  # [Ntot] graph id per node
    lattice: torch.Tensor  # [B, 3, 3]
    y: torch.Tensor  # [B, T] targets (0 where masked out)
    y_mask: torch.Tensor  # [B, T] bool
    n_graphs: int

    def to(self, device) -> BatchedGraph:
        return BatchedGraph(
            z=self.z.to(device),
            pos=self.pos.to(device),
            edge_index=self.edge_index.to(device),
            edge_vec=self.edge_vec.to(device),
            edge_len=self.edge_len.to(device),
            batch=self.batch.to(device),
            lattice=self.lattice.to(device),
            y=self.y.to(device),
            y_mask=self.y_mask.to(device),
            n_graphs=self.n_graphs,
        )


def collate(items: list[tuple[CrystalGraph, torch.Tensor, torch.Tensor]]) -> BatchedGraph:
    """Collate (graph, y, mask) tuples into a single BatchedGraph."""
    z, pos, edge_index, edge_vec, edge_len, batch, lattice = [], [], [], [], [], [], []
    ys, masks = [], []
    node_offset = 0
    for i, (g, y, m) in enumerate(items):
        z.append(g.z)
        pos.append(g.pos)
        edge_index.append(g.edge_index + node_offset)
        edge_vec.append(g.edge_vec)
        edge_len.append(g.edge_len)
        batch.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        lattice.append(g.lattice)
        ys.append(y)
        masks.append(m)
        node_offset += g.num_nodes

    return BatchedGraph(
        z=torch.cat(z),
        pos=torch.cat(pos, dim=0),
        edge_index=torch.cat(edge_index, dim=1),
        edge_vec=torch.cat(edge_vec, dim=0),
        edge_len=torch.cat(edge_len, dim=0),
        batch=torch.cat(batch),
        lattice=torch.stack(lattice, dim=0),
        y=torch.stack(ys, dim=0),
        y_mask=torch.stack(masks, dim=0),
        n_graphs=len(items),
    )


class CrystalDataset(Dataset):
    """Featurizes structures on access; optional on-disk graph cache.

    ``entries`` is a list of dicts, each with a ``loader`` callable returning a
    pymatgen Structure, an ``id`` (for caching), and the label fields.
    """

    def __init__(self, entries: list[dict], cutoff: float = 6.0, cache_dir: str | None = None):
        self.entries = entries
        self.cutoff = cutoff
        self.cache_dir = cache_dir
        if cache_dir:
            import os

            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.entries)

    def _graph(self, entry: dict) -> CrystalGraph:
        if self.cache_dir:
            import os

            path = os.path.join(self.cache_dir, f"{entry['id']}.pt")
            if os.path.exists(path):
                d = torch.load(path)
                return CrystalGraph(**d)
        g = structure_to_graph(entry["loader"](), cutoff=self.cutoff)
        if self.cache_dir:
            import os

            path = os.path.join(self.cache_dir, f"{entry['id']}.pt")
            torch.save(g.__dict__, path)
        return g

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        g = self._graph(entry)
        y, m = targets_to_vector(entry)
        return g, y, m


class ShardedCrystalDataset(Dataset):
    """Reads precomputed graph shards (see phlogiston.data.precompute).

    Loads all shards into memory by default (the corpus is ~12 GB of tensors and
    the box has ample RAM), giving fast shuffled access with the GPUs never
    waiting on featurization.
    """

    def __init__(self, data_root: str, in_memory: bool = True, max_shards: int | None = None):
        from phlogiston.data.precompute import shard_dir

        shards = sorted(shard_dir(data_root).glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(
                f"No shards under {shard_dir(data_root)}; run `phlogiston featurize` first."
            )
        if max_shards is not None:  # partial load (tests / quick runs)
            shards = shards[:max_shards]
        self.records: list[dict] = []
        for s in shards:
            self.records.extend(torch.load(s, weights_only=False))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        gd = r["graph"]
        g = CrystalGraph(
            z=torch.as_tensor(gd["z"], dtype=torch.long),
            pos=torch.as_tensor(gd["pos"], dtype=torch.float32),
            lattice=torch.as_tensor(gd["lattice"], dtype=torch.float32),
            edge_index=torch.as_tensor(gd["edge_index"], dtype=torch.long),
            edge_vec=torch.as_tensor(gd["edge_vec"], dtype=torch.float32),
            edge_len=torch.as_tensor(gd["edge_len"], dtype=torch.float32),
            num_nodes=int(gd["num_nodes"]),
        )
        y = torch.tensor(r["y"], dtype=torch.float32)
        # Re-apply physical bounds on load: existing shards were written before
        # bounds-checking, so drop out-of-range labels (mask them absent) here.
        m = torch.tensor(r["mask"], dtype=torch.bool) & physical_mask(y)
        y = torch.where(m, y, torch.zeros_like(y))
        return g, y, m
