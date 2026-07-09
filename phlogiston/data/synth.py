"""Synthesizability labels + dataset for the Tier-1 classifier.

Reuses the precomputed graph shards (each record keeps its provenance ``id`` /
``source``) and attaches a binary label:

  * positive (1): a Materials Project entry that has been experimentally
    observed -- ``theoretical == False`` or present in the ICSD (from
    ``mp_synth.csv``, produced by ``phlogiston fetch-mp-synth``).
  * unlabeled/negative (0): theoretical-only MP entries and all of GNoME
    (DFT-predicted, never synthesized).

MP entries whose provenance we don't have (missing from ``mp_synth.csv``) are
marked *invalid* and excluded from training rather than silently labeled 0.
"""

from __future__ import annotations

import torch

from phlogiston.data.dataset import ShardedCrystalDataset


def build_synth_labels(data_root: str) -> dict[str, int]:
    """Map ``mp:<material_id>`` -> {1 observed, 0 theoretical} from mp_synth.csv.
    Empty if the file is missing (caller should then treat all MP as invalid)."""
    import pandas as pd

    from phlogiston.data import materials_project as mp

    sp = mp.synth_path(data_root)
    labels: dict[str, int] = {}
    if not sp.exists():
        return labels
    df = pd.read_csv(sp)
    theo = df["theoretical"].astype(bool)
    icsd = df["has_icsd"].astype(bool) if "has_icsd" in df.columns else theo & False
    observed = (~theo) | icsd
    for mid, obs in zip(df["material_id"].astype(str), observed, strict=False):
        labels[f"mp:{mid}"] = int(bool(obs))
    return labels


class SynthesizabilityDataset(ShardedCrystalDataset):
    """Sharded graphs yielding ``(graph, [label], [valid])`` for binary training.

    ``include_gnome`` toggles whether the (large) GNoME negative pool is used;
    ``.valid`` / ``.labels`` are per-record arrays for stratified splitting and
    ``pos_weight`` computation.
    """

    def __init__(
        self,
        data_root: str,
        *,
        in_memory: bool = True,
        max_shards: int | None = None,
        include_gnome: bool = True,
    ):
        super().__init__(data_root, in_memory=in_memory, max_shards=max_shards)
        observed = build_synth_labels(data_root)
        self.labels: list[float] = []
        self.valid: list[bool] = []
        for r in self.records:
            rid, src = r["id"], r.get("source", "")
            if src == "mp":
                if rid in observed:
                    self.labels.append(float(observed[rid]))
                    self.valid.append(True)
                else:  # provenance unknown -> exclude
                    self.labels.append(0.0)
                    self.valid.append(False)
            else:  # gnome (hypothetical) -> unlabeled negative
                self.labels.append(0.0)
                self.valid.append(bool(include_gnome))

    def valid_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.valid) if v]

    def positive_count(self, indices=None) -> tuple[int, int]:
        """(#positive, #total) over ``indices`` (default: all valid)."""
        idx = indices if indices is not None else self.valid_indices()
        pos = sum(1 for i in idx if self.labels[i] > 0.5)
        return pos, len(idx)

    def __getitem__(self, idx: int):
        g, _, _ = super().__getitem__(idx)  # reuse parent graph construction
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        m = torch.tensor([self.valid[idx]], dtype=torch.bool)
        return g, y, m
