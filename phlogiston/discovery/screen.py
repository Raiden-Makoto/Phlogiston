"""Property/stability screen: score candidate structures with the trained
Predictor (pipeline §7). Density is analytic; every other target comes from the
predictor in physical units. This is the independent verifier that gates and
ranks generated candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from phlogiston.data.dataset import collate
from phlogiston.data.graph import structure_to_graph
from phlogiston.models.predictor import PREDICT_KEYS, Predictor


def load_predictor(ckpt_path: str, device: str | None = None) -> Predictor:
    """Rebuild a Predictor from a checkpoint (architecture from stored hparams,
    normalization from stored mean/std)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt.get("hparams", {})
    model = Predictor(
        mul=hp.get("mul", 128), n_layers=hp.get("n_layers", 2), correlation=hp.get("correlation", 3)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    if "mean" in ckpt and "std" in ckpt:
        model.set_normalization(ckpt["mean"].to(device), ckpt["std"].to(device))
    model.eval()
    return model


@dataclass
class ScoredCandidate:
    structure: object  # pymatgen Structure
    properties: dict[str, float] = field(default_factory=dict)  # physical units
    formula: str = ""
    score: float = float("nan")  # multi-objective score (set by rank_candidates)
    is_pareto: bool = False  # on the Pareto front of the ranked pool

    @property
    def energy_above_hull(self) -> float:
        return self.properties.get("energy_above_hull", float("nan"))


class PropertyScreen:
    """Featurize + predict every target for candidate structures."""

    def __init__(self, predictor: Predictor, cutoff: float = 6.0, device: str | None = None):
        self.model = predictor
        self.cutoff = cutoff
        self.device = device or next(predictor.parameters()).device

    @torch.no_grad()
    def score(self, structures, batch_size: int = 64) -> list[ScoredCandidate]:
        """Return a ScoredCandidate per input structure. Structures that fail to
        featurize (e.g. isolated atoms) are skipped."""
        out: list[ScoredCandidate] = []
        buf, keep = [], []
        for s in structures:
            try:
                g = structure_to_graph(s, cutoff=self.cutoff)
            except (ValueError, RuntimeError):
                continue  # invalid geometry -> drop
            buf.append((g, torch.zeros(1), torch.zeros(1, dtype=torch.bool)))
            keep.append(s)
            if len(buf) >= batch_size:
                out.extend(self._run(buf, keep))
                buf, keep = [], []
        if buf:
            out.extend(self._run(buf, keep))
        return out

    def _run(self, buf, structs) -> list[ScoredCandidate]:
        batch = collate(buf).to(self.device)
        preds = self.model(batch).cpu()  # [B, n_targets] physical units
        results = []
        for i, s in enumerate(structs):
            props = {k: float(preds[i, j]) for j, k in enumerate(PREDICT_KEYS)}
            props["density"] = float(s.density)  # analytic (g/cm^3)
            results.append(
                ScoredCandidate(structure=s, properties=props, formula=s.composition.reduced_formula)
            )
        return results
