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
from phlogiston.models.predictor import PREDICT_KEYS, STABILITY_KEYS, Predictor


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


def load_synth_model(ckpt_path: str, device: str | None = None):
    """Rebuild the Tier-1 SynthesizabilityModel from a checkpoint."""
    from phlogiston.models.synth import SynthesizabilityModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt.get("hparams", {})
    model = SynthesizabilityModel(
        mul=hp.get("mul", 128), n_layers=hp.get("n_layers", 2), correlation=hp.get("correlation", 3)
    ).to(device)
    model.load_state_dict(ckpt["model"])
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
    """Featurize + predict every target for candidate structures.

    Optionally **decoupled**: a specialized ``stability_model`` supplies the
    stability columns (formation energy, energy_above_hull) for the gate, while
    ``predictor`` supplies the mechanical/thermal properties. This is the
    recommended setup — a property model fine-tuned hard on the labeled subset
    scores properties best but erodes the stability gate, so we keep a separate
    stability specialist (e.g. the Stage-1 checkpoint, AUC ~0.92) for gating.
    """

    def __init__(
        self,
        predictor: Predictor,
        stability_model: Predictor | None = None,
        synth_model=None,
        cutoff: float = 6.0,
        device: str | None = None,
    ):
        self.model = predictor
        self.stability_model = stability_model
        self.synth_model = synth_model  # optional Tier-1 synthesizability classifier
        self.cutoff = cutoff
        self.device = device or next(predictor.parameters()).device
        self._stab_cols = [PREDICT_KEYS.index(k) for k in STABILITY_KEYS]

    @torch.no_grad()
    def score(self, structures, batch_size: int = 64) -> list[ScoredCandidate]:
        """Return a ScoredCandidate per input structure. Structures that fail to
        featurize (e.g. isolated atoms) are skipped."""
        out: list[ScoredCandidate] = []
        buf, keep = [], []
        for s in structures:
            try:
                g = structure_to_graph(s, cutoff=self.cutoff)
            except Exception:  # noqa: BLE001  invalid geometry / degenerate cell -> drop
                continue
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
        if self.stability_model is not None:  # decoupled gate: override stability cols
            sp = self.stability_model(batch).cpu()
            for c in self._stab_cols:
                preds[:, c] = sp[:, c]
        synth = self.synth_model.predict_proba(batch).cpu() if self.synth_model is not None else None
        results = []
        for i, s in enumerate(structs):
            props = {k: float(preds[i, j]) for j, k in enumerate(PREDICT_KEYS)}
            props["density"] = float(s.density)  # analytic (g/cm^3)
            if synth is not None:
                props["synthesizability"] = float(synth[i])
            results.append(
                ScoredCandidate(structure=s, properties=props, formula=s.composition.reduced_formula)
            )
        return results
