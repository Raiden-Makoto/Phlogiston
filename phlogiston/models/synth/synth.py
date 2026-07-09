"""Synthesizability classifier (Tier-1 feasibility).

Same E(3)-equivariant ``CrystalEncoder`` as the Predictor, topped with a single
sigmoid head that estimates P(experimentally synthesizable). It's a
positive-unlabeled problem: the positives are crystals that have actually been
made (Materials Project entries with ICSD provenance / ``theoretical=False``);
everything else -- theoretical MP entries and the entire GNoME hypothetical set
-- is treated as unlabeled/negative. The resulting score is a *learned* synthesis
prior that complements the Tier-0 composition rules and gates/ranks candidates
before the Tier-2 physics verification (ensemble uMLIP).

Because the encoder is architecturally identical to the Predictor's, the head
can be warm-started from a trained stability/predictor checkpoint
(``load_encoder_from``), which already knows good crystal representations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from phlogiston.layers import ScalarReadout
from phlogiston.models.encoder import CrystalEncoder


class SynthesizabilityModel(nn.Module):
    def __init__(self, mul: int = 128, head_hidden: tuple[int, ...] | None = None, **encoder_kwargs):
        super().__init__()
        self.encoder = CrystalEncoder(mul=mul, **encoder_kwargs)
        hh = head_hidden if head_hidden is not None else (mul,)
        self.head = ScalarReadout(f"{mul}x0e", n_out=1, hidden=hh, reduce="mean")

    def forward(self, graph) -> torch.Tensor:
        """Return per-graph logits ``[B]`` (pre-sigmoid)."""
        node_feats = self.encoder(graph).node_feats
        return self.head(node_feats, graph.batch).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, graph) -> torch.Tensor:
        """Return P(synthesizable) in [0, 1], shape ``[B]``."""
        return torch.sigmoid(self.forward(graph))

    def loss(self, logits, target, pos_weight: torch.Tensor | None = None):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def load_encoder_from(self, ckpt_path: str, map_location="cpu") -> int:
        """Warm-start the encoder from a Predictor/stability checkpoint (identical
        architecture). Returns the number of encoder tensors loaded."""
        sd = torch.load(ckpt_path, map_location=map_location)["model"]
        prefix = "encoder."
        enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        self.encoder.load_state_dict(enc, strict=False)
        return len(enc)
