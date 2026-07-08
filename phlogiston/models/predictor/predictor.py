"""Predictor — shared encoder + per-target heads (see DESIGN.md).

CrystalEncoder -> one ScalarReadout head per target -> de-standardized outputs.
Masked multi-task loss (only present labels count). Per-target normalization is
stored as buffers. Stage param groups support schedule B (pretrain stability,
then fine-tune property heads at a low encoder LR).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from phlogiston.data.dataset import TARGET_KEYS
from phlogiston.layers import ScalarReadout
from phlogiston.models.encoder import CrystalEncoder

# Learnable targets (density is analytic, so excluded).
PREDICT_KEYS: tuple[str, ...] = (
    "formation_energy_per_atom",
    "energy_above_hull",
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "vickers_hardness",
    "fracture_toughness",
    "debye_temperature",
    "slack_thermal_conductivity",
)
STABILITY_KEYS: tuple[str, ...] = ("formation_energy_per_atom", "energy_above_hull")

# Targets learned in log1p space then inverted at the output. These are strongly
# nonlinear derived quantities (Hv ~ G^0.585 with a subtraction; kappa ~
# theta_D^3 / gamma^2) with a wide, right-skewed dynamic range; log1p linearizes
# them and stabilizes the standardized Huber loss. Both are >= 0, so log1p is
# well defined (and log1p(0)=0). Predictions are expm1'd back to physical units,
# so MAE/R2 remain in physical units.
LOG_TARGETS: frozenset[str] = frozenset({"vickers_hardness", "slack_thermal_conductivity"})


class Predictor(nn.Module):
    def __init__(
        self,
        mul: int = 128,
        head_hidden: tuple[int, ...] | None = None,
        huber_delta: float = 1.0,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = CrystalEncoder(mul=mul, **encoder_kwargs)
        hh = head_hidden if head_hidden is not None else (mul,)
        # one independent head per target
        self.heads = nn.ModuleList(
            ScalarReadout(f"{mul}x0e", n_out=1, hidden=hh, reduce="mean") for _ in PREDICT_KEYS
        )
        self.n_targets = len(PREDICT_KEYS)
        self.huber_delta = huber_delta

        # standardization buffers (identity until set from train stats)
        self.register_buffer("target_mean", torch.zeros(self.n_targets))
        self.register_buffer("target_std", torch.ones(self.n_targets))
        # column indices of PREDICT_KEYS within the dataset's TARGET_KEYS vector
        self.register_buffer(
            "pred_idx",
            torch.tensor([TARGET_KEYS.index(k) for k in PREDICT_KEYS], dtype=torch.long),
        )
        self._stability_idx = [PREDICT_KEYS.index(k) for k in STABILITY_KEYS]
        self._property_idx = [i for i in range(self.n_targets) if i not in self._stability_idx]
        # which prediction columns are learned in log1p space (non-persistent:
        # it's a constant derived from LOG_TARGETS, and keeping it out of the
        # state_dict avoids breaking older checkpoints on load).
        self.register_buffer(
            "log_mask",
            torch.tensor([k in LOG_TARGETS for k in PREDICT_KEYS], dtype=torch.bool),
            persistent=False,
        )

    # --- normalization ---------------------------------------------------
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set per-target mean/std. NOTE: these live in the *transform* space
        (log1p for LOG_TARGETS), i.e. compute them over ``to_transform(y)``."""
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std).clamp(min=1e-8))

    # --- target transform (physical <-> model space) --------------------
    def to_transform(self, y: torch.Tensor) -> torch.Tensor:
        """Physical units -> transform space (log1p on LOG_TARGET columns)."""
        return torch.where(self.log_mask, torch.log1p(y.clamp(min=-1 + 1e-6)), y)

    def from_transform(self, yt: torch.Tensor) -> torch.Tensor:
        """Transform space -> physical units (expm1 on LOG_TARGET columns).

        The log columns are clamped before expm1: valid physical values give
        transform-space magnitudes <= ~8.5 (log1p of the ~5000 upper bound), so
        clamping at 20 (expm1(20)~5e8) is far above any real label yet prevents
        float32 overflow to +inf during early, unconverged training -- an inf
        here would make the loss inf and NaN out the whole model.
        """
        safe = yt.clamp(min=-20.0, max=20.0)
        return torch.where(self.log_mask, torch.expm1(safe), yt)

    # --- forward ---------------------------------------------------------
    def forward(self, graph) -> torch.Tensor:
        """Return predictions ``[B, n_targets]`` in physical units."""
        node_feats = self.encoder(graph).node_feats
        preds = [head(node_feats, graph.batch) for head in self.heads]  # each [B,1]
        pred_norm = torch.cat(preds, dim=1)  # [B, T] standardized transform space
        return self.from_transform(pred_norm * self.target_std + self.target_mean)

    def slice_targets(self, y_full: torch.Tensor, mask_full: torch.Tensor):
        """Extract the PREDICT_KEYS columns from a batch's y / y_mask."""
        return y_full[:, self.pred_idx], mask_full[:, self.pred_idx]

    # --- loss ------------------------------------------------------------
    def loss(self, pred, y, mask, weights: torch.Tensor | None = None):
        """Masked multi-task loss in standardized space.

        pred, y, mask: ``[B, n_targets]`` (physical units for pred/y; bool mask).
        Returns (total, per_target dict).
        """
        # standardize in transform space (log1p for LOG_TARGETS); for those
        # columns this makes the loss operate on the linearized quantity.
        pred_n = (self.to_transform(pred) - self.target_mean) / self.target_std
        y_n = (self.to_transform(y) - self.target_mean) / self.target_std
        m = mask.to(pred_n.dtype)
        per_elem = F.huber_loss(pred_n, y_n, reduction="none", delta=self.huber_delta)
        per_elem = per_elem * m  # zero out absent labels
        denom = m.sum(dim=0).clamp(min=1.0)  # [T] present per target
        per_target = per_elem.sum(dim=0) / denom  # [T]
        if weights is None:
            weights = torch.ones_like(per_target)
        total = (per_target * weights.to(per_target)).sum()
        return total, {k: per_target[i] for i, k in enumerate(PREDICT_KEYS)}

    # --- schedule-B parameter groups ------------------------------------
    def stage1_parameters(self):
        """Encoder + stability heads (pretrain)."""
        params = list(self.encoder.parameters())
        for i in self._stability_idx:
            params += list(self.heads[i].parameters())
        return params

    def stage2_param_groups(self, encoder_lr: float, head_lr: float):
        """Low-LR encoder + all heads (fine-tune)."""
        head_params = [p for h in self.heads for p in h.parameters()]
        return [
            {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ]
