"""Exponential moving average of model weights.

Diffusion/score models sample much better from an EMA of the training weights
than from the raw (noisy) weights, so the CDVAE trainer keeps an EMA and
checkpoints it alongside the live weights. Usage:

    ema = EMA(model, decay=0.999)
    ...
    opt.step(); ema.update(model)          # after each optimizer step
    with ema.averaged(model):              # temporarily swap in EMA weights
        evaluate(model)                    # ... restored on exit
"""

from __future__ import annotations

import contextlib
import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        # shadow copy of all learnable parameters (detached, on the model device)
        self.shadow = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def copy_to(self, model: torch.nn.Module) -> None:
        """Overwrite the model's parameters with the EMA weights (in place)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    @contextlib.contextmanager
    def averaged(self, model: torch.nn.Module):
        """Context manager: swap EMA weights in for the body, restore on exit."""
        backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = copy.deepcopy(state["shadow"])
