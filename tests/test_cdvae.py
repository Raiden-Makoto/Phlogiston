"""Tests for the assembled CDVAE. Run: python -m tests.test_cdvae"""

from __future__ import annotations

import sys

import torch

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.cdvae import CDVAE

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _batch(n=6):
    ds = ShardedCrystalDataset("data", max_shards=1)
    return collate([ds[i] for i in range(n)])


def _model():
    return CDVAE(
        latent_dim=16, mul=16, n_max=64, n_elements=100, n_levels=10, n_layers=2, correlation=1
    )


def test_training_loss():
    b = _batch(6)
    m = _model()
    total, parts = m.training_loss(b)
    finite = torch.isfinite(total) and all(torch.isfinite(v) for v in parts.values())
    _check(
        "training loss finite",
        bool(finite),
        " ".join(f"{k}={v.item():.2f}" for k, v in parts.items()),
    )
    _check(
        "all loss components present",
        set(parts) == {"kl", "num", "lattice", "composition", "coord", "type"},
    )


def test_backward_reaches_submodules():
    b = _batch(6)
    m = _model()
    total, _ = m.training_loss(b)
    total.backward()

    def has_grad(mod):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())

    _check("grad -> encoder", has_grad(m.encoder))
    _check("grad -> predictors", has_grad(m.predictors))
    _check("grad -> decoder", has_grad(m.decoder))


def test_generate_prototype():
    m = _model()
    cubic = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    struct = m.generate(n_atoms=8, lattice=cubic, steps_per_level=1)
    ok = len(struct) == 8 and abs(struct.lattice.a - 5.0) < 1e-6
    _check(
        "generate returns a structure of requested size",
        bool(ok),
        f"N={len(struct)} a={struct.lattice.a:.2f}",
    )


if __name__ == "__main__":
    test_training_loss()
    test_backward_reaches_submodules()
    test_generate_prototype()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
