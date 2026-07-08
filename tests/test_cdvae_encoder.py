"""Tests for CDVAE encoder + latent predictors. Run: python -m tests.test_cdvae_encoder"""

from __future__ import annotations

import sys

import torch

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.cdvae import CDVAEEncoder, LatentPredictors

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _batch(n=8):
    ds = ShardedCrystalDataset("data", max_shards=1)
    return collate([ds[i] for i in range(n)])


def test_encoder_shapes_and_kl():
    b = _batch(8)
    enc = CDVAEEncoder(latent_dim=32, mul=16, n_layers=2, correlation=1)
    out = enc(b)
    ok = out.z.shape == (8, 32) and out.mu.shape == (8, 32) and out.logvar.shape == (8, 32)
    _check("VAE output shapes", bool(ok), f"z={tuple(out.z.shape)}")
    kl = CDVAEEncoder.kl_loss(out.mu, out.logvar)
    _check(
        "KL finite and >= 0", bool(torch.isfinite(kl) and kl.item() >= -1e-6), f"kl={kl.item():.3f}"
    )


def test_reparameterization_stochastic():
    b = _batch(4)
    enc = CDVAEEncoder(latent_dim=32, mul=16, n_layers=2, correlation=1)
    o1, o2 = enc(b), enc(b)
    _check("mu deterministic across calls", torch.allclose(o1.mu, o2.mu, atol=1e-5))
    _check("z stochastic (reparam noise)", not torch.allclose(o1.z, o2.z))


def test_latent_predictors():
    z = torch.randn(5, 32)
    lp = LatentPredictors(latent_dim=32, n_max=64, n_elements=100)
    p = lp(z)
    ok = (
        p.num_atoms_logits.shape == (5, 64)
        and p.lattice.shape == (5, 6)
        and p.composition_logits.shape == (5, 100)
    )
    _check(
        "latent predictor shapes",
        bool(ok),
        f"N={tuple(p.num_atoms_logits.shape)} L={tuple(p.lattice.shape)} C={tuple(p.composition_logits.shape)}",
    )


def test_gradient_flow():
    b = _batch(4)
    enc = CDVAEEncoder(latent_dim=32, mul=16, n_layers=2, correlation=1)
    lp = LatentPredictors(latent_dim=32)
    out = enc(b)
    p = lp(out.z)
    loss = (
        CDVAEEncoder.kl_loss(out.mu, out.logvar)
        + p.lattice.pow(2).mean()
        + p.num_atoms_logits.pow(2).mean()
        + p.composition_logits.pow(2).mean()
    )
    loss.backward()
    enc_grad = any(
        g is not None and g.abs().sum() > 0 for g in (pp.grad for pp in enc.parameters())
    )
    lp_grad = any(g is not None and g.abs().sum() > 0 for g in (pp.grad for pp in lp.parameters()))
    _check("gradients reach encoder + predictors", enc_grad and lp_grad)


if __name__ == "__main__":
    test_encoder_shapes_and_kl()
    test_reparameterization_stochastic()
    test_latent_predictors()
    test_gradient_flow()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
