"""Tests for the CDVAE score decoder. Run: python -m tests.test_cdvae_decoder

The decoder is conditioned on invariant (z, sigma) and must be equivariant in
geometry: rotating the structure rotates the coord score (1o) and leaves the
type logits (invariant) unchanged.
"""

from __future__ import annotations

import sys

import torch
from e3nn import o3

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.cdvae import CDVAEDecoder

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _batch(n=3, dtype=torch.float64):
    ds = ShardedCrystalDataset("data", max_shards=1)
    b = collate([ds[i] for i in range(n)])
    b.edge_vec = b.edge_vec.to(dtype)
    b.edge_len = b.edge_len.to(dtype)
    return b


def test_shapes():
    b = _batch(3, torch.float32)
    dec = CDVAEDecoder(latent_dim=16, n_elements=100, mul=16, n_layers=2, correlation=1)
    z = torch.randn(3, 16)
    sigma = torch.rand(3)
    out = dec(b, z, sigma)
    n = b.z.shape[0]
    ok = out.coord_score.shape == (n, 3) and out.type_logits.shape == (n, 100)
    _check(
        "decoder output shapes",
        bool(ok),
        f"score={tuple(out.coord_score.shape)} types={tuple(out.type_logits.shape)}",
    )


def test_equivariance():
    b = _batch(3, torch.float64)
    dec = CDVAEDecoder(latent_dim=16, n_elements=100, mul=16, n_layers=2, correlation=1).double()
    z = torch.randn(3, 16, dtype=torch.float64)
    sigma = torch.rand(3, dtype=torch.float64)

    out = dec(b, z, sigma)
    g = -o3.rand_matrix().double()  # rotation + inversion
    b.edge_vec = b.edge_vec @ g.T  # rotate geometry (z, sigma invariant)
    out_rot = dec(b, z, sigma)

    score_err = (out_rot.coord_score - out.coord_score @ g.T).abs().max().item()
    type_err = (out_rot.type_logits - out.type_logits).abs().max().item()
    _check("coord score equivariant (1o)", score_err < 1e-5, f"err={score_err:.2e}")
    _check("type logits invariant", type_err < 1e-5, f"err={type_err:.2e}")


if __name__ == "__main__":
    test_shapes()
    test_equivariance()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
