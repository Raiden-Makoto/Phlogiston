"""Tests for CDVAE-specific layers. Run: python -m tests.test_cdvae_layers"""

from __future__ import annotations

import sys

import torch

from phlogiston.layers import EquivariantVectorReadout, NoiseEmbedding
from tests.equivariance import assert_equivariant

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_noise_embedding():
    emb = NoiseEmbedding(dim=64)
    sigma = torch.tensor([0.01, 0.1, 1.0, 10.0])
    e = emb(sigma)
    _check("noise embedding shape", e.shape == (4, 64), str(tuple(e.shape)))
    _check("different sigma -> different embedding", not torch.allclose(e[0], e[3]))
    _check("deterministic", torch.allclose(emb(sigma), e))
    _check(
        "finite + bounded (sin/cos)", bool(torch.isfinite(e).all() and e.abs().max() <= 1.0 + 1e-6)
    )


def test_vector_readout_equivariance():
    irreps_in = "8x0e+4x1o+2x2e"
    ro = EquivariantVectorReadout(irreps_in, n_vectors=1).double()
    # output is a 1o vector -> must rotate with the input (equivariant)
    err = assert_equivariant(lambda x: ro(x), irreps_in, "1x1o", atol=1e-5)
    _check("vector readout equivariance (1o)", True, f"err={err:.2e}")


def test_vector_readout_shape():
    ro = EquivariantVectorReadout("8x0e+4x1o+2x2e", n_vectors=1)
    from e3nn import o3

    x = o3.Irreps("8x0e+4x1o+2x2e").randn(10, -1)
    _check("vector readout shape", ro(x).shape == (10, 3), str(tuple(ro(x).shape)))


if __name__ == "__main__":
    test_noise_embedding()
    test_vector_readout_equivariance()
    test_vector_readout_shape()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
