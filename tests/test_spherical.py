"""Tests for phlogiston.layers.spherical. Run: python -m tests.test_spherical"""

from __future__ import annotations

import sys

import torch

from phlogiston.layers import SphericalHarmonics
from tests.equivariance import assert_equivariant

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_equivariance():
    sh = SphericalHarmonics(l_max=3).double()
    # edge vectors transform as a 1o (vector) input.
    err = assert_equivariant(lambda x: sh(x), "1o", sh.irreps_out, atol=1e-5)
    _check("spherical equivariance (rotation + inversion)", True, f"err={err:.2e}")


def test_shape():
    sh = SphericalHarmonics(l_max=3)
    y = sh(torch.randn(10, 3))
    _check("output shape", y.shape == (10, sh.irreps_out.dim),
           f"{tuple(y.shape)} vs (10,{sh.irreps_out.dim})")


def test_l0_invariance():
    # component normalization => the l=0 component is the constant 1 for every
    # direction (rotationally invariant).
    sh = SphericalHarmonics(l_max=2)
    y = sh(torch.randn(32, 3))
    _check("l=0 block == 1 (invariant)", torch.allclose(y[:, 0], torch.ones(32), atol=1e-5))


def test_normalize_direction_only():
    # normalize=True => output depends only on direction, not magnitude.
    sh = SphericalHarmonics(l_max=3)
    v = torch.randn(20, 3)
    _check("normalize: magnitude-independent", torch.allclose(sh(v), sh(3.7 * v), atol=1e-5))


if __name__ == "__main__":
    test_equivariance()
    test_shape()
    test_l0_invariance()
    test_normalize_direction_only()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
