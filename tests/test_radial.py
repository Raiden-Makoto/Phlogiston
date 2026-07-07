"""Tests for phlogiston.layers.radial. Run: python -m tests.test_radial"""

from __future__ import annotations

import sys

import torch

from phlogiston.layers.radial import BesselBasis, PolynomialCutoff, RadialBasis

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_shape():
    rb = RadialBasis(n_out=32, r_max=6.0)
    w = rb(torch.rand(50) * 6.0)
    _check("radial output shape", w.shape == (50, 32), str(tuple(w.shape)))


def test_zero_beyond_cutoff():
    rb = RadialBasis(n_out=16, r_max=6.0)
    d = torch.tensor([6.0, 6.5, 10.0])
    w = rb(d)
    _check("radial == 0 at/beyond cutoff", torch.allclose(w, torch.zeros_like(w), atol=1e-6))


def test_cutoff_continuity():
    # envelope value and slope -> 0 at r_max (C¹): just-inside value is tiny.
    cut = PolynomialCutoff(r_max=6.0, p=6)
    just_inside = cut(torch.tensor([5.999]))
    at = cut(torch.tensor([6.0]))
    _check("cutoff continuous -> 0 at r_max",
           bool(at.abs().item() < 1e-6 and just_inside.item() < 1e-2),
           f"just_inside={just_inside.item():.2e}")


def test_bessel_finite_small_d():
    b = BesselBasis(r_max=6.0, n_bessel=8)
    v = b(torch.tensor([1e-3, 0.5, 2.0]))
    _check("bessel finite at small d", bool(torch.isfinite(v).all()))


def test_invariant_wrt_geometry():
    # radial is a function of the scalar distance only -> identical for identical
    # lengths regardless of direction (invariance is trivial but worth asserting).
    rb = RadialBasis(n_out=8, r_max=6.0)
    d = torch.tensor([2.5, 2.5, 4.1])
    w = rb(d)
    _check("equal lengths -> equal weights", torch.allclose(w[0], w[1], atol=1e-6))


if __name__ == "__main__":
    test_shape()
    test_zero_beyond_cutoff()
    test_cutoff_continuity()
    test_bessel_finite_small_d()
    test_invariant_wrt_geometry()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
