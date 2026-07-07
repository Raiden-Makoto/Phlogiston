"""Tests for phlogiston.layers linear layers. Run: python -m tests.test_linear"""

from __future__ import annotations

import sys

import torch

from phlogiston.layers import EquivariantLinear, SpeciesLinear
from tests.equivariance import assert_equivariant

_results: list[tuple[str, bool, str]] = []
IN, OUT = "4x0e+4x1o+2x2e", "8x0e+4x1o+2x2e"


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_equivariant_linear():
    lin = EquivariantLinear(IN, OUT).double()
    err = assert_equivariant(lambda x: lin(x), IN, OUT, atol=1e-5)
    _check("EquivariantLinear equivariance", True, f"err={err:.2e}")


def test_species_linear_equivariance():
    sl = SpeciesLinear(IN, OUT).double()
    z = torch.randint(1, 30, (16,))
    err = assert_equivariant(lambda x: sl(x, z), IN, OUT, atol=1e-5)
    _check("SpeciesLinear equivariance (fixed z)", True, f"err={err:.2e}")


def test_species_selectivity():
    sl = SpeciesLinear(IN, OUT).double()
    from e3nn import o3
    x = o3.Irreps(IN).randn(1, -1).double().repeat(2, 1)   # identical features
    out_a = sl(x[:1], torch.tensor([6]))
    out_b = sl(x[1:], torch.tensor([26]))
    out_a2 = sl(x[:1], torch.tensor([6]))
    _check("different species -> different map", not torch.allclose(out_a, out_b))
    _check("same species -> same map", torch.allclose(out_a, out_a2))


if __name__ == "__main__":
    test_equivariant_linear()
    test_species_linear_equivariance()
    test_species_selectivity()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
