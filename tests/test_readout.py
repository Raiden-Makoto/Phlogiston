"""Tests for phlogiston.layers.readout. Run: python -m tests.test_readout"""

from __future__ import annotations

import sys

import torch
from e3nn import o3

from phlogiston.layers import ScalarReadout
from tests.equivariance import assert_equivariant

_results: list[tuple[str, bool, str]] = []
IRREPS = "8x0e+4x1o+2x2e"


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_invariance():
    ro = ScalarReadout(IRREPS, n_out=1, hidden=(16,)).double()
    # output is 0e (invariant); harness with irreps_out = "1x0e"
    err = assert_equivariant(lambda x: ro(x), IRREPS, "1x0e", atol=1e-5)
    _check("readout invariance", True, f"err={err:.2e}")


def test_pool_sum_extensive():
    ro = ScalarReadout(IRREPS, n_out=1, reduce="sum")
    x = o3.Irreps(IRREPS).randn(6, -1)
    # two graphs: graph 0 = rows 0..2, graph 1 = an exact copy -> pooled equal
    x2 = torch.cat([x[:3], x[:3]], dim=0)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    g = ro(x2, batch)
    _check(
        "sum pooling: identical graphs -> equal totals",
        torch.allclose(g[0], g[1], atol=1e-5),
        f"{g.squeeze().tolist()}",
    )


def test_shape():
    ro = ScalarReadout(IRREPS, n_out=3)
    r = ro(o3.Irreps(IRREPS).randn(10, -1))
    _check("per-atom readout shape", r.shape == (10, 3), str(tuple(r.shape)))


if __name__ == "__main__":
    test_invariance()
    test_pool_sum_extensive()
    test_shape()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
