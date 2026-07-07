"""Tests for phlogiston.layers.interaction (v1). Run: python -m tests.test_interaction

The interaction block has structured multi-inputs (node features + geometry), so
we test equivariance directly: rotate node features by their Wigner-D AND rotate
edge vectors (recomputing spherical harmonics), then assert the output rotates by
its Wigner-D. Inversion is included via a full O(3) matrix.
"""

from __future__ import annotations

import sys

import torch
from e3nn import o3

from phlogiston.layers import Interaction, SphericalHarmonics

_results: list[tuple[str, bool, str]] = []
IRREPS_IN = "4x0e+4x1o+2x2e"
IRREPS_OUT = "4x0e+4x1o+2x2e"


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _random_graph(n=6, e=20, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.randint(0, n, (2, e), generator=g)
    edge_vec = torch.randn(e, 3, generator=g, dtype=dtype)
    edge_len = edge_vec.norm(dim=1)
    z = torch.randint(1, 30, (n,), generator=g)
    return n, edge_index, edge_vec, edge_len, z


def _build(dtype=torch.float64):
    sh = SphericalHarmonics(l_max=3)
    inter = Interaction(IRREPS_IN, sh.irreps_out, IRREPS_OUT, l_feat=2).to(dtype)
    return sh, inter


def test_equivariance():
    dtype = torch.float64
    sh, inter = _build(dtype)
    n, edge_index, edge_vec, edge_len, z = _random_graph(dtype=dtype)
    h = o3.Irreps(IRREPS_IN).randn(n, -1).to(dtype)
    navg = edge_index.shape[1] / n

    out = inter(h, edge_index, edge_len, sh(edge_vec), z, navg)

    g = -o3.rand_matrix().to(dtype)                       # rotation + inversion
    D_in = o3.Irreps(IRREPS_IN).D_from_matrix(g).to(dtype)
    D_out = o3.Irreps(IRREPS_OUT).D_from_matrix(g).to(dtype)

    h_rot = h @ D_in.T
    edge_vec_rot = edge_vec @ g.T                         # geometry transforms
    out_rot = inter(h_rot, edge_index, edge_len, sh(edge_vec_rot), z, navg)

    err = (out_rot - out @ D_out.T).abs().max().item()
    _check("interaction equivariance (rotation+inversion)", err < 1e-5, f"err={err:.2e}")


def test_scalars_in_grow_higher_l():
    # layer-0 case: scalar-only input should produce ℓ>0 features via the SH TP.
    sh = SphericalHarmonics(l_max=3)
    inter = Interaction("8x0e", sh.irreps_out, "8x0e+8x1o+4x2e", l_feat=2).double()
    n, edge_index, edge_vec, edge_len, z = _random_graph(dtype=torch.float64)
    h = o3.Irreps("8x0e").randn(n, -1).double()
    out = inter(h, edge_index, edge_len, sh(edge_vec), z, edge_index.shape[1] / n)
    # the 1o block (cols 8..8+24) should be non-zero
    l1 = out[:, 8:8 + 8 * 3]
    _check("scalar input -> nonzero ℓ=1 output", l1.abs().max().item() > 1e-6)


def test_shape():
    sh = SphericalHarmonics(l_max=3)
    inter = Interaction(IRREPS_IN, sh.irreps_out, IRREPS_OUT, l_feat=2)
    n, edge_index, edge_vec, edge_len, z = _random_graph(dtype=torch.float32)
    out = inter(o3.Irreps(IRREPS_IN).randn(n, -1), edge_index, edge_len,
                sh(edge_vec), z, edge_index.shape[1] / n)
    _check("output shape", out.shape == (n, o3.Irreps(IRREPS_OUT).dim),
           f"{tuple(out.shape)}")


if __name__ == "__main__":
    test_equivariance()
    test_scalars_in_grow_higher_l()
    test_shape()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
