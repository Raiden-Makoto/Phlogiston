"""Reusable O(3)-equivariance test harness for layers.

The contract every equivariant layer must satisfy: for a rotation/inversion
``g`` with input rep ``D_in`` and output rep ``D_out``,

    f(D_in(g) · x)  ==  D_out(g) · f(x)

so scalars (`0e`) stay put, vectors (`1o`) rotate, etc. `assert_equivariant`
checks this numerically with a random rotation, in float64 for a tight bound.
Used by every ``tests/test_<layer>.py``.
"""

from __future__ import annotations

import torch
from e3nn import o3


def assert_equivariant(
    f,
    irreps_in,
    irreps_out,
    *,
    n: int = 16,
    atol: float = 1e-5,
    seed: int = 0,
    include_inversion: bool = True,
    dtype=torch.float64,
) -> float:
    """Assert ``f`` is equivariant from ``irreps_in`` to ``irreps_out``.

    ``f`` takes a tensor ``[n, irreps_in.dim]`` (e.g. edge vectors are ``1o``).
    Returns the max abs equivariance error.
    """
    irreps_in = o3.Irreps(irreps_in)
    irreps_out = o3.Irreps(irreps_out)
    torch.manual_seed(seed)

    x = irreps_in.randn(n, -1).to(dtype)
    # random rotation (+ optional inversion to also probe parity)
    rot = o3.rand_matrix().to(dtype)
    g = -rot if include_inversion else rot
    D_in = irreps_in.D_from_matrix(g).to(dtype)
    D_out = irreps_out.D_from_matrix(g).to(dtype)

    y_rot_then_f = f(x @ D_in.transpose(-1, -2))
    y_f_then_rot = f(x) @ D_out.transpose(-1, -2)
    err = (y_rot_then_f - y_f_then_rot).abs().max().item()
    assert err < atol, (
        f"equivariance broken: max err {err:.3e} >= {atol} ({irreps_in} -> {irreps_out})"
    )
    return err
