"""Tests for the batched sampler coordinate transforms.

Regression guard for the cartesian<->fractional round trip: an earlier version
transposed the inverse lattice (``(L^-1)_ij`` instead of ``(L^-1)_ji``), which is
a no-op only for symmetric cells but silently scrambles triclinic coordinates.

Run: python -m tests.test_cdvae_sampler
"""

from __future__ import annotations

import sys

import torch

from phlogiston.models.cdvae.sampler import cart_to_frac, frac_to_cart

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _random_triclinic(n: int) -> torch.Tensor:
    """A batch of well-conditioned, deliberately non-symmetric lattices."""
    torch.manual_seed(0)
    lat = torch.randn(n, 3, 3) * 2.0
    lat = lat + torch.eye(3) * 6.0  # bias diagonal so cells stay invertible
    return lat


def test_roundtrip_frac_cart_frac():
    lat = _random_triclinic(16)
    frac = torch.rand(16, 3)
    back = cart_to_frac(frac_to_cart(frac, lat), lat)
    err = (back - frac).abs().max().item()
    _check("frac -> cart -> frac recovers input", err < 1e-5, f"max_err={err:.2e}")


def test_roundtrip_cart_frac_cart():
    lat = _random_triclinic(16)
    cart = torch.randn(16, 3) * 3.0
    back = frac_to_cart(cart_to_frac(cart, lat), lat)
    err = (back - cart).abs().max().item()
    _check("cart -> frac -> cart recovers input", err < 1e-5, f"max_err={err:.2e}")


def test_inverse_orientation_matters():
    """The wrong-orientation inverse must actually differ on a triclinic cell,
    otherwise this test could not catch the original bug."""
    lat = _random_triclinic(4)
    cart = torch.randn(4, 3) * 3.0
    inv = torch.linalg.inv(lat)
    correct = torch.einsum("nj,nji->ni", cart, inv)  # (L^-1)_ji  -- right
    wrong = torch.einsum("nj,nij->ni", cart, inv)  # (L^-1)_ij  -- old bug
    diff = (correct - wrong).abs().max().item()
    _check("transposed inverse is genuinely different (triclinic)", diff > 1e-2, f"diff={diff:.3f}")


if __name__ == "__main__":
    test_roundtrip_frac_cart_frac()
    test_roundtrip_cart_frac_cart()
    test_inverse_orientation_matters()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
