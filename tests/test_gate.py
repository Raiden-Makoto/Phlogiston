"""Tests for phlogiston.layers.gate. Run: python -m tests.test_gate"""

from __future__ import annotations

import sys

from phlogiston.layers import EquivariantGate
from tests.equivariance import assert_equivariant

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_equivariance():
    gate = EquivariantGate("16x0e+8x1o+4x2e").double()
    err = assert_equivariant(lambda x: gate(x), gate.irreps_in, gate.irreps_out, atol=1e-5)
    _check("gate equivariance", True, f"err={err:.2e}")


def test_irreps_contract():
    gate = EquivariantGate("16x0e+8x1o+4x2e")
    # input = scalars + gates + gated; output = scalars + gated
    _check(
        "irreps_in has gate scalars",
        gate.irreps_in.num_irreps > gate.irreps_out.num_irreps,
        f"in={gate.irreps_in} out={gate.irreps_out}",
    )
    _check("irreps_out keeps ℓ>0", any(ir.l > 0 for _, ir in gate.irreps_out))


if __name__ == "__main__":
    test_equivariance()
    test_irreps_contract()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
