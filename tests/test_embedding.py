"""Tests for phlogiston.layers.embedding. Run: python -m tests.test_embedding"""

from __future__ import annotations

import sys

import torch

from phlogiston.layers import AtomEmbedding

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_shape():
    emb = AtomEmbedding(mul=128)
    z = torch.tensor([1, 6, 8, 26, 26])
    h = emb(z)
    _check("embedding shape", h.shape == (5, 128), str(tuple(h.shape)))


def test_same_species_same_row():
    emb = AtomEmbedding(mul=64)
    z = torch.tensor([26, 8, 26])            # two Fe, one O
    h = emb(z)
    _check("identical z -> identical rows", torch.allclose(h[0], h[2]))
    _check("different z -> different rows", not torch.allclose(h[0], h[1]))


def test_permutation_consistency():
    emb = AtomEmbedding(mul=32)
    z = torch.tensor([1, 6, 8, 26])
    h = emb(z)
    perm = torch.tensor([2, 0, 3, 1])
    _check("permutation consistency", torch.allclose(emb(z[perm]), h[perm]))


def test_irreps():
    emb = AtomEmbedding(mul=48)
    _check("irreps_out == 48x0e", str(emb.irreps_out) == "48x0e", str(emb.irreps_out))


if __name__ == "__main__":
    test_shape()
    test_same_species_same_row()
    test_permutation_consistency()
    test_irreps()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
