"""End-to-end encoder tests. Run: python -m tests.test_encoder

Integrates interaction v1 into the full CrystalEncoder and checks it on REAL
precomputed graphs (requires data/processed on the box), plus invariance of the
scalar output under a global rotation of the geometry.
"""

from __future__ import annotations

import sys

import torch
from e3nn import o3

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.encoder import CrystalEncoder

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _batch(n=8, dtype=torch.float32):
    # load just one shard for a fast test (full corpus is ~30 GB)
    ds = ShardedCrystalDataset("data", max_shards=1)
    b = collate([ds[i] for i in range(n)])
    if dtype != torch.float32:
        b.pos = b.pos.to(dtype)
        b.edge_vec = b.edge_vec.to(dtype)
        b.edge_len = b.edge_len.to(dtype)
        b.lattice = b.lattice.to(dtype)
    return b, len(ds)


def test_integration_real_graphs():
    b, size = _batch(8)
    enc = CrystalEncoder(mul=32, n_layers=2)
    out = enc(b)
    n = b.z.shape[0]
    ok = (
        out.node_feats.shape == (n, 32)
        and out.graph_feats.shape == (8, 32)
        and torch.isfinite(out.node_feats).all()
    )
    _check(
        f"encoder runs on real graphs (dataset size {size:,})",
        bool(ok),
        f"node={tuple(out.node_feats.shape)} graph={tuple(out.graph_feats.shape)}",
    )


def test_output_invariance():
    b, _ = _batch(4, dtype=torch.float64)
    enc = CrystalEncoder(mul=32, n_layers=2).double()
    out1 = enc(b)
    # rotate + invert the geometry; scalar output must be unchanged
    g = -o3.rand_matrix().double()
    b.edge_vec = b.edge_vec @ g.T
    out2 = enc(b)
    err = (out1.node_feats - out2.node_feats).abs().max().item()
    _check("encoder output invariant under O(3)", err < 1e-5, f"err={err:.2e}")


def test_batch_independence():
    # a graph's features must not depend on what else is in the batch.
    b1, _ = _batch(1)
    b8, _ = _batch(8)
    enc = CrystalEncoder(mul=16, n_layers=2)
    torch.manual_seed(0)
    o1 = enc(b1).node_feats
    n0 = int((b8.batch == 0).sum())
    o8 = enc(b8).node_feats[:n0]
    _check(
        "first graph features independent of batch",
        torch.allclose(o1, o8, atol=1e-5),
        f"maxdiff={(o1 - o8).abs().max().item():.2e}",
    )


if __name__ == "__main__":
    test_integration_real_graphs()
    test_output_invariance()
    test_batch_independence()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
