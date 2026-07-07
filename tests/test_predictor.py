"""Tests for phlogiston.models.predictor. Run: python -m tests.test_predictor"""

from __future__ import annotations

import sys

import torch

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.predictor import PREDICT_KEYS, STABILITY_KEYS, Predictor

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _batch(n=8):
    ds = ShardedCrystalDataset("data", max_shards=1)
    return collate([ds[i] for i in range(n)])


def _model():
    # small + ν=1 for a fast test
    return Predictor(mul=16, n_layers=2, correlation=1)


def test_forward_shape():
    b = _batch(8)
    model = _model()
    pred = model(b)
    ok = pred.shape == (8, len(PREDICT_KEYS)) and torch.isfinite(pred).all()
    _check("forward shape", bool(ok), str(tuple(pred.shape)))


def test_masked_loss_ignores_absent():
    b = _batch(6)
    model = _model()
    pred = model(b)
    y, mask = model.slice_targets(b.y, b.y_mask)
    # pick a target and force it fully masked-out; changing its label must not
    # change the loss.
    t = PREDICT_KEYS.index("bulk_modulus_vrh")
    mask = mask.clone()
    mask[:, t] = False
    l1, _ = model.loss(pred, y, mask)
    y2 = y.clone()
    y2[:, t] += 1234.0
    l2, _ = model.loss(pred, y2, mask)
    _check(
        "masked target value does not affect loss",
        torch.allclose(l1, l2, atol=1e-8),
        f"{l1.item():.4f} vs {l2.item():.4f}",
    )


def test_stage_param_groups():
    model = _model()
    stage1 = {id(p) for p in model.stage1_parameters()}
    prop_params = {id(p) for i in model._property_idx for p in model.heads[i].parameters()}
    stab_params = {id(p) for i in model._stability_idx for p in model.heads[i].parameters()}
    enc_params = {id(p) for p in model.encoder.parameters()}
    ok = (stab_params <= stage1) and enc_params <= stage1 and prop_params.isdisjoint(stage1)
    _check("stage1 = encoder + stability heads only", bool(ok))

    groups = model.stage2_param_groups(encoder_lr=1e-4, head_lr=1e-3)
    lrs = {g["lr"] for g in groups}
    _check("stage2 groups have distinct encoder/head LRs", lrs == {1e-4, 1e-3})


def test_backward_runs_and_grads():
    b = _batch(6)
    model = _model()
    pred = model(b)
    y, mask = model.slice_targets(b.y, b.y_mask)
    # stage-1 style: only stability targets contribute
    m = torch.zeros_like(mask)
    for k in STABILITY_KEYS:
        m[:, PREDICT_KEYS.index(k)] = mask[:, PREDICT_KEYS.index(k)]
    total, per = model.loss(pred, y, m)
    total.backward()
    enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()
    )
    _check("stage-1 loss backprops into encoder", enc_grad, f"loss={total.item():.4f}")


if __name__ == "__main__":
    test_forward_shape()
    test_masked_loss_ignores_absent()
    test_stage_param_groups()
    test_backward_runs_and_grads()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
