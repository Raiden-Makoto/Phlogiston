"""Tiny grid hyperparameter sweep (no framework needed).

Every hyperparameter is a `train()` kwarg and every run saves `best_val` in its
`_best.pt`, so a sweep is just: run a list of configs on a subset, read back
`best_val`, and rank. Two coarse passes:

  python scripts/sweep.py --phase lr        --max-shards 8 --epochs 15
  python scripts/sweep.py --phase capacity  --max-shards 8 --epochs 15 --best-lr 1e-3

Pin to a free GPU with HIP_VISIBLE_DEVICES=6 so it runs beside the baseline.
"""

from __future__ import annotations

import argparse

import torch

from phlogiston.train import train

# fixed base config; overridden per grid point
BASE = dict(
    stage=1,
    batch_size=128,
    mul=128,
    n_layers=2,
    correlation=2,
    lr=1e-3,
    encoder_lr=1e-4,
    warmup_epochs=2,
)


def lr_grid():
    return [{"lr": lr} for lr in (3e-4, 1e-3, 3e-3)]


def capacity_grid(best_lr: float):
    return [
        {"lr": best_lr, **c}
        for c in (
            {"mul": 96, "correlation": 2, "n_layers": 2},
            {"mul": 128, "correlation": 2, "n_layers": 2},
            {"mul": 192, "correlation": 2, "n_layers": 2},
            {"mul": 128, "correlation": 3, "n_layers": 2},
            {"mul": 128, "correlation": 2, "n_layers": 3},
        )
    ]


def _name(override: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in override.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["lr", "capacity"], required=True)
    ap.add_argument("--max-shards", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--best-lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default="data/runs/sweep")
    args = ap.parse_args()

    grid = lr_grid() if args.phase == "lr" else capacity_grid(args.best_lr)
    results = []
    for override in grid:
        name = _name(override)
        out_dir = f"{args.out_root}/{args.phase}/{name}"
        cfg = {**BASE, **override}
        print(f"\n===== sweep[{args.phase}] {name} =====", flush=True)
        train(
            data_root="data",
            out_dir=out_dir,
            max_shards=args.max_shards,
            epochs=args.epochs,
            patience=args.epochs + 1,  # no early stop in short runs
            **cfg,
        )
        best = torch.load(f"{out_dir}/predictor_stage1_best.pt", map_location="cpu")
        results.append((name, best["best_val"]))

    results.sort(key=lambda x: x[1])
    print(f"\n===== sweep[{args.phase}] ranking (best val_loss) =====", flush=True)
    for rank, (name, val) in enumerate(results, 1):
        print(f"  {rank}. {name:40s} val_loss={val:.4f}", flush=True)


if __name__ == "__main__":
    main()
