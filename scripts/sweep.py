"""Tiny grid hyperparameter sweep (no framework needed).

Every hyperparameter is a `train()` kwarg and every run saves its selection
metric in `best_val` in the stage's `_best.pt`, so a sweep is just: run a list of
configs, read back `best_val`, and rank. (For stage 2 `best_val` = -mean property
R², so lower is better and ranking directly maximizes property R².)

Stage 1 (encoder + stability), coarse passes on a shard subset:
  python scripts/sweep.py --stage 1 --phase lr       --max-shards 8 --epochs 15
  python scripts/sweep.py --stage 1 --phase capacity --max-shards 8 --epochs 15 --best-lr 1e-3

Stage 2 (property fine-tuning). The encoder architecture is fixed by the
stage-1 checkpoint, so we tune the *fine-tune* knobs. `--restrict-labeled` trains
on just the ~12k property-labeled structures (fast, dense signal):
  python scripts/sweep.py --stage 2 --phase enc_lr --init-ckpt data/runs/predictor_stage1_best.pt --epochs 20
  python scripts/sweep.py --stage 2 --phase head_lr --init-ckpt ... --best-enc-lr 1e-4 --epochs 20
  python scripts/sweep.py --stage 2 --phase wd --init-ckpt ... --best-enc-lr 1e-4 --best-lr 1e-3 --epochs 20

Pin to a free GPU with HIP_VISIBLE_DEVICES=N so it runs beside a baseline.
"""

from __future__ import annotations

import argparse

import torch

from phlogiston.train import train

# fixed base config; overridden per grid point. Capacity must match the stage-1
# checkpoint when fine-tuning (stage 2 warm-starts its weights).
BASE = dict(batch_size=128, mul=128, n_layers=2, correlation=2, warmup_epochs=2)


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


# --- stage-2 fine-tune grids (encoder architecture fixed) -------------------
def enc_lr_grid():
    # 0.0 = frozen encoder (heads only); how hard to fine-tune is THE stage-2 knob
    return [{"encoder_lr": e} for e in (0.0, 3e-5, 1e-4, 3e-4, 1e-3)]


def head_lr_grid(best_enc_lr: float):
    return [{"encoder_lr": best_enc_lr, "lr": lr} for lr in (3e-4, 1e-3, 3e-3)]


def wd_grid(best_enc_lr: float, best_lr: float):
    return [
        {"encoder_lr": best_enc_lr, "lr": best_lr, "weight_decay": wd}
        for wd in (1e-5, 1e-4, 1e-3)
    ]


def _name(override: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in override.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--phase", required=True, choices=["lr", "capacity", "enc_lr", "head_lr", "wd"])
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--best-lr", type=float, default=1e-3)
    ap.add_argument("--best-enc-lr", type=float, default=1e-4)
    ap.add_argument("--init-ckpt", default=None, help="Stage-1 checkpoint (required for stage 2)")
    ap.add_argument(
        "--restrict-labeled",
        action="store_true",
        default=None,
        help="Restrict to the stage's labeled subset (auto-on for stage 2)",
    )
    ap.add_argument("--out-root", default="data/runs/sweep")
    args = ap.parse_args()

    if args.stage == 2 and not args.init_ckpt:
        ap.error("--init-ckpt (stage-1 best) is required for stage 2 sweeps")
    restrict = args.restrict_labeled if args.restrict_labeled is not None else (args.stage == 2)

    grids = {
        "lr": lambda: lr_grid(),
        "capacity": lambda: capacity_grid(args.best_lr),
        "enc_lr": lambda: enc_lr_grid(),
        "head_lr": lambda: head_lr_grid(args.best_enc_lr),
        "wd": lambda: wd_grid(args.best_enc_lr, args.best_lr),
    }
    grid = grids[args.phase]()

    results = []
    for override in grid:
        name = _name(override)
        out_dir = f"{args.out_root}/stage{args.stage}/{args.phase}/{name}"
        cfg = {**BASE, **override}
        print(f"\n===== sweep[s{args.stage}/{args.phase}] {name} =====", flush=True)
        train(
            data_root="data",
            stage=args.stage,
            out_dir=out_dir,
            max_shards=args.max_shards,
            epochs=args.epochs,
            patience=args.epochs + 1,  # no early stop in short runs
            init_ckpt=args.init_ckpt,
            restrict_labeled=restrict,
            **cfg,
        )
        best = torch.load(
            f"{out_dir}/predictor_stage{args.stage}_best.pt", map_location="cpu"
        )
        results.append((name, best["best_val"]))

    metric = "-mean_R2" if args.stage == 2 else "val_loss"
    results.sort(key=lambda x: x[1])  # lower is better for both metrics
    print(f"\n===== sweep[s{args.stage}/{args.phase}] ranking (best {metric}) =====", flush=True)
    for rank, (name, val) in enumerate(results, 1):
        extra = f"  (mean R2={-val:.4f})" if args.stage == 2 else ""
        print(f"  {rank}. {name:40s} {metric}={val:.4f}{extra}", flush=True)


if __name__ == "__main__":
    main()
