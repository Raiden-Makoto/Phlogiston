# Training — DESIGN

The schedule-B training loop for the `Predictor` (`phlogiston/train/trainer.py`).
Implements `pipeline.md` §5 and `models/predictor/DESIGN.md` §5.

---

## 0. What it does

Turns the (untrained) `Predictor` into a trained property/stability model in two
stages, on the precomputed graph shards, with data-parallel multi-GPU support.

```
shards ──► ShardedCrystalDataset ──► split(train/val/test)
                                        │
                              normalization (train stats)
                                        │
   ┌── Stage 1 ──────────────────────────────────────────────┐
   │  train encoder + stability heads on ALL graphs           │
   │  (loss masked to formation_energy, energy_above_hull)    │
   └──────────────────────────────────────────────────────────┘
                                        │  warm-start (--init-ckpt)
   ┌── Stage 2 ──────────────────────────────────────────────┐
   │  fine-tune encoder (low LR) + ALL heads on every target  │
   │  (mechanical/thermal grads come only from the ~12k)      │
   └──────────────────────────────────────────────────────────┘
                                        ▼
                              checkpoint (model + norm stats)
```

## 1. Data

- **`ShardedCrystalDataset(data_root, max_shards=None)`** loads precomputed
  graph shards into memory. `max_shards` limits how many (quick runs / tests).
- **`split_indices(n, ratios=(0.8,0.1,0.1), seed)`** — deterministic material-
  level train/val/test split (same seed → same split on every rank).
- **`compute_normalization(dataset, train_idx, pred_idx)`** — masked per-target
  mean/std over the **train split only** (physical units), stored on the model
  as buffers; predictions are de-standardized at inference.

## 2. Stages (schedule B)

| | Stage 1 (pretrain) | Stage 2 (fine-tune) |
|---|---|---|
| Params | `stage1_parameters()` = encoder + stability heads | `stage2_param_groups(encoder_lr, lr)` = low-LR encoder + all heads |
| Loss mask | stability columns only (`_stage_mask`) | all targets |
| Data | all graphs (stability labels ~everywhere) | all graphs (mechanical grads only from labeled ~12k) |
| Start | random init | `--init-ckpt runs/predictor_stage1.pt` |

## 3. Loop mechanics

- **Loss**: `Predictor.loss` — masked multi-task Huber in standardized space
  (per-target mean over present labels).
- **Optimizer**: AdamW; **schedule**: linear LR **warmup** (`--warmup-epochs`)
  then cosine anneal.
- **Best-checkpoint / early stopping**: selection is by a **stage-aware metric**
  (`--select-by`, default stage 1 → stability **AUC**, stage 2 → mean property
  **R²**; `loss` selects by val_loss). Higher-is-better metrics are negated and
  NaN falls back to `val_loss`. `--patience` epochs without improvement on that
  metric stops training (0 disables); `_best.pt` holds the best model.
  Rationale: for stage 2, `val_loss` is diluted by the easy stability targets,
  so a low-loss checkpoint can still be mediocre on the mech/thermal properties.
- **Eval**: `evaluate()` returns, over masked val entries (all-reduced/gathered
  across ranks under DDP):
  - per-target **MAE** in physical units;
  - per-target **R²** (`1 − SS_res/SS_tot`) — flags "predicts the mean" cases
    that MAE alone hides. Computed in the model's **transform space** (log1p for
    `LOG_TARGETS` = Hv, κ; identity otherwise): physical-space R² for a log
    target is dominated by exponential error amplification and is unstable,
    while the transform space is where the target is linear;
  - **stability ROC-AUC + average precision (AP)** for `energy_above_hull`
    (positive class = unstable, i.e. `e_hull > threshold`) — the imbalance-aware
    separation metric, since ~98% of rows are stable and MAE looks small
    regardless.
  These are logged every epoch. `evaluate_checkpoint()` (CLI: `phlogiston
  evaluate --ckpt … --split {val,test}`) scores any saved checkpoint standalone.
- **Checkpoint**: rank 0 saves a **per-epoch** `_last.pt` and a **best-by-val-loss**
  `_best.pt` under `out_dir/predictor_stage{N}_{last,best}.pt`. Each contains
  `{model, stage, mean, std, epoch, optimizer, scheduler, best_val, hparams}` —
  enough to **resume** mid-training (`--resume`) and to reconstruct the
  architecture for standalone eval. `--init-ckpt` instead warm-starts weights
  only (for stage 1 → stage 2).

## 4. Multi-GPU (data-parallel, not tensor-parallel)

The model fits on one 288 GB GPU, so we replicate it and split the batch (DDP)
rather than sharding weights.

```bash
# single GPU
phlogiston train --stage 1 --epochs 30 --batch-size 128

# 2 GPUs (target), up to 4 (max) — RCCL backend on ROCm
torchrun --nproc_per_node=2 -m phlogiston.cli train --stage 1 --epochs 30
```

Under DDP: `DistributedSampler` shards the data per rank, gradients are
all-reduced (RCCL), normalization is computed on rank 0 and broadcast, metrics
are all-reduced, and only rank 0 logs/checkpoints. Verified across 2× MI350X.

## 5. CLI

`phlogiston train` flags: `--stage {1,2}`, `--epochs`, `--batch-size`, `--lr`,
`--encoder-lr` (stage-2 encoder LR), `--mul`, `--n-layers`, `--correlation`,
`--max-shards`, `--out-dir`, `--init-ckpt` (warm-start weights),
`--resume` (restore optimizer/scheduler/epoch/best and continue),
`--warmup-epochs`, `--patience` (early stopping).

Typical schedule-B run:
```bash
torchrun --nproc_per_node=2 -m phlogiston.cli train --stage 1 --epochs 40 \
    --mul 128 --correlation 3 --out-dir runs
torchrun --nproc_per_node=2 -m phlogiston.cli train --stage 2 --epochs 40 \
    --init-ckpt runs/predictor_stage1.pt --encoder-lr 1e-4 --lr 1e-3 --out-dir runs
```

## 6. Design rationale (why these choices)

- **AdamW over Adam**: AdamW *decouples* weight decay from the adaptive step
  (plain Adam folds L2 into the moment estimate, scaling decay per-parameter and
  breaking proper regularization). Correct decay matters most in Stage 2, where
  we fine-tune on only ~12k labels and overfitting is a real risk.
- **Huber loss (smooth-L1)** over MSE/MAE: quadratic near zero (precise, smooth
  gradients) but linear for large errors (bounded gradient → robust to the
  outlier/noisy DFT labels materials data contains). Computed in standardized
  space so `delta=1.0` ≈ 1 std. MSE would let outliers dominate; L1 is
  non-smooth at 0 and converges less precisely.
- **Target standardization (train-split only)**: puts energies (eV), moduli
  (GPa), κ (W/m/K) on comparable scales so the multi-task loss is balanced and
  no unit dominates; train-only avoids val/test leakage. Predictions are
  de-standardized to physical units at inference.
- **Cosine-annealing LR**: smooth decay without manual step milestones (warmup
  not yet added).
- **Checkpoint contents**: model state + `stage` + `mean`/`std`. The norm stats
  are saved *with* the weights because they are required to de-standardize
  outputs — a checkpoint without them yields wrong physical predictions. Rank-0
  only, unwrapped (non-DDP) state so it loads for single- or multi-GPU.

## 7. Hyperparameter tuning outcome

Coarse grid sweep (`scripts/sweep.py`, 8-shard subset, 15 epochs):

- **Learning rate** (biggest lever): `3e-3` (0.313) > `1e-3` (0.334) > `3e-4` (0.356)
  on the short subset runs — higher LR converged fastest there.
- **Capacity**: differences were small (val_loss 0.307–0.315). `mul=192` was
  marginally best; extra depth (`L3`) was marginal; and **ν=3 did *not* beat ν=2**
  (rank 4) despite being much slower → **use ν=2** (drop the 4-body contraction).

**Chosen config**: the full-corpus **baseline** `lr=1e-3, mul=128, ν=2, n_layers=2`,
which already reaches **~0.021 eV/atom** formation-energy val MAE on the *full*
corpus (far better than any subset number, and not comparable to them). Launching
the marginally-different sweep-optimal config (`3e-3/mul192`) was judged
diminishing returns. The sweep's main value: confirming the config is robust and
that **ν=3 isn't worth the cost**.

## 7b. Performance investigation (negative result)

GPU utilization sat at **~39%** during training. We tried three levers and
**none moved it**: larger batch (128→512), more DataLoader workers (up to 16 on a
384-core box), and `torch.compile(dynamic=True)` (epoch time unchanged after the
one-off compile cost). Conclusion: the bottleneck is the **equivariant model's
execution profile** — many small `e3nn` tensor-product kernels with host/launch
gaps (and `rocm-smi` likely under-reporting bursty small-kernel occupancy), not
data loading, batch size, or kernel fusion. We **accept ~250s/epoch**; it still
trains to ~0.019 eV/atom. `--compile` is kept as an opt-in (off by default) but
did not help this workload. `num_workers`/`batch_size` defaults are kept as
reasonable (they don't hurt).

## 8. Status & open items

- **Real Stage-1 run in progress** on the full corpus (baseline config), 2-GPU
  DDP; formation-energy val MAE ~0.021 eV/atom and converging.
- **Open**:
  - **Dataset RAM**: in-memory load is ~30 GB *per rank* (~60 GB for 2). Confirm
    box RAM or add lazy per-shard loading before the full run.
  - Test-set evaluation + parity plots + stability AUC (currently only val MAE).
  - Per-target loss weights; optional `log1p` for skewed targets (predictor §6).
  - EMA of weights (deferred to the CDVAE generator; the predictor needs it little).
  - (Done: per-epoch `_last` + best-by-val `_best` checkpoints + `--resume`;
    linear LR warmup; early stopping on val.)
  - Set `avg_num_neighbors` precisely from data (currently the ~50 default).
