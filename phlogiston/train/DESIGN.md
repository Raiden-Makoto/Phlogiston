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
- **Optimizer**: AdamW; **schedule**: cosine annealing over `epochs`.
- **Eval**: `evaluate()` returns per-target **MAE in physical units** over masked
  val entries (all-reduced across ranks under DDP).
- **Checkpoint**: rank 0 saves `{model state, stage, mean, std}` to
  `out_dir/predictor_stage{N}.pt`.

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
`--max-shards`, `--out-dir`, `--init-ckpt`.

Typical schedule-B run:
```bash
torchrun --nproc_per_node=2 -m phlogiston.cli train --stage 1 --epochs 40 \
    --mul 128 --correlation 3 --out-dir runs
torchrun --nproc_per_node=2 -m phlogiston.cli train --stage 2 --epochs 40 \
    --init-ckpt runs/predictor_stage1.pt --encoder-lr 1e-4 --lr 1e-3 --out-dir runs
```

## 6. Status & open items

- **Done & smoke-validated** (tiny subset, 1–2 GPUs): loop runs end-to-end, loss
  and MAE decrease, DDP works. **No real training run yet** (no trained model).
- **Open**:
  - **Dataset RAM**: in-memory load is ~30 GB *per rank* (~60 GB for 2). Confirm
    box RAM or add lazy per-shard loading before the full run.
  - Test-set evaluation + parity plots + stability AUC (currently only val MAE).
  - Per-target loss weights; optional `log1p` for skewed targets (predictor §6).
  - EMA of weights; early stopping on val.
  - Set `avg_num_neighbors` precisely from data (currently the ~50 default).
