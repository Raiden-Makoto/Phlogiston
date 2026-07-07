"""Schedule-B training loop for the Predictor.

Stage 1 pretrains the encoder + stability heads (loss masked to the stability
targets); Stage 2 fine-tunes the encoder (low LR) + all heads on every target.

Multi-GPU is **data-parallel (DDP)**, not tensor-parallel: launch with
``torchrun --nproc_per_node=N -m phlogiston.cli train ...`` (N up to 4). Each
rank replicates the model, processes a shard of the batch, and gradients are
all-reduced. Single-process (N=1) works unchanged.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.predictor import PREDICT_KEYS, STABILITY_KEYS, Predictor


def _dist_info():
    """(world_size, rank, local_rank) from the torchrun env (defaults: single)."""
    return (
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def _unwrap(model):
    return model.module if isinstance(model, DDP) else model


def split_indices(n: int, ratios=(0.8, 0.1, 0.1), seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


def compute_normalization(dataset, indices, pred_idx: torch.Tensor):
    """Masked per-target mean/std over the train split (physical units)."""
    t = len(pred_idx)
    s = torch.zeros(t, dtype=torch.float64)
    ss = torch.zeros(t, dtype=torch.float64)
    cnt = torch.zeros(t, dtype=torch.float64)
    for i in indices:
        _, y, m = dataset[i]
        y = y[pred_idx].double()
        m = m[pred_idx].double()
        s += y * m
        ss += (y * y) * m
        cnt += m
    cnt = cnt.clamp(min=1)
    mean = s / cnt
    var = (ss / cnt) - mean * mean
    std = var.clamp(min=1e-12).sqrt()
    return mean.float(), std.float()


def _stage_mask(mask, stage: int):
    """Stage 1 keeps only stability columns; stage 2 keeps all."""
    if stage == 2:
        return mask
    keep = torch.zeros(mask.shape[1], dtype=torch.bool, device=mask.device)
    for k in STABILITY_KEYS:
        keep[PREDICT_KEYS.index(k)] = True
    return mask & keep


@torch.no_grad()
def evaluate(model, loader, device, stage: int = 2, distributed: bool = False):
    """Per-target MAE (physical units) over masked entries (all-reduced if DDP)."""
    model.eval()
    base = _unwrap(model)
    t = base.n_targets
    abs_err = torch.zeros(t, device=device)
    cnt = torch.zeros(t, device=device)
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y, mask = base.slice_targets(batch.y, batch.y_mask)
        mask = _stage_mask(mask, stage).float()
        abs_err += (pred - y).abs().mul(mask).sum(0)
        cnt += mask.sum(0)
    if distributed:
        dist.all_reduce(abs_err)
        dist.all_reduce(cnt)
    mae = (abs_err / cnt.clamp(min=1)).cpu()
    return {k: mae[i].item() for i, k in enumerate(PREDICT_KEYS)}


def train(
    data_root: str = "data",
    *,
    stage: int = 1,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    encoder_lr: float = 1e-4,
    weight_decay: float = 1e-5,
    mul: int = 128,
    n_layers: int = 2,
    correlation: int = 3,
    max_shards: int | None = None,
    device: str | None = None,
    out_dir: str = "runs",
    init_ckpt: str | None = None,
    seed: int = 42,
):
    world, rank, local = _dist_info()
    distributed = world > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    if is_main:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    log(f"[train] stage {stage} | world {world} | device {device} | loading data ...")

    dataset = ShardedCrystalDataset(data_root, max_shards=max_shards)
    tr, va, te = split_indices(len(dataset), seed=seed)  # identical across ranks
    log(
        f"[train] {len(dataset):,} graphs -> train {len(tr):,} / val {len(va):,} / test {len(te):,}"
    )

    model = Predictor(mul=mul, n_layers=n_layers, correlation=correlation).to(device)
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location=device)["model"])
        log(f"[train] loaded init checkpoint {init_ckpt}")

    # normalization: compute on rank 0, broadcast to all ranks
    mean = torch.zeros(model.n_targets, device=device)
    std = torch.ones(model.n_targets, device=device)
    if is_main:
        m0, s0 = compute_normalization(dataset, tr, model.pred_idx.cpu())
        mean.copy_(m0.to(device))
        std.copy_(s0.to(device))
    if distributed:
        dist.broadcast(mean, src=0)
        dist.broadcast(std, src=0)
    model.set_normalization(mean, std)

    base = model
    if distributed:
        model = DDP(model, device_ids=[local])

    if stage == 1:
        opt = torch.optim.AdamW(base.stage1_parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(base.stage2_param_groups(encoder_lr, lr), weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if distributed:
        train_sampler = DistributedSampler(Subset(dataset, tr), shuffle=True, seed=seed)
        train_loader = DataLoader(
            Subset(dataset, tr), batch_size=batch_size, sampler=train_sampler, collate_fn=collate
        )
        val_loader = DataLoader(
            Subset(dataset, va),
            batch_size=batch_size,
            sampler=DistributedSampler(Subset(dataset, va), shuffle=False),
            collate_fn=collate,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            Subset(dataset, tr), batch_size=batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = DataLoader(
            Subset(dataset, va), batch_size=batch_size, shuffle=False, collate_fn=collate
        )

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            y, mask = base.slice_targets(batch.y, batch.y_mask)
            mask = _stage_mask(mask, stage)
            loss, _ = base.loss(pred, y, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()
        val_mae = evaluate(model, val_loader, device, stage, distributed)
        stab = ", ".join(f"{k.split('_')[0]}={val_mae[k]:.3f}" for k in STABILITY_KEYS)
        log(
            f"[train] epoch {epoch + 1}/{epochs} loss={running / max(n_batches, 1):.4f} "
            f"val_MAE({stab}) {time.time() - t0:.1f}s"
        )

    ckpt = Path(out_dir) / f"predictor_stage{stage}.pt"
    if is_main:
        torch.save(
            {"model": base.state_dict(), "stage": stage, "mean": mean.cpu(), "std": std.cpu()},
            ckpt,
        )
        log(f"[train] saved {ckpt}")
    if distributed:
        dist.destroy_process_group()
    return str(ckpt)
