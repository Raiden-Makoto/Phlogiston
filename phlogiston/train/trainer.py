"""Schedule-B training loop for the Predictor.

Stage 1 pretrains the encoder + stability heads (loss masked to the stability
targets); Stage 2 fine-tunes the encoder (low LR) + all heads on every target.
Single-GPU for now; data-parallel across 2 (max 4) GPUs drops in next.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from phlogiston.data.dataset import ShardedCrystalDataset, collate
from phlogiston.models.predictor import PREDICT_KEYS, STABILITY_KEYS, Predictor


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


def _loader(dataset, indices, batch_size, shuffle, num_workers=0):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
    )


def _stage_mask(mask, stage: int):
    """Stage 1 keeps only stability columns; stage 2 keeps all."""
    if stage == 2:
        return mask
    keep = torch.zeros(mask.shape[1], dtype=torch.bool, device=mask.device)
    for k in STABILITY_KEYS:
        keep[PREDICT_KEYS.index(k)] = True
    return mask & keep


@torch.no_grad()
def evaluate(model, loader, device, stage: int = 2):
    """Per-target MAE (physical units) over masked entries."""
    model.eval()
    t = model.n_targets
    abs_err = torch.zeros(t, device=device)
    cnt = torch.zeros(t, device=device)
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y, mask = model.slice_targets(batch.y, batch.y_mask)
        mask = _stage_mask(mask, stage).float()
        abs_err += (pred - y).abs().mul(mask).sum(0)
        cnt += mask.sum(0)
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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[train] stage {stage} | device {device} | loading data ...")

    dataset = ShardedCrystalDataset(data_root, max_shards=max_shards)
    tr, va, te = split_indices(len(dataset), seed=seed)
    print(
        f"[train] {len(dataset):,} graphs -> train {len(tr):,} / val {len(va):,} / test {len(te):,}"
    )

    model = Predictor(mul=mul, n_layers=n_layers, correlation=correlation).to(device)
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location=device)["model"])
        print(f"[train] loaded init checkpoint {init_ckpt}")

    mean, std = compute_normalization(dataset, tr, model.pred_idx.cpu())
    model.set_normalization(mean.to(device), std.to(device))

    if stage == 1:
        opt = torch.optim.AdamW(model.stage1_parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(
            model.stage2_param_groups(encoder_lr, lr), weight_decay=weight_decay
        )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = _loader(dataset, tr, batch_size, shuffle=True)
    val_loader = _loader(dataset, va, batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            y, mask = model.slice_targets(batch.y, batch.y_mask)
            mask = _stage_mask(mask, stage)
            loss, _ = model.loss(pred, y, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()
        val_mae = evaluate(model, val_loader, device, stage)
        stab = ", ".join(f"{k.split('_')[0]}={val_mae[k]:.3f}" for k in STABILITY_KEYS)
        print(
            f"[train] epoch {epoch + 1}/{epochs} loss={running / max(n_batches, 1):.4f} "
            f"val_MAE({stab}) {time.time() - t0:.1f}s"
        )

    ckpt = Path(out_dir) / f"predictor_stage{stage}.pt"
    torch.save({"model": model.state_dict(), "stage": stage, "mean": mean, "std": std}, ckpt)
    print(f"[train] saved {ckpt}")
    return str(ckpt)
