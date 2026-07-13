"""Training loop for the CDVAE generator (see models/cdvae/DESIGN.md §6).

Separate from the predictor trainer: the objective is the CDVAE composite loss
(KL + num/lattice/composition + coord-score + type), and generation quality
depends on an **EMA** of the weights, so we keep one and select/checkpoint by
the EMA's validation loss.

Multi-GPU is data-parallel (DDP), launched with
``torchrun --nproc_per_node=N -m phlogiston.cli train-cdvae ...`` (N up to 4).
CDVAE.forward delegates to training_loss so DDP's grad all-reduce fires.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from phlogiston.data.dataset import ShardedCrystalDataset, collate, physical_mask
from phlogiston.models.cdvae import CDVAE
from phlogiston.train.ema import EMA

# reuse the predictor trainer's building blocks (identical semantics)
from phlogiston.train.trainer import (
    _build_scheduler,
    _dist_info,
    _unwrap,
    split_indices,
)

_LOSS_KEYS = ("kl", "num", "lattice", "composition", "coord", "type")


def stable_indices(dataset, e_hull_max: float) -> list[int]:
    """Indices whose energy_above_hull is present and <= ``e_hull_max`` (the
    generator should learn the distribution of *stable* crystals)."""
    from phlogiston.data.dataset import TARGET_KEYS

    tcol = TARGET_KEYS.index("energy_above_hull")
    records = getattr(dataset, "records", None)
    if records is None:
        return list(range(len(dataset)))
    keep = []
    for i, r in enumerate(records):
        y = torch.tensor(r["y"])
        m = torch.tensor(r["mask"], dtype=torch.bool) & physical_mask(y)
        if bool(m[tcol]) and float(y[tcol]) <= e_hull_max:
            keep.append(i)
    return keep


@torch.no_grad()
def evaluate(model, loader, device, distributed: bool = False):
    """Mean composite loss (+ parts) over a loader. All-reduced under DDP."""
    model.eval()
    base = _unwrap(model)
    acc = torch.zeros(len(_LOSS_KEYS) + 1, device=device)  # [total, *parts]
    nb = torch.zeros(1, device=device)
    for batch in loader:
        batch = batch.to(device)
        total, parts = base.training_loss(batch)
        acc[0] += total.detach()
        for j, k in enumerate(_LOSS_KEYS, start=1):
            acc[j] += parts[k].detach()
        nb += 1
    if distributed:
        dist.all_reduce(acc)
        dist.all_reduce(nb)
    acc = (acc / nb.clamp(min=1)).cpu()
    return acc[0].item(), {k: acc[j + 1].item() for j, k in enumerate(_LOSS_KEYS)}


def _save_ckpt(path, base, ema, epoch, opt, sched, best_val, hparams):
    torch.save(
        {
            "model": base.state_dict(),
            "ema": ema.state_dict(),
            "epoch": epoch,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_val": best_val,
            "hparams": hparams,
        },
        path,
    )


def train_cdvae(
    data_root: str = "data",
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    latent_dim: int = 256,
    mul: int = 128,
    n_layers: int = 3,
    correlation: int = 2,
    n_max: int = 64,
    beta: float = 0.01,
    ema_decay: float = 0.999,
    grad_clip: float = 5.0,
    stable_max: float | None = None,
    max_shards: int | None = None,
    device: str | None = None,
    out_dir: str = "runs",
    resume: str | None = None,
    init_ckpt: str | None = None,
    warmup_epochs: int = 2,
    patience: int = 8,
    num_workers: int = 4,
    distill_root: str | None = None,
    distill_weight: int = 1,
    seed: int = 42,
):
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")

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
    log(f"[cdvae] world {world} | device {device} | loading data ...")

    dataset = ShardedCrystalDataset(data_root, max_shards=max_shards)
    if stable_max is not None:
        idx = stable_indices(dataset, stable_max)
        log(f"[cdvae] stable filter e_hull<={stable_max}: {len(idx):,}/{len(dataset):,} kept")
        dataset = Subset(dataset, idx)
    tr, va, _ = split_indices(len(dataset), seed=seed)
    log(f"[cdvae] {len(dataset):,} graphs -> train {len(tr):,} / val {len(va):,}")

    resume_state = torch.load(resume, map_location=device) if resume else None
    # warm-start: load weights (+EMA) only, then train with a FRESH optimizer /
    # schedule / epoch counter -- unlike --resume, which restores the annealed
    # scheduler (LR ~0). Use this to continue a capped run to convergence.
    init_state = torch.load(init_ckpt, map_location=device) if init_ckpt else None
    model = CDVAE(
        latent_dim=latent_dim, mul=mul, n_max=n_max, beta=beta,
        n_layers=n_layers, correlation=correlation,
    ).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        log(f"[cdvae] resuming from {resume} (epoch {resume_state['epoch']})")
    elif init_state is not None:
        model.load_state_dict(init_state["model"])
        log(f"[cdvae] warm-started (weights+EMA, fresh schedule) from {init_ckpt}")

    base = model
    if distributed:
        model = DDP(model, device_ids=[local])

    ema = EMA(base, decay=ema_decay)
    if resume_state is not None and resume_state.get("ema"):
        ema.load_state_dict(resume_state["ema"])
    elif init_state is not None and init_state.get("ema"):
        ema.load_state_dict(init_state["ema"])

    opt = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _build_scheduler(opt, epochs, warmup_epochs)
    start_epoch, best_val = 0, float("inf")
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        sched.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        best_val = resume_state.get("best_val", float("inf"))

    train_subset = Subset(dataset, tr)
    # Relaxation self-distillation: mix the corpus of uMLIP-relaxed generated
    # structures into the TRAIN split only (never val), replicated to up-weight
    # the scarce corpus vs. the large original set. Single-process only.
    if distill_root:
        if distributed:
            raise ValueError("distill mixing is single-process; launch without torchrun")
        from torch.utils.data import ConcatDataset

        corpus = ShardedCrystalDataset(distill_root)
        w = max(1, distill_weight)
        train_subset = ConcatDataset([train_subset] + [corpus] * w)
        log(f"[cdvae] distill mix: +{len(corpus):,} relaxed-generated records x{w} "
            f"-> train {len(train_subset):,}")

    dl_kw = dict(collate_fn=collate, num_workers=num_workers, persistent_workers=num_workers > 0)
    if distributed:
        train_sampler = DistributedSampler(train_subset, shuffle=True, seed=seed)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, **dl_kw)
        val_loader = DataLoader(
            Subset(dataset, va), batch_size=batch_size,
            sampler=DistributedSampler(Subset(dataset, va), shuffle=False), **dl_kw,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, **dl_kw)
        val_loader = DataLoader(Subset(dataset, va), batch_size=batch_size, shuffle=False, **dl_kw)

    last_path = Path(out_dir) / "cdvae_last.pt"
    best_path = Path(out_dir) / "cdvae_best.pt"
    hparams = {
        "latent_dim": latent_dim, "mul": mul, "n_layers": n_layers,
        "correlation": correlation, "n_max": n_max, "beta": beta,
    }
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            total, _ = model(batch)  # DDP.forward -> CDVAE.forward -> training_loss
            opt.zero_grad()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
            opt.step()
            ema.update(base)
            running += total.item()
            n_batches += 1
        sched.step()

        # validate with EMA weights (what we'll sample from)
        with ema.averaged(base):
            val_loss, parts = evaluate(model, val_loader, device, distributed)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        parts_s = " ".join(f"{k}={parts[k]:.3f}" for k in _LOSS_KEYS)
        log(
            f"[cdvae] epoch {epoch + 1}/{epochs} train={running / max(n_batches, 1):.4f} "
            f"val(EMA)={val_loss:.4f}{' *' if improved else ''} | {parts_s} "
            f"{time.time() - t0:.1f}s"
        )
        if is_main:
            _save_ckpt(last_path, base, ema, epoch, opt, sched, best_val, hparams)
            if improved:
                _save_ckpt(best_path, base, ema, epoch, opt, sched, best_val, hparams)

        if patience > 0 and epochs_no_improve >= patience:
            log(f"[cdvae] early stopping at epoch {epoch + 1} (best_val={best_val:.4f})")
            break

    if is_main:
        log(f"[cdvae] done. last={last_path} best={best_path} (best_val={best_val:.4f})")
    if distributed:
        dist.destroy_process_group()
    return str(best_path)
