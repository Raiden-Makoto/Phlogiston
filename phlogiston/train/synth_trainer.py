"""Training loop for the Tier-1 synthesizability classifier.

Binary PU classification (observed MP vs. hypothetical MP+GNoME) on the shared
equivariant encoder. Data-parallel (DDP) like the Predictor trainer: launch with
``torchrun --nproc_per_node=N -m phlogiston.cli train-synth ...``. Selection and
early stopping use validation ROC-AUC (discrimination under heavy class
imbalance is the goal). The encoder can be warm-started from a trained
stability/predictor checkpoint via ``init_ckpt``.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from phlogiston.data.dataset import collate
from phlogiston.data.synth import SynthesizabilityDataset
from phlogiston.models.synth import SynthesizabilityModel
from phlogiston.train.trainer import _build_scheduler, _dist_info, _unwrap, split_indices


@torch.no_grad()
def evaluate_synth(model, loader, device, distributed: bool = False) -> dict:
    """ROC-AUC + average precision + accuracy over the (gathered) val set."""
    import numpy as np

    model.eval()
    base = _unwrap(model)
    logits_all, y_all = [], []
    loss_sum = torch.zeros(1, device=device)
    n = torch.zeros(1, device=device)
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y[:, 0]
        loss_sum += base.loss(logits, y).detach()
        n += 1
        logits_all.append(logits.detach())
        y_all.append(y.detach())
    lg = torch.cat(logits_all) if logits_all else torch.zeros(0, device=device)
    yy = torch.cat(y_all) if y_all else torch.zeros(0, device=device)
    if distributed:
        for t in (loss_sum, n):
            dist.all_reduce(t)
        gl, gy = [None] * dist.get_world_size(), [None] * dist.get_world_size()
        dist.all_gather_object(gl, lg.cpu().tolist())
        dist.all_gather_object(gy, yy.cpu().tolist())
        p = np.array([x for sub in gl for x in sub])
        y = np.array([x for sub in gy for x in sub])
    else:
        p, y = lg.cpu().numpy(), yy.cpu().numpy()

    val_loss = (loss_sum / n.clamp(min=1)).item()
    out = {"val_loss": val_loss, "auc": float("nan"), "ap": float("nan"),
           "acc": float("nan"), "frac_pos": float("nan"), "n": int(len(y))}
    if len(y) == 0 or y.min() == y.max():
        return out
    from sklearn.metrics import average_precision_score, roc_auc_score

    prob = 1.0 / (1.0 + np.exp(-p))
    label = (y > 0.5).astype(int)
    out["auc"] = float(roc_auc_score(label, prob))
    out["ap"] = float(average_precision_score(label, prob))
    out["acc"] = float(((prob > 0.5).astype(int) == label).mean())
    out["frac_pos"] = float(label.mean())
    return out


def _save(path, base, epoch, opt, sched, best, hparams):
    torch.save(
        {
            "model": base.state_dict(),
            "epoch": epoch,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_auc": best,
            "hparams": hparams,
        },
        path,
    )


def train_synth(
    data_root: str = "data",
    *,
    epochs: int = 8,
    batch_size: int = 512,
    lr: float = 1e-3,
    encoder_lr: float | None = None,
    weight_decay: float = 1e-5,
    mul: int = 128,
    n_layers: int = 2,
    correlation: int = 3,
    max_shards: int | None = None,
    include_gnome: bool = True,
    device: str | None = None,
    out_dir: str = "runs",
    init_ckpt: str | None = None,
    resume: str | None = None,
    warmup_epochs: int = 1,
    patience: int = 6,
    num_workers: int = 4,
    grad_clip: float = 5.0,
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

    def log(m):
        if is_main:
            print(m, flush=True)

    if is_main:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    log(f"[synth] world {world} | device {device} | loading shards ...")

    dataset = SynthesizabilityDataset(data_root, max_shards=max_shards, include_gnome=include_gnome)
    valid = dataset.valid_indices()
    pos, tot = dataset.positive_count(valid)
    log(
        f"[synth] {len(dataset):,} graphs -> {tot:,} labeled "
        f"({pos:,} observed / {tot - pos:,} hypothetical, {100 * pos / max(tot, 1):.1f}% positive)"
    )
    # split positions within the valid pool (identical across ranks)
    p_tr, p_va, p_te = split_indices(len(valid), seed=seed)
    tr = [valid[i] for i in p_tr]
    va = [valid[i] for i in p_va]
    log(f"[synth] train {len(tr):,} / val {len(va):,} / test {len(p_te):,}")

    # pos_weight = #neg/#pos over the train split, to counter class imbalance
    n_pos = sum(1 for i in tr if dataset.labels[i] > 0.5)
    n_neg = len(tr) - n_pos
    pos_weight = torch.tensor([max(n_neg, 1) / max(n_pos, 1)], device=device)
    log(f"[synth] pos_weight={pos_weight.item():.2f}")

    resume_state = torch.load(resume, map_location=device) if resume else None
    model = SynthesizabilityModel(mul=mul, n_layers=n_layers, correlation=correlation).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        log(f"[synth] resuming from {resume} (epoch {resume_state['epoch']})")
    elif init_ckpt:
        n_enc = model.load_encoder_from(init_ckpt, map_location=device)
        log(f"[synth] warm-started encoder ({n_enc} tensors) from {init_ckpt}")

    base = model
    if distributed:
        model = DDP(model, device_ids=[local])

    # optional lower LR for a warm-started encoder (fine-tune) vs. the fresh head
    if encoder_lr is not None:
        groups = [
            {"params": list(base.encoder.parameters()), "lr": encoder_lr},
            {"params": list(base.head.parameters()), "lr": lr},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _build_scheduler(opt, epochs, warmup_epochs)

    start_epoch, best_auc = 0, -1.0
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        sched.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        best_auc = resume_state.get("best_auc", -1.0)

    dl_kw = dict(collate_fn=collate, num_workers=num_workers, persistent_workers=num_workers > 0)
    if distributed:
        train_sampler = DistributedSampler(Subset(dataset, tr), shuffle=True, seed=seed)
        train_loader = DataLoader(Subset(dataset, tr), batch_size=batch_size, sampler=train_sampler, **dl_kw)
        val_loader = DataLoader(
            Subset(dataset, va), batch_size=batch_size,
            sampler=DistributedSampler(Subset(dataset, va), shuffle=False), **dl_kw,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(Subset(dataset, tr), batch_size=batch_size, shuffle=True, **dl_kw)
        val_loader = DataLoader(Subset(dataset, va), batch_size=batch_size, shuffle=False, **dl_kw)

    last_path = Path(out_dir) / "synth_last.pt"
    best_path = Path(out_dir) / "synth_best.pt"
    hp = {"mul": mul, "n_layers": n_layers, "correlation": correlation}
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = base.loss(logits, batch.y[:, 0], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        m = evaluate_synth(model, val_loader, device, distributed)
        improved = m["auc"] > best_auc
        if improved:
            best_auc = m["auc"]
            no_improve = 0
        else:
            no_improve += 1
        log(
            f"[synth] epoch {epoch + 1}/{epochs} loss={running / max(nb, 1):.4f} "
            f"val_loss={m['val_loss']:.4f} AUC={m['auc']:.4f} AP={m['ap']:.4f} "
            f"acc={m['acc']:.3f} (pos={m['frac_pos']:.2f}){' *' if improved else ''} "
            f"{time.time() - t0:.1f}s"
        )
        if is_main:
            _save(last_path, base, epoch, opt, sched, best_auc, hp)
            if improved:
                _save(best_path, base, epoch, opt, sched, best_auc, hp)
        if patience > 0 and no_improve >= patience:
            log(f"[synth] early stopping at epoch {epoch + 1} (best AUC={best_auc:.4f})")
            break

    log(f"[synth] done. best={best_path} (AUC={best_auc:.4f})")
    if distributed:
        dist.destroy_process_group()
    return str(best_path)
