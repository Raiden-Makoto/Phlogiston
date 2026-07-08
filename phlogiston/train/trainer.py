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

from phlogiston.data.dataset import ShardedCrystalDataset, collate, physical_mask
from phlogiston.models.predictor import PREDICT_KEYS, STABILITY_KEYS, Predictor

_LABELS = {
    "formation_energy_per_atom": "formE",
    "energy_above_hull": "Ehull",
    "bulk_modulus_vrh": "K",
    "shear_modulus_vrh": "G",
    "vickers_hardness": "Hv",
    "fracture_toughness": "Kic",
    "debye_temperature": "Debye",
    "slack_thermal_conductivity": "kappa",
}


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


def compute_normalization(dataset, indices, pred_idx: torch.Tensor, log_mask=None):
    """Masked per-target mean/std over the train split, in the model's *transform*
    space: columns flagged by ``log_mask`` (pred order) are log1p'd first, so the
    stats match what the loss standardizes (see Predictor.to_transform)."""
    t = len(pred_idx)
    s = torch.zeros(t, dtype=torch.float64)
    ss = torch.zeros(t, dtype=torch.float64)
    cnt = torch.zeros(t, dtype=torch.float64)
    lm = log_mask.cpu() if log_mask is not None else torch.zeros(t, dtype=torch.bool)
    # read labels directly from shard records if available (avoids rebuilding
    # graph tensors for every training example just to grab y/mask).
    records = getattr(dataset, "records", None)
    for i in indices:
        if records is not None:
            r = records[i]
            yf = torch.tensor(r["y"], dtype=torch.float64)
            # match __getitem__: drop physically-implausible labels (poison std)
            mf = torch.tensor(r["mask"], dtype=torch.bool) & physical_mask(yf)
            y = yf[pred_idx]
            m = mf.double()[pred_idx]
        else:
            _, y, m = dataset[i]
            y = y[pred_idx].double()
            m = m[pred_idx].double()
        y = torch.where(lm, torch.log1p(y.clamp(min=-1 + 1e-6)), y)  # transform space
        s += y * m
        ss += (y * y) * m
        cnt += m
    cnt = cnt.clamp(min=1)
    mean = s / cnt
    var = (ss / cnt) - mean * mean
    std = var.clamp(min=1e-12).sqrt()
    return mean.float(), std.float()


def _build_scheduler(opt, epochs: int, warmup_epochs: int):
    """Linear LR warmup for ``warmup_epochs`` then cosine anneal (per-epoch)."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    w = max(0, min(warmup_epochs, epochs - 1))
    if w == 0:
        return CosineAnnealingLR(opt, T_max=max(epochs, 1))
    warm = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=w)
    cos = CosineAnnealingLR(opt, T_max=max(epochs - w, 1))
    return SequentialLR(opt, [warm, cos], milestones=[w])


def _save_ckpt(path, base, stage, mean, std, epoch, opt, sched, best_val, hparams=None):
    """Full checkpoint: weights + norm stats + optimizer/scheduler for resume.

    ``hparams`` records the architecture (mul/n_layers/correlation) so a
    checkpoint can be reconstructed for standalone evaluation without knowing
    the original CLI flags.
    """
    torch.save(
        {
            "model": base.state_dict(),
            "stage": stage,
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "epoch": epoch,
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_val": best_val,
            "hparams": hparams or {},
        },
        path,
    )


def _stage_mask(mask, stage: int):
    """Stage 1 keeps only stability columns; stage 2 keeps all."""
    if stage == 2:
        return mask
    keep = torch.zeros(mask.shape[1], dtype=torch.bool, device=mask.device)
    for k in STABILITY_KEYS:
        keep[PREDICT_KEYS.index(k)] = True
    return mask & keep


def _stability_metrics(pred_ehull, true_ehull, threshold, distributed):
    """ROC-AUC + average-precision for unstable (e_above_hull > threshold)."""
    import numpy as np

    if distributed:
        gathered_p, gathered_t = [None] * dist.get_world_size(), [None] * dist.get_world_size()
        dist.all_gather_object(gathered_p, pred_ehull.cpu().tolist())
        dist.all_gather_object(gathered_t, true_ehull.cpu().tolist())
        p = np.array([x for sub in gathered_p for x in sub])
        y = np.array([x for sub in gathered_t for x in sub])
    else:
        p, y = pred_ehull.cpu().numpy(), true_ehull.cpu().numpy()

    finite = np.isfinite(p) & np.isfinite(y)  # a diverged model can emit NaN/inf
    p, y = p[finite], y[finite]
    if len(y) == 0:
        return {"auc": float("nan"), "ap": float("nan"), "frac_unstable": float("nan")}
    label = (y > threshold).astype(int)  # positive class = unstable (the minority)
    frac = float(label.mean())
    if label.min() == label.max():  # only one class present -> AUC undefined
        return {"auc": float("nan"), "ap": float("nan"), "frac_unstable": frac}
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {"auc": float(roc_auc_score(label, p)),
            "ap": float(average_precision_score(label, p)), "frac_unstable": frac}


def _selection_score(select_by, metrics, report_keys, val_loss):
    """Lower-is-better scalar for best-checkpoint selection.

    ``auc``/``r2`` are higher-is-better, so we negate them; if the metric is
    NaN (e.g. degenerate val fold, or no property labels present) we fall back
    to ``val_loss`` so checkpointing never stalls.
    """
    import math

    if select_by == "loss":
        return val_loss
    if select_by == "auc":
        auc = metrics["stability"]["auc"]
        return -auc if not math.isnan(auc) else val_loss
    # "r2": mean over the stage-relevant targets
    vals = [metrics["r2"][k] for k in report_keys]
    vals = [v for v in vals if not math.isnan(v)]
    return -(sum(vals) / len(vals)) if vals else val_loss


@torch.no_grad()
def evaluate(model, loader, device, stage: int = 2, distributed: bool = False,
             stability_threshold: float = 0.0):
    """Per-target MAE + R² (physical units), standardized val loss, and stability
    ROC-AUC/AP for energy_above_hull. All all-reduced/gathered under DDP.

    Why more than MAE: R² catches "predicts the mean" (low MAE, no variance
    explained); AUC/AP catch poor stable/unstable separation under the ~98%
    class imbalance despite a small e-hull MAE.
    """
    model.eval()
    base = _unwrap(model)
    t = base.n_targets
    z = lambda: torch.zeros(t, device=device)  # noqa: E731
    abs_err, sq_err, sum_y, sum_y2, cnt = z(), z(), z(), z(), z()
    loss_sum = torch.zeros(1, device=device)
    n_batches = torch.zeros(1, device=device)
    ehull = PREDICT_KEYS.index("energy_above_hull")
    ep, et = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y, mask = base.slice_targets(batch.y, batch.y_mask)
        mask = _stage_mask(mask, stage)
        mf = mask.float()
        abs_err += (pred - y).abs().mul(mf).sum(0)  # MAE in physical units
        # R² is computed in the model's *transform* space (log1p for LOG_TARGETS,
        # identity otherwise): physical-space R² for a log target is dominated by
        # exponential error amplification and is wildly unstable, whereas the
        # transform space is exactly where the quantity is linear.
        pt, yt = base.to_transform(pred), base.to_transform(y)
        difft = pt - yt
        sq_err += (difft * difft).mul(mf).sum(0)
        sum_y += yt.mul(mf).sum(0)
        sum_y2 += (yt * yt).mul(mf).sum(0)
        cnt += mf.sum(0)
        loss, _ = base.loss(pred, y, mask)
        loss_sum += loss.detach()
        n_batches += 1
        m = mask[:, ehull]
        if m.any():
            ep.append(pred[m, ehull].detach())
            et.append(y[m, ehull].detach())
    if distributed:
        for tns in (abs_err, sq_err, sum_y, sum_y2, cnt, loss_sum, n_batches):
            dist.all_reduce(tns)

    c = cnt.clamp(min=1)
    mae = (abs_err / c).cpu()
    ss_tot = (sum_y2 - sum_y * sum_y / c).clamp(min=1e-8)
    r2 = (1 - sq_err / ss_tot).cpu()
    val_loss = (loss_sum / n_batches.clamp(min=1)).item()
    stab = _stability_metrics(
        torch.cat(ep) if ep else torch.zeros(0, device=device),
        torch.cat(et) if et else torch.zeros(0, device=device),
        stability_threshold, distributed)
    return {
        "mae": {k: mae[i].item() for i, k in enumerate(PREDICT_KEYS)},
        "r2": {k: r2[i].item() for i, k in enumerate(PREDICT_KEYS)},
        "val_loss": val_loss,
        "stability": stab,
    }


def train(
    data_root: str = "data",
    *,
    stage: int = 1,
    epochs: int = 10,
    batch_size: int = 512,  # 288 GB HBM has ample headroom; keeps GPUs fed
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
    resume: str | None = None,
    warmup_epochs: int = 2,
    patience: int = 20,
    num_workers: int = 4,
    compile: bool = False,
    select_by: str | None = None,
    grad_clip: float = 5.0,
    seed: int = 42,
):
    # Best-checkpoint selection metric. Default is stage-aware: stability AUC
    # for stage 1 (discrimination is the goal), mean property R² for stage 2
    # (val_loss would be diluted by the easy stability targets). All are
    # NaN-guarded and fall back to val_loss.
    if select_by is None:
        select_by = "auc" if stage == 1 else "r2"
    if select_by not in ("loss", "auc", "r2"):
        raise ValueError(f"select_by must be loss|auc|r2, got {select_by!r}")
    if num_workers > 0:
        # DataLoader workers return batches (tensors) via IPC; the file_system
        # sharing strategy avoids file-descriptor exhaustion with many tensors.
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
    log(
        f"[train] stage {stage} | world {world} | device {device} | "
        f"select best by '{select_by}' | loading data ..."
    )

    dataset = ShardedCrystalDataset(data_root, max_shards=max_shards)
    tr, va, te = split_indices(len(dataset), seed=seed)  # identical across ranks
    log(
        f"[train] {len(dataset):,} graphs -> train {len(tr):,} / val {len(va):,} / test {len(te):,}"
    )

    resume_state = torch.load(resume, map_location=device) if resume else None

    model = Predictor(mul=mul, n_layers=n_layers, correlation=correlation).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        log(f"[train] resuming from {resume} (epoch {resume_state['epoch']})")
    elif init_ckpt:  # warm-start weights only (e.g. stage 1 -> stage 2)
        model.load_state_dict(torch.load(init_ckpt, map_location=device)["model"])
        log(f"[train] warm-started from {init_ckpt}")

    # normalization: from the resume checkpoint, else compute on rank 0 + broadcast
    mean = torch.zeros(model.n_targets, device=device)
    std = torch.ones(model.n_targets, device=device)
    if resume_state is not None:
        mean.copy_(resume_state["mean"].to(device))
        std.copy_(resume_state["std"].to(device))
    else:
        if is_main:
            m0, s0 = compute_normalization(dataset, tr, model.pred_idx.cpu(), model.log_mask)
            mean.copy_(m0.to(device))
            std.copy_(s0.to(device))
        if distributed:
            dist.broadcast(mean, src=0)
            dist.broadcast(std, src=0)
    model.set_normalization(mean, std)

    base = model
    if distributed:
        model = DDP(model, device_ids=[local])
    if compile:
        # fuse the many small e3nn kernels (the util bottleneck); dynamic=True
        # for variable node/edge counts across batches.
        log("[train] torch.compile(dynamic=True) ...")
        model = torch.compile(model, dynamic=True)

    if stage == 1:
        opt = torch.optim.AdamW(base.stage1_parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(base.stage2_param_groups(encoder_lr, lr), weight_decay=weight_decay)
    sched = _build_scheduler(opt, epochs, warmup_epochs)

    start_epoch, best_val = 0, float("inf")
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        sched.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        best_val = resume_state.get("best_val", float("inf"))

    dl_kw = dict(collate_fn=collate, num_workers=num_workers, persistent_workers=num_workers > 0)
    if distributed:
        train_sampler = DistributedSampler(Subset(dataset, tr), shuffle=True, seed=seed)
        train_loader = DataLoader(
            Subset(dataset, tr), batch_size=batch_size, sampler=train_sampler, **dl_kw
        )
        val_loader = DataLoader(
            Subset(dataset, va),
            batch_size=batch_size,
            sampler=DistributedSampler(Subset(dataset, va), shuffle=False),
            **dl_kw,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(Subset(dataset, tr), batch_size=batch_size, shuffle=True, **dl_kw)
        val_loader = DataLoader(Subset(dataset, va), batch_size=batch_size, shuffle=False, **dl_kw)

    last_path = Path(out_dir) / f"predictor_stage{stage}_last.pt"
    best_path = Path(out_dir) / f"predictor_stage{stage}_best.pt"
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
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
            if grad_clip > 0:  # guard against divergence (esp. log-space heads)
                torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()
        metrics = evaluate(model, val_loader, device, stage, distributed)
        val_loss = metrics["val_loss"]
        # report the stage-relevant targets (stability for stage 1; the
        # mechanical/thermal properties for stage 2)
        report_keys = (
            STABILITY_KEYS if stage == 1 else [k for k in PREDICT_KEYS if k not in STABILITY_KEYS]
        )
        mae_s = ", ".join(f"{_LABELS[k]}={metrics['mae'][k]:.3f}" for k in report_keys)
        r2_s = ", ".join(f"{_LABELS[k]}={metrics['r2'][k]:.2f}" for k in report_keys)
        st = metrics["stability"]
        # selection score: lower-is-better (negate metrics where higher=better),
        # NaN-safe fall back to val_loss so we always keep checkpointing.
        score = _selection_score(select_by, metrics, report_keys, val_loss)
        improved = score < best_val
        if improved:
            best_val = score
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        log(
            f"[train] epoch {epoch + 1}/{epochs} loss={running / max(n_batches, 1):.4f} "
            f"val_loss={val_loss:.4f} | MAE({mae_s}) | "
            f"R2({r2_s}) | stab AUC={st['auc']:.3f} AP={st['ap']:.3f} | "
            f"select[{select_by}]={score:.4f}{' *' if improved else ''} "
            f"{time.time() - t0:.1f}s"
        )
        if is_main:  # per-epoch resumable checkpoint (+ best-by-selection-metric)
            hp = {"mul": mul, "n_layers": n_layers, "correlation": correlation}
            _save_ckpt(last_path, base, stage, mean, std, epoch, opt, sched, best_val, hp)
            if improved:
                _save_ckpt(best_path, base, stage, mean, std, epoch, opt, sched, best_val, hp)

        if patience > 0 and epochs_no_improve >= patience:
            log(
                f"[train] early stopping at epoch {epoch + 1} "
                f"(no {select_by} improvement for {patience} epochs; "
                f"best[{select_by}]={best_val:.4f})"
            )
            break

    if is_main:
        log(f"[train] done. last={last_path} best={best_path} (best_val={best_val:.4f})")
    if distributed:
        dist.destroy_process_group()
    return str(best_path)


def evaluate_checkpoint(
    ckpt_path: str,
    data_root: str,
    split: str = "val",
    stage: int = 2,
    batch_size: int = 512,
    max_shards: int | None = None,
    num_workers: int = 4,
    device: str | None = None,
    seed: int = 42,
):
    """Score a saved checkpoint on the val/test split: MAE + R² per target and
    stability ROC-AUC/AP. Single-process (no DDP); mirrors train()'s split/norm.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    # infer architecture from checkpoint (fall back to defaults if absent)
    hp = ckpt.get("hparams", {})
    model = Predictor(
        mul=hp.get("mul", 128),
        n_layers=hp.get("n_layers", 2),
        correlation=hp.get("correlation", 3),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.set_normalization(ckpt["mean"].to(device), ckpt["std"].to(device))

    dataset = ShardedCrystalDataset(data_root, max_shards=max_shards)
    tr, va, te = split_indices(len(dataset), seed=seed)
    idx = {"train": tr, "val": va, "test": te}[split]
    loader = DataLoader(
        Subset(dataset, idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )
    print(f"[eval] {ckpt_path} on {split} split ({len(idx):,} graphs), stage {stage}", flush=True)
    metrics = evaluate(model, loader, device, stage=stage, distributed=False)
    st = metrics["stability"]
    print(f"[eval] val_loss={metrics['val_loss']:.4f}")
    print(f"[eval] {'target':<28}{'MAE':>12}{'R2':>10}")
    for k in PREDICT_KEYS:
        print(f"[eval]   {_LABELS.get(k, k):<26}{metrics['mae'][k]:>12.4f}{metrics['r2'][k]:>10.3f}")
    print(
        f"[eval] stability (Ehull>{0.0:.2f}): AUC={st['auc']:.3f} AP={st['ap']:.3f} "
        f"frac_unstable={st['frac_unstable']:.3f}"
    )
    return metrics
