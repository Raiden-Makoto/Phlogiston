"""Property-conditioned generation via latent optimization (DESIGN.md §6.5, §8).

The CDVAE is trained unconditionally; conditioning is added on top by (1) fitting
a small property head ``f_p(z)`` on the *frozen* trained encoder's latent, then
(2) gradient-ascending ``z`` toward a target property profile before decoding
with the same trained decoder. This is CDVAE's native latent-optimization route
(no decoder retrain), and every candidate is still re-verified by the
independent Predictor screen.

Objective maximized in latent space (all in standardized units):

    J(z) = Σ_t  w_t · f_p(z)_t   −   α · ||z||²

``w_t`` are signed weights over PREDICT_KEYS (positive to maximize a target,
negative for energy_above_hull so it's minimized). The ``α·||z||²`` term is a
Gaussian-prior / trust-region penalty that keeps ``z`` in the region the decoder
was trained on (otherwise optimization drifts off-manifold and decodes garbage).
"""

from __future__ import annotations

import torch
from torch import nn

from phlogiston.data.dataset import TARGET_KEYS
from phlogiston.models.predictor import PREDICT_KEYS

_PRED_IDX = [TARGET_KEYS.index(k) for k in PREDICT_KEYS]

# Default "light + strong + tough + heat-resistant" profile (signed, standardized
# space): maximize the mechanical/thermal targets, minimize energy_above_hull,
# ignore formation energy. Density (lightness) is handled by the screen's ceiling.
DEFAULT_PROFILE: dict[str, float] = {
    "energy_above_hull": -1.0,
    "bulk_modulus_vrh": 1.0,
    "shear_modulus_vrh": 1.0,
    "vickers_hardness": 1.0,
    "fracture_toughness": 1.0,
    "debye_temperature": 1.0,
    "slack_thermal_conductivity": 1.0,
}


class LatentPropertyHead(nn.Module):
    """MLP f_p: z -> standardized PREDICT_KEYS targets. Stores target mean/std
    (physical) so predictions can be reported in physical units."""

    def __init__(self, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_targets = len(PREDICT_KEYS)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.n_targets),
        )
        self.register_buffer("target_mean", torch.zeros(self.n_targets))
        self.register_buffer("target_std", torch.ones(self.n_targets))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Standardized target predictions [B, n_targets]."""
        return self.net(z)

    def predict_physical(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z) * self.target_std + self.target_mean


def profile_weights(profile: dict[str, float] | None = None) -> torch.Tensor:
    """Signed weight vector over PREDICT_KEYS (0 for unlisted targets)."""
    p = profile or DEFAULT_PROFILE
    return torch.tensor([p.get(k, 0.0) for k in PREDICT_KEYS], dtype=torch.float32)


@torch.no_grad()
def _encode_mu(cdvae, graph) -> torch.Tensor:
    """Deterministic latent (posterior mean) for a batch."""
    return cdvae.encoder(graph).mu


def fit_latent_property_head(
    cdvae,
    data_root: str,
    *,
    device: str | None = None,
    hidden: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_shards: int | None = None,
    num_workers: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> LatentPropertyHead:
    """Fit f_p(z) on the frozen CDVAE encoder over the property-labeled subset.

    Encodes each labeled structure to its posterior-mean latent (deterministic),
    then fits the head with a masked MSE on standardized targets.
    """
    from torch.utils.data import DataLoader, Subset

    from phlogiston.data.dataset import ShardedCrystalDataset, collate
    from phlogiston.train.trainer import labeled_indices, split_indices

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cdvae = cdvae.to(device).eval()
    for p in cdvae.parameters():
        p.requires_grad_(False)

    ds = ShardedCrystalDataset(data_root, max_shards=max_shards)
    idx = labeled_indices(ds, stage=2)
    ds = Subset(ds, idx)
    tr, va, _ = split_indices(len(ds), seed=seed)
    pred_idx = torch.tensor(_PRED_IDX, device=device)

    def log(m):
        if verbose:
            print(m, flush=True)

    log(f"[cond] fitting latent property head on {len(idx):,} labeled ({len(tr):,} train)")

    # 1) encode train latents + collect standardized-target stats
    loader_kw = dict(collate_fn=collate, num_workers=num_workers)
    tr_loader = DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=False, **loader_kw)
    zs, ys, ms = [], [], []
    for batch in tr_loader:
        batch = batch.to(device)
        zs.append(_encode_mu(cdvae, batch).detach())
        ys.append(batch.y[:, pred_idx].detach())
        ms.append(batch.y_mask[:, pred_idx].detach())
    Z, Y, M = torch.cat(zs), torch.cat(ys), torch.cat(ms).float()

    cnt = M.sum(0).clamp(min=1)
    mean = (Y * M).sum(0) / cnt
    var = ((Y * Y) * M).sum(0) / cnt - mean * mean
    std = var.clamp(min=1e-8).sqrt()

    head = LatentPropertyHead(cdvae.latent_dim, hidden=hidden).to(device)
    head.target_mean.copy_(mean)
    head.target_std.copy_(std)

    Yn = (Y - mean) / std  # standardized targets
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        head.train()
        opt.zero_grad()
        pred = head(Z)
        per = ((pred - Yn) ** 2) * M
        loss = per.sum() / M.sum().clamp(min=1)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 25 == 0:
            log(f"[cond]   epoch {ep + 1}/{epochs} masked MSE={loss.item():.4f}")
    return head.eval()


def optimize_latent(
    head: LatentPropertyHead,
    n: int,
    *,
    profile: dict[str, float] | None = None,
    steps: int = 200,
    lr: float = 0.1,
    alpha: float = 0.1,
    device: str | None = None,
    z0: torch.Tensor | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Gradient-ascend ``n`` latents toward the target profile. Returns z [n, d]."""
    device = device or next(head.parameters()).device
    w = profile_weights(profile).to(device)
    if z0 is None:
        g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        z0 = torch.randn(n, head.latent_dim, generator=g, device=device)
    z = z0.clone().detach().to(device).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = head(z)  # standardized [n, T]
        objective = (pred * w).sum(dim=-1) - alpha * (z * z).sum(dim=-1)
        (-objective.sum()).backward()  # ascent = minimize negative
        opt.step()
    return z.detach()


@torch.no_grad()
def generate_conditioned(
    cdvae,
    head: LatentPropertyHead,
    n: int,
    *,
    profile: dict[str, float] | None = None,
    steps: int = 200,
    lr: float = 0.1,
    alpha: float = 0.1,
    steps_per_level: int = 4,
    device: str | None = None,
) -> list:
    """Optimize latents toward the profile, then decode each into a Structure."""
    device = device or next(cdvae.parameters()).device
    with torch.enable_grad():
        Z = optimize_latent(head, n, profile=profile, steps=steps, lr=lr, alpha=alpha, device=device)
    from phlogiston.models.cdvae.sampler import batched_sample

    return batched_sample(cdvae, Z, steps_per_level=steps_per_level)
