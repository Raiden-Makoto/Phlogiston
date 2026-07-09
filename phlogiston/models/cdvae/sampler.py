"""GPU-native batched annealed-Langevin sampler for the CDVAE.

The reference ``CDVAE.generate`` samples one structure at a time and rebuilds the
crystal graph with pymatgen (CPU) at every Langevin step -- tens of thousands of
serial CPU neighbor searches with the GPU idle. This sampler instead:

  * builds ONE batched graph over all structures so the decoder denoises them in
    a single forward (GPU saturated), and
  * does periodic neighbor search in torch on the GPU (candidate pairs +
    image offsets are precomputed once per structure; only edge vectors and the
    cutoff mask are recomputed each step),

keeping coordinates, Langevin updates, and cell-wrapping entirely on device.
pymatgen is touched only once at the end to export Structures.
"""

from __future__ import annotations

import torch

from phlogiston.models.cdvae.decoder import ScoreOutput


class _Batch:
    """Minimal batched-graph view the decoder consumes (all fields on device)."""

    def __init__(self, z, batch, edge_index, edge_vec, edge_len):
        self.z = z
        self.batch = batch
        self.edge_index = edge_index
        self.edge_vec = edge_vec
        self.edge_len = edge_len


def _image_range(lattice: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Per-axis number of periodic images needed to cover ``cutoff`` [3] ints."""
    recip = torch.linalg.inv(lattice).transpose(-1, -2)  # rows = reciprocal vecs (no 2pi)
    inv_spacing = recip.norm(dim=-1)  # 1 / interplanar spacing per axis
    return torch.ceil(cutoff * inv_spacing).clamp(min=1).to(torch.int64)


def _candidate_pairs(n: int, lattice: torch.Tensor, cutoff: float, device):
    """All (src, dst, cart_offset) intra-cell pairs within the image range that
    could fall inside ``cutoff`` (self-pairs kept only for nonzero images).
    Computed once per structure; the per-step cutoff mask does the filtering."""
    k = _image_range(lattice, cutoff).tolist()
    ranges = [torch.arange(-k[d], k[d] + 1, device=device) for d in range(3)]
    na, nb, nc = (r.numel() for r in ranges)
    grid = torch.cartesian_prod(*ranges).float()  # [I, 3] integer image vectors
    offsets = grid @ lattice  # [I, 3] cartesian image offsets
    i = torch.arange(n, device=device)
    src, dst = torch.meshgrid(i, i, indexing="ij")
    src, dst = src.reshape(-1), dst.reshape(-1)  # [N*N]
    n_img = grid.shape[0]
    src = src.repeat(n_img)
    dst = dst.repeat(n_img)
    off = offsets.repeat_interleave(n * n, dim=0)  # [N*N*I, 3]
    zero_img = (grid.abs().sum(-1) == 0).repeat_interleave(n * n)
    self_pair = src == dst
    keep = ~(self_pair & zero_img)  # drop the trivial i==i at image 0
    _ = (na, nb, nc)
    return src[keep], dst[keep], off[keep]


@torch.no_grad()
def batched_sample(
    cdvae,
    z: torch.Tensor,
    *,
    steps_per_level: int = 8,
    cutoff: float = 6.0,
    n_atoms: list[int] | None = None,
):
    """Decode a batch of latents ``z`` [B, d] into pymatgen Structures via a
    single batched annealed-Langevin trajectory on the GPU."""
    import numpy as np
    from pymatgen.core import Structure

    device = next(cdvae.parameters()).device
    z = z.to(device)
    b = z.shape[0]
    pred = cdvae.predictors(z)

    # per-structure size, lattice matrix, and initial species (from composition)
    counts = (
        [max(1, min(int(x), cdvae.n_max)) for x in n_atoms]
        if n_atoms is not None
        else [int(pred.num_atoms_logits[i].argmax()) + 1 for i in range(b)]
    )
    lattices, species, fracs, batch_id = [], [], [], []
    for i in range(b):
        lat = cdvae._lattice_from_params(pred.lattice[i])
        lattices.append(torch.tensor(lat.matrix, dtype=torch.float32, device=device))
        comp = torch.softmax(pred.composition_logits[i], dim=-1)
        sp = torch.multinomial(comp, counts[i], replacement=True) + 1  # Z >= 1
        species.append(sp)
        fracs.append(torch.rand(counts[i], 3, device=device))
        batch_id.append(torch.full((counts[i],), i, dtype=torch.long, device=device))

    L = torch.stack(lattices)  # [B, 3, 3]
    Z = torch.cat(species).to(device)  # [T]
    batch = torch.cat(batch_id)  # [T]
    frac = torch.cat(fracs)  # [T, 3]
    node_lat = L[batch]  # [T, 3, 3]
    node_offset = torch.tensor([0, *np.cumsum(counts).tolist()[:-1]], device=device)

    # precompute candidate pairs per structure (global node indices), once
    src_all, dst_all, off_all = [], [], []
    for i in range(b):
        s, d, o = _candidate_pairs(counts[i], L[i], cutoff, device)
        src_all.append(s + node_offset[i])
        dst_all.append(d + node_offset[i])
        off_all.append(o)
    src = torch.cat(src_all)
    dst = torch.cat(dst_all)
    off = torch.cat(off_all)

    def cart():
        return torch.einsum("ni,nij->nj", frac, node_lat)  # frac -> cartesian

    sigmas = cdvae.sigmas.tolist()
    for sigma in sigmas:
        sig_vec = torch.full((b,), sigma, device=device)
        alpha = 2e-5 * (sigma / cdvae.sigma_min) ** 2  # matches diffusion.langevin_step
        for _ in range(steps_per_level):
            pos = cart()
            edge_vec = pos[dst] + off - pos[src]
            edge_len = edge_vec.norm(dim=-1)
            m = (edge_len < cutoff) & (edge_len > 1e-3)
            if not m.any():
                break
            graph = _Batch(Z, batch, torch.stack([src[m], dst[m]]), edge_vec[m], edge_len[m])
            out: ScoreOutput = cdvae.decoder(graph, z, sig_vec)
            pos = pos + 0.5 * alpha * out.coord_score + (alpha**0.5) * torch.randn_like(pos)
            # wrap back into each cell (cartesian -> frac, mod 1)
            frac = torch.einsum("nj,nij->ni", pos, torch.linalg.inv(node_lat)) % 1.0

    # export
    structures = []
    for i in range(b):
        sl = batch == i
        structures.append(
            Structure(
                lattice=L[i].cpu().numpy(),
                species=Z[sl].cpu().tolist(),
                coords=frac[sl].cpu().numpy(),
                coords_are_cartesian=False,
            )
        )
    return structures
