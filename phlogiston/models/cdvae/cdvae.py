"""CDVAE assembly (see DESIGN.md §6-7).

Ties the VAE encoder, latent predictors, score decoder, and diffusion utilities
into one model: a composite training loss and an ab-initio ``generate()``.

Coordinate score matching perturbs node positions with per-node noise; because
the stored ``edge_vec`` already bakes in periodic-image offsets, the noisy edge
vectors are ``edge_vec + delta[j] - delta[i]`` — no neighbor rebuild needed
during training.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from phlogiston.models.cdvae import diffusion as D
from phlogiston.models.cdvae.decoder import CDVAEDecoder
from phlogiston.models.cdvae.encoder import CDVAEEncoder
from phlogiston.models.cdvae.predictors import LatentPredictors

_LAT_SCALE = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])  # rough length/angle scales


def _lattice_params(L: torch.Tensor) -> torch.Tensor:
    """Lattice matrices [B,3,3] -> [B,6] (a,b,c, alpha,beta,gamma in degrees)."""
    lengths = L.norm(dim=-1)  # [B,3]
    a, b, c = L[:, 0], L[:, 1], L[:, 2]

    def angle(u, v):
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-8)
        return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    angles = torch.stack([angle(b, c), angle(a, c), angle(a, b)], dim=-1)
    return torch.cat([lengths, angles], dim=-1)


class CDVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        mul: int = 128,
        n_max: int = 64,
        n_elements: int = 100,
        beta: float = 0.01,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        n_levels: int = 50,
        loss_weights: dict | None = None,
        **backbone_kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_max = n_max
        self.n_elements = n_elements
        self.beta = beta
        self.sigma_min = sigma_min
        self.encoder = CDVAEEncoder(latent_dim=latent_dim, mul=mul, **backbone_kwargs)
        self.predictors = LatentPredictors(
            latent_dim=latent_dim, n_max=n_max, n_elements=n_elements
        )
        self.decoder = CDVAEDecoder(
            latent_dim=latent_dim, n_elements=n_elements, mul=mul, **backbone_kwargs
        )
        self.register_buffer("sigmas", D.geometric_sigmas(sigma_min, sigma_max, n_levels))
        self.register_buffer("_lat_scale", _LAT_SCALE.clone())
        self.w = {"num": 1.0, "lattice": 1.0, "composition": 1.0, "coord": 1.0, "type": 1.0}
        if loss_weights:
            self.w.update(loss_weights)

    # ---- targets from a batch ------------------------------------------
    def _targets(self, graph):
        b = graph.batch
        n_graphs = int(b.max()) + 1
        n_atoms = torch.bincount(b, minlength=n_graphs)  # [B]
        num_target = (n_atoms - 1).clamp(0, self.n_max - 1)
        elem_idx = (graph.z - 1).clamp(0, self.n_elements - 1)  # [N]
        comp = torch.zeros(n_graphs, self.n_elements, device=b.device)
        comp.index_put_((b, elem_idx), torch.ones_like(elem_idx, dtype=comp.dtype), accumulate=True)
        comp = comp / comp.sum(-1, keepdim=True).clamp(min=1)
        lat = _lattice_params(graph.lattice) / self._lat_scale
        return num_target, comp, lat, elem_idx

    def forward(self, graph, pairs=None, consistency_weight: float = 0.0):
        """Composite training loss, routed through ``__call__`` so DDP's grad
        all-reduce hooks fire. When ``pairs`` (a ``(relaxed_graph, disp)`` tuple)
        and a positive ``consistency_weight`` are supplied, the
        relaxation-consistency term is added in the *same* forward/backward so its
        gradients all-reduce alongside the reconstruction loss."""
        total, parts = self.training_loss(graph)
        if pairs is not None and consistency_weight > 0:
            pg, disp = pairs
            l_cons = self.consistency_loss(pg, disp)
            total = total + consistency_weight * l_cons
            parts = {**parts, "consistency": l_cons}
        return total, parts

    # ---- training loss --------------------------------------------------
    def training_loss(self, graph):
        vae = self.encoder(graph)
        pred = self.predictors(vae.z)
        num_target, comp_target, lat_target, elem_idx = self._targets(graph)

        l_kl = CDVAEEncoder.kl_loss(vae.mu, vae.logvar)
        l_num = F.cross_entropy(pred.num_atoms_logits, num_target)
        l_lat = F.mse_loss(pred.lattice, lat_target)
        l_comp = -(comp_target * F.log_softmax(pred.composition_logits, dim=-1)).sum(-1).mean()

        # coordinate score matching (perturb node positions -> noisy edge vectors)
        n_graphs = comp_target.shape[0]
        sigma_g = D.sample_sigma(self.sigmas, n_graphs).to(graph.edge_vec)
        sigma_node = sigma_g[graph.batch].unsqueeze(-1)  # [N,1]
        eps = torch.randn(
            graph.z.shape[0], 3, device=graph.edge_vec.device, dtype=graph.edge_vec.dtype
        )
        delta = sigma_node * eps
        noisy_edge_vec = graph.edge_vec + delta[graph.edge_index[1]] - delta[graph.edge_index[0]]
        noisy = dataclasses.replace(
            graph, edge_vec=noisy_edge_vec, edge_len=noisy_edge_vec.norm(dim=-1)
        )
        score = self.decoder(noisy, vae.z, sigma_g)
        # Defensive clamp on the raw score head: at random init (or on rare
        # unstable batches) the decoder's unconstrained equivariant readout can
        # spike to extreme magnitudes, which squared in dsm_loss overflows to
        # inf and poisons the whole step. +-1e4 is generous relative to any
        # legitimate score (~-eps/sigma, at most a few hundred even at
        # sigma_min=0.01), so this only clips genuinely degenerate outputs.
        coord_score = score.coord_score.clamp(-1e4, 1e4)
        l_coord = D.dsm_loss(coord_score, eps, sigma_node)
        l_type = F.cross_entropy(score.type_logits, elem_idx)

        total = (
            self.beta * l_kl
            + self.w["num"] * l_num
            + self.w["lattice"] * l_lat
            + self.w["composition"] * l_comp
            + self.w["coord"] * l_coord
            + self.w["type"] * l_type
        )
        parts = {
            "kl": l_kl,
            "num": l_num,
            "lattice": l_lat,
            "composition": l_comp,
            "coord": l_coord,
            "type": l_type,
        }
        return total, parts

    # ---- relaxation-consistency loss -----------------------------------
    def consistency_loss(self, graph, disp):
        """Denoise the generator's *own* geometry onto the uMLIP minimum.

        ``graph`` is a batch of **relaxed** structures (canonical minima) and
        ``disp`` [Ntot,3] is the per-atom Cartesian displacement
        ``cart_generated - cart_relaxed`` (relaxed-cell frame). Treating the
        generated structure ``G = R + disp`` as a noisy observation of the mode
        ``R``, this is denoising score matching with the *actual* off-manifold
        perturbation instead of synthetic Gaussian noise: the decoder is
        conditioned at an effective per-graph ``sigma`` (the displacement RMS) and
        the score is driven to ``-disp / sigma^2`` -- i.e. to point from the
        generator's guess back to the physical minimum. This directly attacks the
        ~1 A drift that vanilla (train-manifold) score matching leaves behind.
        """
        vae = self.encoder(graph)  # z from the relaxed (clean) structure
        b = graph.batch
        n_graphs = int(b.max()) + 1

        # per-graph effective sigma = RMS(||disp||), clamped into the schedule.
        node_sq = (disp**2).sum(-1)  # [Ntot]
        sum_sq = torch.zeros(n_graphs, device=disp.device, dtype=disp.dtype).index_add(0, b, node_sq)
        cnt = torch.bincount(b, minlength=n_graphs).clamp(min=1).to(disp.dtype)
        sigma_g = (sum_sq / cnt).sqrt().clamp(self.sigma_min, float(self.sigmas[0]))  # [B]
        sigma_node = sigma_g[b].unsqueeze(-1)  # [Ntot,1]

        # place the geometry at G: relaxed edges + (disp[j] - disp[i]).
        noisy_edge_vec = graph.edge_vec + disp[graph.edge_index[1]] - disp[graph.edge_index[0]]
        noisy = dataclasses.replace(
            graph, edge_vec=noisy_edge_vec, edge_len=noisy_edge_vec.norm(dim=-1)
        )
        score = self.decoder(noisy, vae.z, sigma_g)
        coord_score = score.coord_score.clamp(-1e4, 1e4)  # see training_loss
        eps = disp / sigma_node  # so that G = R + sigma_node * eps
        return D.dsm_loss(coord_score, eps, sigma_node)

    # ---- lattice reconstruction ----------------------------------------
    def _lattice_from_params(self, params6: torch.Tensor):
        """[6] normalized (a,b,c,alpha,beta,gamma) -> pymatgen Lattice.

        Clamps lengths/angles to a physical range and rejects degenerate
        (near-singular / zero-volume) cells -- otherwise pymatgen's neighbor
        search divides by a zero cell-matrix determinant. Conditioned latents in
        particular can push the lattice head off-manifold, so fall back to a
        cubic cell of the mean length when the reconstruction is invalid.
        """
        import math

        from pymatgen.core import Lattice

        p = (params6.detach().cpu() * self._lat_scale.cpu()).tolist()
        a, b, c = (min(max(v, 2.0), 50.0) for v in p[:3])
        al, be, ga = (min(max(v, 40.0), 140.0) for v in p[3:])  # keep the cell non-degenerate
        lat = Lattice.from_parameters(a, b, c, al, be, ga)
        if not math.isfinite(lat.volume) or lat.volume < 1e-3:
            lat = Lattice.cubic(max((a + b + c) / 3.0, 2.0))
        return lat

    # ---- batched GPU sampling ------------------------------------------
    @torch.no_grad()
    def sample_batch(self, z=None, n: int = 64, steps_per_level: int = 8, cutoff: float = 6.0,
                     gen_batch_size: int | None = None):
        """GPU-native batched sampler (see sampler.py): decodes ``z`` [B,d] (or
        ``n`` random latents) into Structures in one batched Langevin trajectory.
        Far faster than looping :meth:`sample` (one decoder forward per step for
        the whole batch, torch neighbor search on-device, no per-step pymatgen).

        ``gen_batch_size`` chunks the decode to avoid OOM when ``n`` is large."""
        from phlogiston.models.cdvae.sampler import batched_sample

        if z is None:
            z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return batched_sample(self, z, steps_per_level=steps_per_level, cutoff=cutoff,
                              gen_batch_size=gen_batch_size)

    # ---- full ab-initio sampling ---------------------------------------
    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor | None = None,
        n_atoms: int | None = None,
        steps_per_level: int = 4,
        cutoff: float = 6.0,
    ):
        """Ab-initio: draw ``z``, predict N / lattice / composition from it, then
        denoise coordinates by annealed Langevin. Returns a pymatgen Structure.

        ``n_atoms`` may be given to override the predicted size (useful for
        targeted sizes); otherwise it is taken from the num_atoms head.
        """
        device = next(self.parameters()).device
        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)
        pred = self.predictors(z)
        if n_atoms is None:
            n_atoms = int(pred.num_atoms_logits[0].argmax().item()) + 1
        n_atoms = max(1, min(n_atoms, self.n_max))
        lattice = self._lattice_from_params(pred.lattice[0])
        return self.generate(
            n_atoms=n_atoms, lattice=lattice, z=z,
            steps_per_level=steps_per_level, cutoff=cutoff,
        )

    # ---- ab-initio generation (prototype) ------------------------------
    @torch.no_grad()
    def generate(
        self,
        n_atoms: int,
        lattice,
        z: torch.Tensor | None = None,
        steps_per_level: int = 4,
        cutoff: float = 6.0,
    ):
        """Prototype ab-initio sampler for a single structure with given cell/size.

        (Full pipeline predicts n_atoms/lattice/composition from z; here they are
        provided so generation is well-posed even with an untrained model.)
        """
        from pymatgen.core import Lattice, Structure

        from phlogiston.data.dataset import collate
        from phlogiston.data.graph import structure_to_graph

        device = next(self.parameters()).device
        if z is None:
            z = torch.randn(1, self.latent_dim, device=device)
        pred = self.predictors(z)
        comp = torch.softmax(pred.composition_logits[0], dim=-1)
        species = (torch.multinomial(comp, n_atoms, replacement=True) + 1).tolist()
        frac = torch.rand(n_atoms, 3)
        lat = Lattice(lattice) if not isinstance(lattice, Lattice) else lattice

        for sigma in self.sigmas.tolist():
            for _ in range(steps_per_level):
                struct = Structure(lat, species, frac.tolist())
                try:
                    g = collate(
                        [
                            (
                                structure_to_graph(struct, cutoff=cutoff),
                                torch.zeros(1),
                                torch.zeros(1, dtype=torch.bool),
                            )
                        ]
                    )
                except Exception:  # noqa: BLE001  isolated atoms / degenerate cell
                    break  # stop refining this config; return the current structure
                g = g.to(device)
                score = self.decoder(g, z, torch.tensor([sigma], device=device)).coord_score
                cart = torch.tensor(struct.cart_coords, dtype=torch.float32)
                cart = D.langevin_step(cart, score.cpu(), sigma, self.sigma_min)
                frac = (
                    torch.tensor(lat.get_fractional_coords(cart.numpy()), dtype=torch.float32) % 1.0
                )
        return Structure(lat, species, frac.tolist())
