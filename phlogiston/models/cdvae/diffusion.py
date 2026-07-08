"""CDVAE diffusion utilities (see DESIGN.md §5).

Score-based diffusion over atom coordinates: a geometric noise schedule, a
denoising-score-matching loss, and an annealed-Langevin sampling step. The score
network (decoder) learns ``s_theta ≈ -eps/sigma`` for coords perturbed by
``x~ = x + sigma·eps``.

v1 uses plain Gaussian noise on Cartesian coordinates; the periodic
wrapped-Gaussian target is a documented refinement (DESIGN §11).
"""

from __future__ import annotations

import torch


def geometric_sigmas(
    sigma_min: float = 0.01, sigma_max: float = 10.0, n_levels: int = 50
) -> torch.Tensor:
    """Descending geometric noise levels [sigma_max ... sigma_min]."""
    return torch.exp(
        torch.linspace(
            torch.log(torch.tensor(sigma_max)).item(),
            torch.log(torch.tensor(sigma_min)).item(),
            n_levels,
        )
    )


def sample_sigma(
    sigmas: torch.Tensor, batch_size: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Random noise level per graph, drawn from the schedule. Shape [batch_size]."""
    idx = torch.randint(0, len(sigmas), (batch_size,), generator=generator)
    return sigmas[idx]


def perturb_positions(cart: torch.Tensor, sigma_node: torch.Tensor):
    """Add Gaussian noise scaled by per-node sigma. Returns (noisy, eps)."""
    eps = torch.randn_like(cart)
    return cart + sigma_node * eps, eps


def dsm_loss(score: torch.Tensor, eps: torch.Tensor, sigma_node: torch.Tensor) -> torch.Tensor:
    """Denoising score-matching loss E[|| sigma·s + eps ||^2] (sigma^2-weighted).

    Zero exactly when ``score == -eps / sigma`` (the true noise-kernel score).
    """
    return ((sigma_node * score + eps) ** 2).sum(dim=-1).mean()


def langevin_step(
    cart: torch.Tensor,
    score: torch.Tensor,
    sigma: float,
    sigma_min: float,
    step_factor: float = 2e-5,
) -> torch.Tensor:
    """One annealed-Langevin update at noise level ``sigma``."""
    alpha = step_factor * (sigma / sigma_min) ** 2
    return cart + 0.5 * alpha * score + (alpha**0.5) * torch.randn_like(cart)
