"""Tests for CDVAE diffusion utilities. Run: python -m tests.test_cdvae_diffusion"""

from __future__ import annotations

import sys

import torch

from phlogiston.models.cdvae import diffusion as D

_results: list[tuple[str, bool, str]] = []


def _check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_schedule():
    s = D.geometric_sigmas(0.01, 10.0, 50)
    ok = (
        s.shape == (50,)
        and abs(s[0].item() - 10.0) < 1e-4
        and abs(s[-1].item() - 0.01) < 1e-4
        and bool((s[:-1] > s[1:]).all())
    )
    _check("geometric schedule (descending, endpoints)", ok, f"[{s[0]:.2f}..{s[-1]:.3f}]")


def test_perturb_scale():
    torch.manual_seed(0)
    cart = torch.zeros(20000, 3)
    sigma = torch.full((20000, 1), 0.5)
    noisy, eps = D.perturb_positions(cart, sigma)
    _check(
        "perturb noise scale ~ sigma",
        abs(noisy.std().item() - 0.5) < 0.02,
        f"std={noisy.std().item():.3f}",
    )


def test_dsm_loss_zero_at_truth():
    torch.manual_seed(0)
    eps = torch.randn(100, 3)
    sigma = torch.rand(100, 1) + 0.1
    true_score = -eps / sigma  # exact noise-kernel score
    loss = D.dsm_loss(true_score, eps, sigma)
    _check("DSM loss ~0 at the true score", loss.item() < 1e-10, f"loss={loss.item():.2e}")
    worse = D.dsm_loss(torch.zeros_like(eps), eps, sigma)
    _check("DSM loss > 0 for a wrong score", worse.item() > loss.item())


def test_langevin_moves_along_score():
    torch.manual_seed(0)
    cart = torch.zeros(1000, 3)
    score = torch.ones(1000, 3)  # push +
    out = D.langevin_step(cart, score, sigma=0.1, sigma_min=0.01, step_factor=2e-5)
    _check(
        "Langevin step follows the score (mean drift +)",
        out.mean().item() > 0,
        f"mean={out.mean().item():.2e}",
    )


if __name__ == "__main__":
    test_schedule()
    test_perturb_scale()
    test_dsm_loss_zero_at_truth()
    test_langevin_moves_along_score()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    sys.exit(1 if n_fail else 0)
