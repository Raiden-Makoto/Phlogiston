"""uMLIP backend adapters for Tier-2 verification (DESIGN.md §2).

Each supported foundation potential is exposed as an ASE ``Calculator`` so that
relaxation (``relax.py``), phonons (``phonons.py``), and the ensemble
cross-check (``ensemble.py``) all drive it through one interface, regardless of
the underlying model. Every backend here is **MPtrj-trained**, so its energies
are Materials-Project-frame comparable — the property that lets us place a
candidate on the MP convex hull without any first-principles recomputation.

Backends
--------
- ``chgnet``    — CHGNet (default/primary). Ships an ASE calculator, bundles its
  weights (no network), and is e3nn-free so it coexists with our from-scratch
  models' ``e3nn>=0.5`` requirement.
- ``mattersim`` — MatterSim (independent cross-check). Ships an ASE calculator;
  downloads its pretrained weights on first use (needs network once, then
  cached under ``~/.local/mattersim``).

Two potentials excluded for compatibility (see requirements.txt): **MACE**
hard-pins ``e3nn==0.4.4`` (conflicts with our ``e3nn>=0.5``), and **ORB** 0.7.0
dropped its ASE calculator and depends on CUDA-only ``warp-lang``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

# Registry of backends this module knows how to construct. Order = ensemble
# preference; the first available one is the default primary.
SUPPORTED_BACKENDS: tuple[str, ...] = ("chgnet", "mattersim")
DEFAULT_BACKEND: str = "chgnet"


def resolve_device(device: str | None = None) -> str:
    """Pick a torch device string. ``None`` -> GPU if visible (ROCm shows up as
    ``cuda`` under PyTorch), else CPU."""
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_calculator(
    backend: str = DEFAULT_BACKEND,
    *,
    device: str | None = None,
    **kwargs: Any,
) -> Calculator:
    """Construct an ASE ``Calculator`` for ``backend`` on ``device``.

    Extra kwargs are forwarded to the underlying calculator constructor.
    """
    key = backend.lower()
    dev = resolve_device(device)
    if key == "chgnet":
        return _chgnet_calculator(device=dev, **kwargs)
    if key == "mattersim":
        return _mattersim_calculator(device=dev, **kwargs)
    raise ValueError(
        f"Unknown uMLIP backend {backend!r}; supported: {SUPPORTED_BACKENDS}"
    )


def available_backends() -> dict[str, bool]:
    """Which backends are importable in this environment (best-effort probe)."""
    status: dict[str, bool] = {}
    for name, module in (("chgnet", "chgnet"), ("mattersim", "mattersim")):
        try:
            __import__(module)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def _chgnet_calculator(*, device: str, **kwargs: Any) -> Calculator:
    """CHGNet as an ASE calculator. ``model=None`` loads the packaged pretrained
    (MPtrj) weights. ``use_device`` accepts ``cuda``/``cpu``."""
    from chgnet.model.dynamics import CHGNetCalculator

    return CHGNetCalculator(use_device=device, **kwargs)


def _mattersim_calculator(*, device: str, **kwargs: Any) -> Calculator:
    """MatterSim as an ASE calculator. ``potential=None`` loads (and, on first
    use, downloads + caches) the pretrained weights. ``device`` accepts
    ``cuda``/``cpu``."""
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(device=device, **kwargs)
