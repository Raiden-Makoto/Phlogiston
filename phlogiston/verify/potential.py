"""uMLIP backend adapters for Tier-2 verification (DESIGN.md §2).

Each supported foundation potential is exposed as an ASE ``Calculator`` so that
relaxation (``relax.py``), phonons (``phonons.py``), and the ensemble
cross-check (``ensemble.py``) all drive it through one interface, regardless of
the underlying model. Every backend here is **MPtrj-trained**, so its energies
are Materials-Project-frame comparable — the property that lets us place a
candidate on the MP convex hull without any first-principles recomputation.

Backends
--------
- ``chgnet`` — CHGNet (default/primary). Ships an ASE calculator; e3nn-free, so
  it coexists with our from-scratch models' ``e3nn>=0.5`` requirement.
- ``orb``    — ORB. **Not yet wired**: ``orb-models`` 0.7.0 dropped its ASE
  ``Calculator`` and changed the loader API (loaders now return a
  ``(model, adapter)`` tuple), and it pulls in ``warp-lang`` which cannot use
  ROCm GPUs. Needs a small custom ASE wrapper (or a pinned older release) before
  it can join the ensemble. See ``_orb_calculator``.

MACE is intentionally absent: it hard-pins ``e3nn==0.4.4``, which conflicts with
our verified ``e3nn>=0.5`` stack (see requirements.txt).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

# Registry of backends this module knows how to construct. Order = ensemble
# preference; the first available one is the default primary.
SUPPORTED_BACKENDS: tuple[str, ...] = ("chgnet", "orb")
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
    if key == "orb":
        return _orb_calculator(device=dev, **kwargs)
    raise ValueError(
        f"Unknown uMLIP backend {backend!r}; supported: {SUPPORTED_BACKENDS}"
    )


def available_backends() -> dict[str, bool]:
    """Which backends are importable in this environment (best-effort probe)."""
    status: dict[str, bool] = {}
    try:
        import chgnet  # noqa: F401

        status["chgnet"] = True
    except Exception:
        status["chgnet"] = False
    # ORB is installable but not yet wired to ASE here; report False until it is.
    status["orb"] = False
    return status


def _chgnet_calculator(*, device: str, **kwargs: Any) -> Calculator:
    """CHGNet as an ASE calculator. ``model=None`` loads the packaged pretrained
    (MPtrj) weights. ``use_device`` accepts ``cuda``/``cpu``."""
    from chgnet.model.dynamics import CHGNetCalculator

    return CHGNetCalculator(use_device=device, **kwargs)


def _orb_calculator(*, device: str, **kwargs: Any) -> Calculator:
    raise NotImplementedError(
        "ORB backend is not wired yet. orb-models 0.7.0 removed its ASE "
        "Calculator and changed the loader API (pretrained.* now returns a "
        "(model, adapter) tuple), and it depends on warp-lang which has no ROCm "
        "GPU path. Options: (a) write a thin ASE Calculator around the 0.7.0 "
        "model+adapter, (b) pin an older orb-models that ships ORBCalculator, or "
        "(c) swap ORB for another MPtrj-trained, ASE-ready potential. Until then "
        "the ensemble runs with chgnet only."
    )
