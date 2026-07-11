"""Tier-2 physics verification (ensemble uMLIP). See DESIGN.md."""

from phlogiston.verify.potential import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    available_backends,
    load_calculator,
    resolve_device,
)
from phlogiston.verify.relax import RelaxResult, relax_structure, relax_structures
from phlogiston.verify.verify import VerifyReport, VerifyRow, verify_registry

__all__ = [
    "DEFAULT_BACKEND",
    "SUPPORTED_BACKENDS",
    "RelaxResult",
    "VerifyReport",
    "VerifyRow",
    "available_backends",
    "load_calculator",
    "relax_structure",
    "relax_structures",
    "resolve_device",
    "verify_registry",
]
