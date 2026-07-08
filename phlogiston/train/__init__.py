"""Training drivers (schedule B). See phlogiston/models/predictor/DESIGN.md §5."""

from phlogiston.train.trainer import (
    compute_normalization,
    evaluate,
    evaluate_checkpoint,
    split_indices,
    train,
)

__all__ = [
    "train",
    "evaluate",
    "evaluate_checkpoint",
    "split_indices",
    "compute_normalization",
]
