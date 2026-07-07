"""Predictor: shared encoder + stability & property heads. See DESIGN.md."""

from phlogiston.models.predictor.predictor import PREDICT_KEYS, STABILITY_KEYS, Predictor

__all__ = ["Predictor", "PREDICT_KEYS", "STABILITY_KEYS"]
