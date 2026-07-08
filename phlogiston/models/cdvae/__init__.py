"""CDVAE ab-initio crystal generator. See DESIGN.md."""

from phlogiston.models.cdvae import diffusion
from phlogiston.models.cdvae.cdvae import CDVAE
from phlogiston.models.cdvae.conditioning import (
    DEFAULT_PROFILE,
    LatentPropertyHead,
    fit_latent_property_head,
    generate_conditioned,
    optimize_latent,
)
from phlogiston.models.cdvae.decoder import CDVAEDecoder, ScoreOutput
from phlogiston.models.cdvae.encoder import CDVAEEncoder, VAEOutput
from phlogiston.models.cdvae.predictors import LatentPrediction, LatentPredictors

__all__ = [
    "CDVAE",
    "CDVAEEncoder",
    "VAEOutput",
    "LatentPredictors",
    "LatentPrediction",
    "CDVAEDecoder",
    "ScoreOutput",
    "diffusion",
    "LatentPropertyHead",
    "fit_latent_property_head",
    "optimize_latent",
    "generate_conditioned",
    "DEFAULT_PROFILE",
]
