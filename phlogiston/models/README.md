# `phlogiston.models` — assembled models

Each model is a subpackage that composes `phlogiston.layers` blocks and carries
its own `DESIGN.md` (detailed architecture, shapes, hyperparameters, and the
open decisions it resolves). `pipeline.md` stays high-level and links here.

## Index

| Subpackage | Role | Design |
|---|---|---|
| `encoder/` | shared E(3)-equivariant crystal-graph encoder | `encoder/DESIGN.md` |
| `predictor/` | encoder + stability & property heads (schedule B) | `predictor/DESIGN.md` |
| `cdvae/` | ab-initio generator (VAE encoder + latent predictors + score decoder) | `cdvae/DESIGN.md` |

## Relationships

```
layers/*  ─────►  models/encoder   ─────►  models/predictor   (Phase 4)
                        │
                        └───────────────►  models/cdvae        (Phase 5, separate weights)
```

The `encoder` block design is shared by `predictor` and `cdvae` (same
architecture, separate trained weights).
