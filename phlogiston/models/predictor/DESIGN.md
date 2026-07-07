# Predictor — DESIGN

`encoder` + readout heads for stability and mechanical/thermal properties.
Trained with **schedule B** (pretrain encoder + stability on 629k, then low-LR
fine-tune property heads on 12k). See `pipeline.md` §4–5.

> Status: **skeleton** — head architecture, target normalization, and loss
> details to be filled next.

## Heads (over TARGET_KEYS; density is analytic, no head)
- Stage 1: `formation_energy_per_atom`, `energy_above_hull`
- Stage 2: `bulk_modulus_vrh`, `shear_modulus_vrh`, `vickers_hardness`,
  `fracture_toughness`, `debye_temperature`, `slack_thermal_conductivity`

## To specify
- Readout: pooling (sum/mean), per-head MLP width/depth.
- Per-target standardization (train stats); per-atom vs intensive handling.
- Masked multi-task loss (Huber/MSE), per-target weights.

## Open decisions
- Stage-2: low-LR encoder fine-tune vs partial freeze.
