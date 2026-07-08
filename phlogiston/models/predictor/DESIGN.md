# Predictor — DESIGN

The shared `CrystalEncoder` + readout heads that predict stability and
mechanical/thermal properties, trained with **schedule B** (pretrain encoder +
stability on all ~629k structures, then low-LR fine-tune property heads on the
~12k mechanically-labeled ones). See `pipeline.md` §4–5.

---

## 0. Targets

`PREDICT_KEYS` = the model's outputs, aligned with a slice of `TARGET_KEYS`.
**Density is excluded** — it is analytic from the structure, so there is no
reason to learn it.

```
PREDICT_KEYS = [
  formation_energy_per_atom,   # stability  (stage 1)
  energy_above_hull,           # stability  (stage 1)
  bulk_modulus_vrh,            # mechanical (stage 2)
  shear_modulus_vrh,           # mechanical (stage 2)
  vickers_hardness,            # mechanical (stage 2)
  fracture_toughness,          # mechanical (stage 2)
  debye_temperature,           # thermal    (stage 2)
  slack_thermal_conductivity,  # thermal    (stage 2)
]
```

All targets are **intensive** (per-atom energies / per-material properties), so
readout pools by **mean** over atoms. Each target carries a mask (§3).

## 1. Architecture

```
BatchedGraph
   │
   ▼
CrystalEncoder ──► node_feats [N, mul]  (invariant per-atom scalars)
   │
   ▼
Head (ScalarReadout, reduce="mean") ──► ŷ_norm [B, n_targets]
   │  (mean-pool per graph + MLP)
   ▼
de-standardize ──► ŷ [B, n_targets]  in physical units
```

- **Heads**: one **independent head per target** — a `ModuleList` of
  `layers.ScalarReadout(irreps=mul x0e, n_out=1, hidden=(mul,), reduce="mean")`,
  one per `PREDICT_KEYS` entry; outputs stacked to `[B, n_targets]`. Independent
  heads let each property (energies vs moduli vs thermal) specialize, and make
  schedule B clean: the stability heads and property heads are separate modules,
  so a stage simply enables/freezes the relevant ones (no shared trunk to
  disentangle).
- Output is in **standardized** space; the model stores per-target `mean`/`std`
  buffers and de-standardizes at inference.

## 2. Target normalization

- Compute per-target `mean`, `std` over the **train split** (masked entries only)
  once, store as buffers.
- Train the head to predict `(y − mean)/std`; report/serve `ŷ·std + mean`.
- Energies are already per-atom (intensive); no extra atom-count handling.
- **log1p targets** (`LOG_TARGETS` = Vickers hardness, Slack κ): these are
  strongly nonlinear derived quantities (Hv ∝ G^0.585 with a subtraction; κ ∝
  θ_D³/γ²) with wide, right-skewed range. They are learned in `log1p` space:
  `mean`/`std` are computed over `to_transform(y)`, the head predicts the
  standardized log1p value, and `forward` applies `expm1` so outputs (and thus
  MAE/R²) are back in physical units. `to_transform`/`from_transform` handle the
  per-column mapping; `log_mask` is a non-persistent buffer (constant), so older
  checkpoints still load. Empirically this lifts Hv/κ R² (their weak spot under
  linear-space training).

## 3. Masked multi-task loss

No material has all labels, so the loss only counts present targets:

```
L = Σ_t  w_t · ( Σ_b mask[b,t] · huber(ŷ_norm[b,t], y_norm[b,t]) ) / (Σ_b mask[b,t] + ε)
```

- `huber` (smooth-L1) — robust to label outliers; `MSE` as an alternative.
- `w_t`: per-target weights to balance abundant stability vs scarce mechanical
  labels (and different target scales, mostly handled by normalization).
- Per-target mean over *masked* samples so a target isn't down-weighted just
  because it is rare in a batch.

## 4. Interface (planned)

```
class Predictor(nn.Module):
    def __init__(self, encoder_cfg, n_targets=len(PREDICT_KEYS), head_hidden=(mul,)): ...
    def set_normalization(mean[T], std[T]): ...           # buffers from train stats
    def forward(graph) -> Tensor[B, n_targets]            # de-standardized
    def loss(pred, y, mask, weights) -> (total, per_target_dict)
    def stage1_parameters() / stage2_parameters()         # param groups for schedule B
```

`y`, `mask` are the `PREDICT_KEYS` slice of the batch's `y`/`y_mask`.

## 5. Schedule B (training; drivers live in `phlogiston/train`)

- **Stage 1** — pretrain `encoder` + the two stability outputs on all ~629k
  (masked to the stability columns). Yields a strong general encoder + stability
  predictor.
- **Stage 2** — enable the mechanical/thermal outputs; fine-tune encoder at a
  **low LR** (`stage2_parameters` splits encoder vs head LRs) on the ~12k
  labeled set. Early-stop on a held-out property val split; weight decay to
  resist overfitting the small set.
- Metrics: per-target MAE (physical units), stability classification (e_above_hull
  ≤ τ) AUC/F1, parity plots, per-chemistry breakdown. Anchor sanity vs Pd/diamond.

## 6. Open decisions
- **Resolved**: independent per-target heads (not a shared trunk) — see §1.
- `log1p` transform for κ / hardness / toughness (from label distributions).
- Loss: Huber δ; per-target weights `w_t`.
- Predict `energy_above_hull` directly (label available) vs derive from
  `formation_energy` against a stored convex hull.
- Stage-2: low-LR encoder fine-tune vs partial freeze (decide from val curves).
