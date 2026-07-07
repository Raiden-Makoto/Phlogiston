# `phlogiston.layers` — building blocks

Small, self-contained, independently-testable layers. Models in
`phlogiston.models` compose these; nothing here imports a model. Each non-trivial
layer gets its own detailed spec file (`<layer>.md`) next to its implementation.

## Conventions (contract every layer honors)

- **Irreps notation**: `e3nn` `o3.Irreps`, e.g. `"128x0e + 64x1o + 32x2e"`
  (multiplicity × (ℓ, parity)). Scalars are `0e`.
- **Equivariance contract**: a layer declares `irreps_in → irreps_out`. Under a
  global rotation/inversion `g ∈ O(3)`, outputs transform by the Wigner-D of
  `irreps_out`. Scalars (`0e`) are invariant; vectors (`1o`) rotate.
- **Shapes**: node features `[N, dim(irreps)]`, edges indexed by
  `edge_index [2, E]` (row 0 = center/receiver `i`, row 1 = neighbor/sender `j`),
  `edge_vec [E, 3]`, `edge_len [E]`.
- **Aggregation**: native `torch` scatter over `edge_index[0]` (no `pyg-lib`).
- **Device/dtype**: layers are device-agnostic; ROCm-safe (only `e3nn` + torch).

## Planned components (specs TBD — filled during Phase 4)

| Module | Purpose | Spec |
|---|---|---|
| `radial.py` | Bessel radial basis + smooth polynomial cutoff envelope | `radial.md` |
| `spherical.py` | real spherical harmonics of `edge_vec` (ℓ ≤ L_max) | `spherical.md` |
| `linear.py` | equivariant linear (per-irrep weight mixing) | `linear.md` |
| `interaction.py` | CG tensor-product message passing + scatter aggregate | `interaction.md` |
| `gate.py` | gated equivariant nonlinearity (scalars gate higher ℓ) | `gate.md` |
| `readout.py` | per-atom → graph pooling + scalar head MLP | `readout.md` |
| `embedding.py` | atomic-number embedding (+ optional element descriptors) | `embedding.md` |
| `noise.py` | noise-level / timestep embedding (for the CDVAE decoder) | `noise.md` |

> The precise architecture ("MACE-style E(3)") is defined per-layer in these
> spec files, not in `pipeline.md`.
