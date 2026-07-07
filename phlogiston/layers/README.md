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

## Components

Encoder blocks — **specs written** (`<layer>.md` beside this file); code next.

| Module | Purpose | Spec |
|---|---|---|
| `embedding` | atomic-number embedding (+ optional element descriptors) | [embedding.md](embedding.md) ✅ |
| `radial` | Bessel radial basis + smooth polynomial cutoff + weight MLP | [radial.md](radial.md) ✅ |
| `spherical` | real spherical harmonics of `edge_vec` (ℓ ≤ L_sh) | [spherical.md](spherical.md) ✅ |
| `linear` | equivariant linear (+ species-dependent skip) | [linear.md](linear.md) ✅ |
| `interaction` | A-basis + symmetric contraction (body order ν) + message | [interaction.md](interaction.md) ✅ |
| `gate` | gated equivariant nonlinearity (scalars gate higher ℓ) | [gate.md](gate.md) ✅ |
| `readout` | scalar readout + graph pooling | [readout.md](readout.md) ✅ |

Deferred (belongs to the CDVAE model, spec'd in that phase):

| Module | Purpose | Spec |
|---|---|---|
| `noise` | noise-level / timestep embedding (CDVAE decoder) | (deferred) |

> The precise architecture is defined per-layer in these spec files and assembled
> in `models/encoder/DESIGN.md`, not in `pipeline.md`.
