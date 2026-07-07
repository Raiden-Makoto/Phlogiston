# `phlogiston.layers` — building blocks

Small, self-contained, independently-testable layers. Models in
`phlogiston.models` compose these; nothing here imports a model.

**Layout**: implementations in `src/`, detailed specs in `docs/`. Public classes
are re-exported from the package, so import as
`from phlogiston.layers import SphericalHarmonics`.

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

| Module | Purpose | Spec | Code |
|---|---|---|---|
| `embedding` | atomic-number embedding (+ optional element descriptors) | [docs/embedding.md](docs/embedding.md) | `src/embedding.py` ✅ |
| `radial` | Bessel radial basis + smooth polynomial cutoff + weight MLP | [docs/radial.md](docs/radial.md) | `src/radial.py` ✅ |
| `spherical` | real spherical harmonics of `edge_vec` (ℓ ≤ L_sh) | [docs/spherical.md](docs/spherical.md) | `src/spherical.py` ✅ |
| `linear` | equivariant linear (+ species-dependent skip) | [docs/linear.md](docs/linear.md) | (todo) |
| `interaction` | A-basis + symmetric contraction (body order ν) + message | [docs/interaction.md](docs/interaction.md) | (todo) |
| `gate` | gated equivariant nonlinearity (scalars gate higher ℓ) | [docs/gate.md](docs/gate.md) | (todo) |
| `readout` | scalar readout + graph pooling | [docs/readout.md](docs/readout.md) | (todo) |

Deferred (belongs to the CDVAE model, spec'd in that phase):

| Module | Purpose | Spec |
|---|---|---|
| `noise` | noise-level / timestep embedding (CDVAE decoder) | (deferred) |

> The precise architecture is defined per-layer in these spec files and assembled
> in `models/encoder/DESIGN.md`, not in `pipeline.md`.
