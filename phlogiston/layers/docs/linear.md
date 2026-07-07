# `linear` — equivariant linear

Per-irrep learnable mixing across channels (no cross-ℓ leakage). Thin wrapper
over `e3nn.o3.Linear`; used for message/update projections and the
species-dependent skip connection.

## Contract
- **In**: `x [N, dim(irreps_in)]`.
- **Out**: `[N, dim(irreps_out)]`.
- Equivariance: `o3.Linear` mixes only same-`(ℓ,p)` channels → equivariant.

## Variants
- **Plain**: `o3.Linear(irreps_in, irreps_out)`.
- **Species-dependent skip** `W_skip(z)`: a separate weight set per element.
  Implement as `o3.Linear` with `weight` gathered per node from a
  `[n_species, weight_numel]` table indexed by `z` (MACE self-connection). This
  keeps equivariance (weights are scalars) while letting each element have its
  own residual mixing.

## Params / init
- `biases=False` for non-scalar irreps (bias only valid on `0e`).
- Default `e3nn` init; skip table init so the residual starts near-identity on
  the scalar channels.

## Tests
- Equivariance: rotate input → output transforms identically.
- Species skip: different `z` selects different weights; same `z` identical map.
