# `vector_readout` — equivariant vector readout

Reads node features to a per-atom **equivariant vector** (`1o`). Used for the
CDVAE coord score (∇ log p over positions), which must rotate with the
structure — unlike `readout` (`ScalarReadout`), whose output is invariant.

## Contract
- **In**: node features `[N, dim(irreps_in)]`.
- **Out**: `[N, 3·n_vectors]` — `n_vectors x 1o` per atom.
- Equivariance: output transforms as `1o` (rotates as a vector under O(3)); a
  single `o3.Linear(irreps_in, "n_vectors x 1o")`.

## Definition
```
out = o3.Linear(irreps_in, f"{n_vectors}x1o")(x)
```
Only the `ℓ=1` paths of `irreps_in` feed the output (CG selection), so
`irreps_in` must contain `1o` (or higher-ℓ that couples to it) channels.

## Params
`n_vectors` (default 1 → a single `[N,3]` vector, e.g. the coord score).

## Tests
- Equivariance: rotate input → output rotates by the same `R` (`1o`).
- Shape `[N, 3·n_vectors]`.
