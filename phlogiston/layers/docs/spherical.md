# `spherical` — real spherical harmonics of edge directions

The equivariant angular signal carried on each edge.

## Contract
- **In**: `edge_vec [E,3]` (Å, need not be unit — normalized internally).
- **Out**: `Y [E, dim(Y_irreps)]` with `Y_irreps = Σ_{ℓ=0..L_sh} 1x(ℓ, (-1)^ℓ)`.
  For `L_sh=3`: `1x0e + 1x1o + 1x2e + 1x3o`.
- Equivariance: `Y` transforms by the Wigner-D of each `ℓ` (equivariant by
  construction) — this is what injects rotational structure into messages.

## Definition
```
Y = o3.spherical_harmonics(range(L_sh+1), edge_vec, normalize=True,
                           normalization="component")
```
- `normalize=True`: uses the unit direction `r̂`; the length is handled by
  `radial` — this cleanly separates angular (SH) from radial (Bessel).
- `component` normalization matches the `TensorProduct` normalization in
  `interaction`.

## Params
`L_sh` (default 3).

## Tests
- Rotate `edge_vec` by `R`: `Y` transforms by `D^ℓ(R)` per block (ℓ=0 invariant,
  ℓ=1 rotates as a vector, …). Numerically check against `e3nn` Wigner-D.
- `ℓ=1` block ∝ the unit direction `r̂`.
