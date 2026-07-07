# `gate` — gated equivariant nonlinearity

The nonlinearity between/after interaction layers. Scalars are activated
directly; higher-ℓ irreps are multiplied ("gated") by activated invariant
scalars, which preserves equivariance (scaling a vector by an invariant keeps it
a vector).

## Contract
- **In**: features with irreps `scalars (0e) + gated (ℓ>0)`.
- **Out**: irreps `scalars + gated` (same ℓ>0 structure, now nonlinear).
- Equivariance: gating factors are invariant scalars → preserved.

## Definition
`e3nn.nn.Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)`:
- `irreps_scalars`: the `0e` channels passed through `act_scalars` (SiLU).
- `irreps_gated`: the `ℓ>0` channels to be gated.
- `irreps_gates`: **one extra `0e` scalar per gated irrep group**, passed through
  `act_gates` (sigmoid), then multiplied into the corresponding gated block.
- So the layer *before* `gate` must produce
  `irreps_scalars + irreps_gates + irreps_gated` (the interaction/linear output
  is sized accordingly).

## Params
`act_scalars = SiLU`, `act_gates = sigmoid`. Gate scalar count = number of
non-scalar irrep groups in the hidden irreps.

## Tests
- Equivariance: rotate input → output transforms identically (gates invariant).
- ℓ=0 path matches plain SiLU; zeroing a gate zeroes its vector block.
