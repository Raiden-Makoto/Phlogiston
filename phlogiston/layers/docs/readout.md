# `readout` — per-layer scalar readout + graph pooling

Extracts invariant per-atom scalars from node features and pools them to the
graph level. Heads use this; the encoder exposes per-layer readouts for the
MACE-style summed-energy path.

## Contract
- **In**: node features `h [N, irreps_H]`, `batch [N]`.
- **Out**: per-atom `r [N, n_out]` (invariant) and/or pooled `g [B, n_out]`.
- Equivariance: reads only the `0e` (scalar) part of `h` → invariant output.

## Definition
```
s   = scalar_part(h)                 # take the 0e channels -> [N, mul]
r   = MLP(s)                         # intermediate layers: Linear; final: MLP + SiLU
g   = scatter(r, batch, reduce=R)    # R = "sum" (extensive, e.g. energy)
                                     #     "mean" (intensive, e.g. moduli/density)
```
- Only scalars are read out (properties/energy are invariant); vectors/rank-2
  stay internal to the network. Reduce mode is chosen by the consuming head.

## Params
- `n_out` (1 for a single scalar target; heads may request more).
- `reduce` ∈ {sum, mean}; default per target (energy=sum, intensive=mean).

## Usage
- **Stability (MACE energy)**: sum per-layer per-atom readouts `Σ_t r^{(t)}_i`,
  then `scatter-sum` over `batch`.
- **Property heads**: pool final-layer scalars with the appropriate reduce.

## Tests
- Invariance: rotate input → identical `r`, `g`.
- `sum` pooling is size-extensive; `mean` is size-intensive.
