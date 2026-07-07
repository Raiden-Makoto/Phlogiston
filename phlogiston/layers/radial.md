# `radial` — Bessel radial basis + smooth cutoff + weight MLP

Turns each edge length into the (invariant) weights that parameterize the
interaction tensor product, smoothly vanishing at the cutoff.

## Contract
- **In**: `edge_len [E]` (Å).
- **Out**: `W [E, n_out]` invariant weights (`n_out` = consumer's
  `TensorProduct.weight_numel`).
- Equivariance: input is a scalar distance → output is invariant.

## Definition
1. **Bessel basis** (DimeNet), `n = 1..n_bessel`:
   ```
   b_n(d) = sqrt(2/r_max) · sin(n·π·d/r_max) / d
   ```
2. **Polynomial cutoff envelope** (order `p`), `x = d/r_max`, 0 for `d ≥ r_max`:
   ```
   u(d) = 1 − ((p+1)(p+2)/2)·x^p + p(p+2)·x^{p+1} − (p(p+1)/2)·x^{p+2}
   ```
   C¹-smooth, `u(r_max)=0`, `u'(r_max)=0`.
3. **MLP**: `W = u(d) · MLP( b(d) )`, `MLP: n_bessel → [64,64,64] → n_out`, SiLU.
   Multiplying by `u(d)` makes every edge weight (hence its message) decay
   smoothly to 0 at the cutoff — no discontinuity as neighbors enter/leave.

## Params / defaults
`n_bessel=8`, `p=6`, `r_max=6.0`, hidden `[64,64,64]`, SiLU. `n_out` set by
the interaction layer at construction.

## Tests
- `W → 0` continuously as `d → r_max`; exactly 0 for `d ≥ r_max`.
- Finite at small `d` (sin(x)/x limit).
- Shape `[E, n_out]`.
