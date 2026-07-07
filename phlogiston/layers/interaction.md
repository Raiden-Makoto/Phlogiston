# `interaction` — MACE interaction block (A-basis + symmetric contraction)

The core equivariant message-passing block: pool a per-atom **atomic basis**
`A_i` from neighbors, raise it to body order `ν` via a **symmetric contraction**,
and produce the node update. This is the crux of the "MACE-style" architecture.

## Contract
- **In**: node features `h [N, irreps_H]`; `edge_index [2,E]`;
  edge SH `Y [E, Y_irreps]` (from `spherical`); radial weights source (from
  `radial`); `z [N]` (for the skip). `N̄` = dataset avg neighbor count.
- **Out**: updated node features `h' [N, irreps_H]` (same irreps → residual/stacking).
- Equivariance: every step is a CG tensor product / equivariant linear → exact.

## Steps

### 1. Atomic basis A (2-body, pooled)
Per edge `e=(i←j)`:
```
msg_e = TP( h[j] , Y_e ;  weight = R(edge_len_e) )
A_i   = (1/√N̄) · Σ_{e: center=i} msg_e            # scatter-sum on edge_index[0]
```
- `TP = o3.TensorProduct(irreps_H, Y_irreps, A_irreps, instructions,
  shared_weights=False, internal_weights=False)`; `R` (radial MLP) supplies the
  per-edge path weights (`weight_numel = TP.weight_numel`).
- `A_irreps`: `mul ×` irreps up to `L_feat` (the paths of `irreps_H ⊗ Y_irreps`
  truncated to `ℓ ≤ L_feat`, `mul` preserved via `uvu` instructions).
- Aggregation via native `torch` scatter (no pyg-lib).

### 2. Symmetric contraction → product basis B (body order ν)
Raise `A_i` to correlation order `ν` (per-node, no new neighbors):
```
B_i = Σ_{k=1..ν} SymContract_k( A_i )     # sum of 1-, 2-, …, ν-fold sym. products
```
- `SymContract_k` maps the `k`-fold **symmetric** tensor power of `A_i` to
  `irreps_H`, using generalized Clebsch–Gordan coefficients precomputed at init
  from Wigner-3j / `e3nn.o3.ReducedTensorProducts`. It's a fixed multilinear map;
  only the per-path channel weights are learnable.
- `ν=1` → the 2-body message; `ν=2` adds 3-body; `ν=3` adds 4-body (default).
- **v1 milestone**: implement `ν≤2` (pairwise) first; add `ν=3` in v2.

### 3. Message + update
```
m_i  = Linear_msg(B_i)                       # -> message irreps (= irreps_H)
h'_i = m_i + W_skip(z_i) · h_i               # species-dependent residual (linear.md)
# nonlinearity applied by the caller via `gate` (gate.md)
```

## Params / defaults
`irreps_H = 128x0e+128x1o+128x2e`, `L_feat=2`, `Y` up to `L_sh=3`, `ν=3`,
`N̄` from data. Tensor products use `component` normalization.

## Complexity
Dominated by the `TP` (§1) over edges and `SymContract` (§2) over nodes; both
are `e3nn`/torch and ROCm-safe. Cost scales with `mul²`, number of TP paths, and
`ν`.

## Tests (must pass before trusting)
- **Equivariance**: rotate all inputs by `R` → `A_i`, `B_i`, `h'_i` transform by
  the correct Wigner-D per irrep (scalars invariant, vectors/rank-2 rotate).
- **Permutation**: reindexing atoms permutes outputs consistently.
- **Neighbor-normalization**: mean message magnitude ~O(1) independent of
  coordination (via `1/√N̄`).
- `ν=1` reduces to a plain equivariant MPNN message (sanity vs a hand check).
