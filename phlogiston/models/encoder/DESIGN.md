# Encoder — DESIGN

The shared **E(3)-equivariant** crystal-graph encoder. It is a **MACE-style
higher-order equivariant message-passing network** (Batatia et al., 2022): it
builds a per-node *atomic basis* from neighbors, raises it to higher body order
via a symmetric tensor-product (ACE) contraction, and updates node features over
a few layers. Consumed by `predictor` and `cdvae` (same architecture, separate
weights).

We implement the network ourselves; only low-level equivariant math is reused
from `e3nn` (`o3.Irreps`, `o3.spherical_harmonics`, `o3.TensorProduct`,
Wigner-3j / Clebsch–Gordan, reduced tensor products). No model is imported.

---

## 0. Notation & equivariance

- Features live in `e3nn` irreps `O(3)` reps; a feature block is
  `mul × (ℓ, p)` (`0e` scalars, `1o` vectors, `2e` rank-2, …).
- Under `g ∈ O(3)`, an `(ℓ,p)` feature transforms by its Wigner-D `D^ℓ(g)`
  (scalars invariant, vectors rotate). Every op below is equivariant by
  construction (CG tensor products + spherical harmonics of edge directions).
- Periodicity is already baked into `edge_vec` (nearest-image displacements from
  the featurizer, cutoff 6.0 Å).

## 1. Inputs / outputs

**In** (a `BatchedGraph`): `z [N]`, `edge_index [2,E]` (row0 = center `i`,
row1 = neighbor `j`), `edge_vec [E,3]`, `edge_len [E]`, `batch [N]`.

**Out**: final per-atom features `h^{(T)} [N, dim(H)]` (irreps `H`), and the
per-layer invariant readouts `{r^{(t)}_i}` (0e) that heads can sum (MACE-style
energy readout). Pooling to graph level is done by the consuming head.

## 2. Hyperparameters (defaults resolved)

| Symbol | Meaning | Default |
|---|---|---|
| `r_max` | radial cutoff (matches featurizer) | **6.0 Å** |
| `n_bessel` | Bessel radial basis functions | 8 |
| `p_cutoff` | polynomial envelope order | 6 |
| `L_sh` | max ℓ of edge spherical harmonics | **3** |
| `L_feat` | max ℓ of hidden node features | **2** |
| `mul` | channels (multiplicity) per irrep | **128** |
| `H` | hidden node irreps | `128x0e + 128x1o + 128x2e` |
| `ν` (nu) | correlation / body order of product basis | **3** (v1 starts at 2) |
| `T` | number of interaction layers | **2** |
| `radial_mlp` | radial weight MLP | `[64,64,64]`, SiLU |

Rationale: higher body order (`ν=3` → up to 4-body messages) lets MACE match
deeper nets with only `T=2` layers; `L_feat=2` keeps rank-2 (needed for
tensorial elastic response) at modest cost; `L_sh=3` enriches the angular basis.

## 2.5 Assembly skeleton (forward pass)

Concrete structure with `T=2`, `mul=128`, `L_feat=2`, `L_sh=3`, `ν=3`.

```
inputs: z[N], edge_index[2,E], edge_vec[E,3], edge_len[E], batch[N]

Y   = Spherical(edge_vec)                 # [E, 1x0e+1x1o+1x2e+1x3o]   (computed ONCE, shared)
h   = Embedding(z)                        # [N, 128x0e]                (scalars only at input)
readouts = []

# ── Interaction block 0 (full irreps) ──────────────────────────────
w0  = Radial_0(edge_len)                  # [E, TP0.weight_numel]
A   = (1/√N̄)·scatter_i TP0(h[j], Y; w0)  # A-basis
B   = SymContract_0(A; ν=3)               # 1-,2-,3-,4-body terms
h   = Linear_msg_0(B) + Skip_0(z, h)      # -> pre-gate irreps
h   = Gate_0(h)                           # [N, 128x0e+128x1o+128x2e]  = H
readouts += Readout_0(scalar(h))          # [N,1]

# ── Interaction block 1 (LAST: scalar output, no gate) ─────────────
w1  = Radial_1(edge_len)
A   = (1/√N̄)·scatter_i TP1(h[j], Y; w1)
B   = SymContract_1(A; ν=3)
h   = Linear_msg_1(B) + Skip_1(z, h)      # -> [N, 128x0e]  (scalars only)
readouts += Readout_1(scalar(h))          # [N,1]

E_atom = Σ_t readouts[t]                   # [N]  (MACE energy path; heads pool)
return  h (final node scalars), per-layer readouts, E_atom
```

### Layer inventory
| Component | Count | Notes |
|---|---|---|
| `Spherical` | 1 (shared) | computed once from `edge_vec` |
| `Embedding` | 1 | `z → 128x0e` |
| Interaction block | `T = 2` | each owns its own `Radial`, `TP`, `SymContract`, `Linear_msg`, `Skip` |
| `Radial` MLP | 2 | **not** shared — one per interaction |
| `Gate` | `T−1 = 1` | on every layer **except the last** |
| `Readout` | `T = 2` | one per layer; summed for the energy path |

### Irreps at each stage
| Stage | irreps |
|---|---|
| `h⁰` (after embedding) | `128x0e` |
| block 0, pre-gate | `384x0e + 128x1o + 128x2e` (128 pass-through + 256 gate scalars + gated) |
| block 0, post-gate = `H` | `128x0e + 128x1o + 128x2e` |
| block 1 (last), output | `128x0e` (scalars only — readout-only, cheaper) |
| per-layer readout | `[N, 1]` |

Notes: higher-ℓ features (`1o`, `2e`) are **absent at input** and first appear
after block 0 (the TP with `Y` creates them); the **last** block collapses back
to scalars because readouts/heads only consume invariants. `ν=1` in v1 skips the
`SymContract` beyond the pairwise term.

## 3. Data flow (one interaction layer `t`)

### 3.1 Node embedding (t = 0) — `layers/embedding`
`h^{(0)}_i = W_embed · onehot(z_i)` → `mul × 0e` scalars (optionally concatenated
with fixed element descriptors). Only scalars at input; higher ℓ grow via
interactions.

### 3.2 Edge basis — `layers/radial`, `layers/spherical`
- Radial: `R(r_ij) = MLP( bessel_n(r_ij) · f_cut(r_ij) )`, with
  `f_cut` the smooth polynomial envelope (→0 at `r_max`, C¹). Produces the
  per-path tensor-product weights.
- Angular: `Y_{ℓ}^{m}(r̂_ij)` for `ℓ = 0..L_sh` (real SH, `component` norm).

### 3.3 Atomic basis A (2-body, pooled) — `layers/interaction`
Equivariant, weight-parameterized tensor product of neighbor features with the
edge SH, summed over neighbors:

```
A^{(t)}_{i} = (1/√N̄) · Σ_{j∈N(i)}  TP( Y(r̂_ij), h^{(t)}_j ;  W = R(r_ij) )
```

- `TP` = CG tensor product (`o3.TensorProduct`) coupling `Y` (`ℓ≤L_sh`) with the
  neighbor feature irreps into paths up to `L_feat`; path weights come from the
  radial MLP `R(r_ij)`.
- `N̄` = dataset average neighbor count (message normalization; computed once).
- Aggregation over `j` = native `torch` scatter on `edge_index[0]`.

### 3.4 Product basis B (higher body order) — `layers/interaction`
Raise `A_i` to correlation order `ν` via a **symmetric contraction** with
generalized Clebsch–Gordan coefficients (the ACE step):

```
B^{(t),(η)}_{i} = Σ  𝒞^{η}_{lm...}  ∏_{s=1..ν}  A^{(t)}_{i}
```

i.e. equivariant symmetric products of `A_i` with itself up to `ν` factors,
contracted to output irreps. `ν=1` recovers the 2-body message; `ν=2,3` add 3-
and 4-body terms. Coefficients `𝒞` are precomputed from Wigner-3j / reduced
tensor products (`e3nn.o3.ReducedTensorProducts`), so this is a fixed linear map
over the tensor powers — implemented by us, math from `e3nn`.

### 3.5 Message + update — `layers/linear`, `layers/gate`
```
m^{(t)}_i = W_msg · B^{(t)}_i                       # equivariant linear
h^{(t+1)}_i = Gate( W_update · m^{(t)}_i + W_skip(z_i) · h^{(t)}_i )
```
- `W_skip(z_i)`: species-dependent self-connection (residual).
- `Gate`: gated nonlinearity — scalars pass through SiLU and *gate* (multiply)
  the higher-ℓ channels, preserving equivariance.

### 3.6 Per-layer readout — `layers/readout`
`r^{(t)}_i = MLP_t( scalar-part(h^{(t)}_i) )` (invariant `0e`). The stability
head sums `Σ_t r^{(t)}_i` per atom then over the graph (MACE energy readout);
property heads instead pool the final features (their choice, in `predictor`).

## 4. Normalization & initialization
- Message normalization by `1/√N̄` (avg neighbors), MACE-style.
- `e3nn` `component` normalization for tensor products & SH.
- Radial MLP init small; skip connection init near-identity on scalars.
- Optional per-atom energy shift/scale (mean/std of the target) applied in heads.

## 5. Compute / hardware
- Cost is dominated by the tensor products (§3.3) and symmetric contraction
  (§3.4). `L_sh=3, L_feat=2, ν=3, T=2, mul=128` is the target; knobs let us trade
  accuracy for speed. All ops are `e3nn`/torch → run on ROCm/MI350X.
- Training parallelism: **TP2 (max TP4)** (see `pipeline.md` §5).

## 6. Module mapping (`phlogiston/layers`)
`embedding` · `radial` · `spherical` · `interaction` (A-basis + symmetric
contraction + message) · `linear` · `gate` · `readout`. The encoder assembles
these; each block has its own `<layer>.md` spec.

## 7. Build plan (incremental, each validated for equivariance)
1. **v1 (ν=2)**: A-basis + pairwise tensor products only (NequIP-equivalent),
   `T=3`. Validates the equivariant MP machinery end-to-end.
2. **v2 (ν=3)**: add the symmetric contraction (full MACE body order), `T=2`.
3. Equivariance unit tests at every step: rotate input → outputs transform by the
   correct Wigner-D (scalars invariant, vectors/tensors rotate).

## 8. Resolved vs open
- **Resolved**: `L_sh=3`, `L_feat=2`, `mul=128`, `T=2`, `ν=3` (v1 at 2),
  cutoff 6.0 Å, Bessel(8)+poly(6) radial, gated nonlinearity, MACE readout.
- **Open**: final `mul` (128 vs 256) and `ν` after speed/accuracy profiling on
  the box; whether to add element-descriptor seeding to the node embedding.

## References
MACE (Batatia et al., 2022); ACE (Drautz, 2019); NequIP (Batzner et al., 2022);
`e3nn` (Geiger & Smidt).
