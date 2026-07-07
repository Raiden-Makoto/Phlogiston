# Encoder — DESIGN

Shared E(3)-equivariant crystal-graph encoder. Consumed by `predictor` and
`cdvae` (same architecture, separate weights).

> Status: **skeleton** — architecture details to be filled in the next step
> (this is where the ambiguous "MACE-style E(3)" gets pinned down precisely:
> body order, ℓ_max, irreps per layer, tensor-product paths, normalization).

## Purpose
Map a `BatchedGraph` (z, edge_index, edge_vec, edge_len) to per-atom equivariant
features (and pooled graph features) that downstream heads read.

## Inputs / Outputs
- In: `z [N]`, `edge_index [2,E]`, `edge_vec [E,3]`, `edge_len [E]`, `batch [N]`.
- Out: per-atom features (irreps TBD), pooled graph features.

## Architecture (TBD)
- Node embedding — layers/`embedding`
- Edge radial (Bessel + cutoff) — layers/`radial`
- Edge angular (spherical harmonics, ℓ ≤ L_max) — layers/`spherical`
- N × interaction blocks (CG tensor-product message passing) — layers/`interaction`
- Gated nonlinearity — layers/`gate`
- (readout handled by the consuming model)

## Hyperparameters (to fix)
`L_max`, hidden irreps / multiplicities, N_layers, radial basis size, cutoff
(6.0 Å, matching the featurizer), body order.

## Open decisions (from pipeline.md §11)
- `L_max` (2 vs 3); hidden irreps; N_layers.

## References
MACE, NequIP, e3nn.
