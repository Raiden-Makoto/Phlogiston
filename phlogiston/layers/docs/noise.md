# `noise` — noise-level / timestep embedding

Conditions the CDVAE score decoder on the diffusion noise level. A sinusoidal
(Fourier) embedding of a scalar `sigma` (or timestep) into invariant scalars.

## Contract
- **In**: `sigma [B]` (noise level, or a per-graph timestep).
- **Out**: `[B, dim]` invariant features (`dim x 0e`), to be injected into node
  scalars (broadcast per graph).
- Equivariance: a function of a scalar → invariant.

## Definition
```
freqs_k = exp(-log(max_period)·k/half),  k = 0..half-1   (half = dim/2)
emb(sigma) = [cos(sigma·freqs), sin(sigma·freqs)]         # [B, dim]
```
Same construction as transformer positional / diffusion time embeddings; smooth
and bounded in [-1, 1].

## Params / defaults
`dim=64` (even), `max_period=10000`.

## Tests
- Shape `[B, dim]`; different `sigma` → different embedding; deterministic;
  finite and bounded by 1.
