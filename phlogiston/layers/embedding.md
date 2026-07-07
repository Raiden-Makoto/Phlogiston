# `embedding` — atomic-number node embedding

Initial node features from species (the only input feature; higher-ℓ features
grow through interactions).

## Contract
- **In**: `z [N]` int64 atomic numbers.
- **Out**: `h0 [N, mul]` interpreted as irreps `mul x 0e` (scalars).
- Equivariance: scalars are invariant → trivially equivariant.

## Definition
```
h0 = Embed(z)                         # nn.Embedding(Z_MAX+1, mul)
# optional (config): concat fixed element descriptors, then Linear -> mul
h0 = Linear([Embed(z) ; D(z)])        # D(z): electronegativity, radius, group, period, mass, ...
```
- `Z_MAX = 118` (index by atomic number directly; unused rows harmless).
- Descriptor seeding is **off by default** (open decision in encoder DESIGN §8);
  when on, descriptors are z-scored constants (not learned) then mixed by Linear.

## Params / init
- `mul` (default 128). `nn.Embedding` init `N(0, 1)`; Linear default.

## Tests
- Output shape `[N, mul]`; permutation of atoms permutes rows identically.
- Invariance: identical `z` → identical rows regardless of position.
