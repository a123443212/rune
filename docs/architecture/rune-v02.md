# RUNE v0.2 Architecture

## Starting point

v0.1 audit (`docs/experiments/v01-audit.md`) found: the attention core is
cheap (~2.5 us), QKV projections and the 256->128 head dominate eval cost,
GAB is negligible, and grouping/dim/bias were never ablated. v0.2 therefore
upgrades the mixer and the bias, and ablates tokens, while keeping the
feature layer, accumulator discipline, and file format compatible.

## Gated relational mixer

Input: T tokens x D dims (T in {6, 8, 10}, D in {24, 32, 40}).

```
Q = Linear(X), K = Linear(X), V = Linear(X)      (DxD row-linears)
S = Q K^T + B
G = gate(S)                                       (registry: clip, hard_sigmoid)
Y = G V
out = X + alpha * Y
```

No softmax. No normalization layers. `alpha` is a config float (default 1.0).
Gate choice is a config string shared by C++ (`GateFn`) and Python
(`GATE_FNS`); adding a gate means adding it in both with a parity test.

## Structural bias

- B1 static: learned TxT matrix, v0.1-compatible semantics.
- B2 dynamic: `B = B_static + clip(U c outer W c, -0.25, 0.25)` where
  `c` is the 8-dim context vector and `U, W` are Tx8 matrices (128 extra
  params at T=8). The 0.25 cap keeps the correction small by construction.

Microbench verdict so far: static bias ~1 us; dynamic path adds measurable
but modest overhead (noisy machine: needs a quiet re-run before any claim).
If 100M screening shows NPS regression without metric gain, B2 is dropped.

## Context vector

8 floats from the board, no network: side to move, game phase/2, pawn count,
minor count, rook count, queen count, own king shield pieces, total pieces
(each count normalized by a fixed constant). Computed by `computeContext`
(C++) and `context_vector` (Python), tested for exact parity. Used only to
modulate the mixer, never as a parallel network.

## Token layouts

Embeddings stay per feature group (8 tables, width D). A layout maps each
token to a list of (group, lo, hi) ranges:

- 8-token: identity.
- 6-token compact: rooks+queens share a token, threats+mobility share one.
- 10-token expanded: knights/bishops split, victim/attacker threat terms split.

Quantization granularity follows tokens: member groups of a token share one
scale, so merged-token accumulators stay exact.

## Head and budgets

Head is unchanged (T*D -> 128 -> 32 -> tanh value + WDL) so mixer
comparisons stay clean; only the first layer shape varies with T*D.
Parameter deltas are reported per variant (e.g. dynamic bias +128 at 8x32,
6-token and 24-dim strictly smaller than 8x32).

## Quantization

Embedding tables export as fp32/int16/int8 (symmetric per-token scale).
Accumulators sum in int32 (int8/int16 tables) then dequantize; mixer and
head stay fp32. Mixed precision across stages is allowed only with a
benchmark attached.

## File format

`.rune` format 1 with additive fields: `gate`, `alpha`, `context_dim`.
Old files load unchanged; `RUNE-REL-02` files require the new fields plus
`tokens`/`token_dim` (already present). Arch construction is centralized in
`core/model_io/model_factory`.
