# RUNE v0.1 Architecture

## Motivation

NNUE dominates CPU chess evaluation because it combines sparse binary features,
incremental accumulator updates, and small SIMD-friendly affine layers. RUNE v0.1
keeps all three properties and asks a controlled question: does a lightweight
relational mixer over semantically grouped tokens improve strength per inference
cost compared to an equivalent MLP mixer?

v0.1 deliberately tests one idea only: non-softmax attention over 8 semantic
tokens with a learned 8x8 geometric bias. Everything else is infrastructure.

## Pipeline

```
Board
 -> Sparse feature extraction (GroupedHKAv2FullThreats, versioned)
 -> Incremental grouped accumulator (8 groups x 32 dims)
 -> ClippedReLU tokens T0..T7 in R^32
 -> Mixer: identity (MLP baseline) | linear attention | attention + GAB
 -> Head: 256 -> 128 -> 32 -> value, plus 32 -> 3 WDL branch
```

## Feature representation

Feature set `grouped_hkav2_fullthreats_v01`, implemented once in
`core/features/feature_set.*` and mirrored exactly in
`training/features/python_features.py`:

| Group | Name | Vocab | Content |
|-------|------|-------|---------|
| G0 | pawn_structure | 256 | pawn (color, square) + file x rank-bucket structure terms |
| G1 | king_zone | 256 | king (color, square) + king-neighborhood terms |
| G2 | minor_pieces | 256 | knight and bishop (color, square) |
| G3 | rooks | 128 | rook (color, square) |
| G4 | queens | 128 | queen (color, square) |
| G5 | threats | 512 | victim-type x victim-square + attacker coarse square, for every real attacker-victim pair on the board |
| G6 | mobility | 512 | piece-type x square + side-to-move pseudo-move-count bucket |
| G7 | global | 64 | side to move, castling mask, piece-count bucket, game phase |

Extraction is deterministic: output is sorted and deduplicated. The feature
layer knows nothing about architectures; the architecture layer receives only
the 8x32 token matrix.

## Grouped accumulator

Each group owns an embedding table `[vocab_g][32]`. The accumulator state is
8x32 floats (training) or 8x32 int32 over int8 tables (inference). Updates are
incremental: given sorted feature lists before/after a move, `diffFeatures`
produces added/removed sets and only affected groups are touched. Full refresh
and incremental update agree to 1e-5 in tests over random move sequences
including unmake.

Tokens are `clip(accumulator, 0, 1)` per element. Token count and dimension are
config fields (`tokens`, `token_dim`), not hard-coded in the Python models;
the C++ core uses compile-time constants with a documented path to runtime
shapes.

## RUNE attention block

With `X in R^(8x32)`:

```
Q = Linear32(X), K = Linear32(X), V = Linear32(X)
S = Q K^T (+ B if GAB enabled)
A = clamp(S, 0, 1)
Y = A V
Y_final = X + Y
```

There is no softmax and no transcendental function in the block. All
projections are standard `y = Wx + b` row-linears, identical in PyTorch
(`F.linear`) and C++. `B in R^(8x8)` is the learned geometric bias (GAB):
a static relational prior over token pairs, ablatable by zeroing.

Variants: `RUNE-ATTN` (B fixed at zero) vs `RUNE-ATTN-GAB` (B learned).

## Output head

`flatten(8x32) -> Linear(256,128) -> clip -> Linear(128,32) -> clip ->
Linear(32,1) -> tanh` for value in [-1, 1] (white perspective), plus
`Linear(32,3)` WDL logits. The SFNN reference baseline (`RUNE-SFNN`) widens
the first layer to 256 and shares the feature layer in v0.1; a fully separate
HalfKA feature set is listed as future work.

Parameter counts (mixer+head, excluding embeddings): RUNE-MLP 37156,
RUNE-ATTN(+GAB block) 40388 + 64 with GAB, RUNE-SFNN 74276.

## Loss

`L = L_value + l1 * L_WDL + l2 * L_rank` with all lambdas in config.
`L_value` is MSE on the tanh output, `L_WDL` cross-entropy, `L_rank` a margin
hinge over sibling value pairs from the same mini-batch group. Ranking never
replaces absolute evaluation; it is an auxiliary term toggled per experiment.

## Quantization

Per-tensor symmetric int8 for the 8 embedding tables: `scale = max|x| / 127`,
`q = clamp(round(x / scale))`. The accumulator sums int8 rows in int32, then
dequantizes with the per-group scale before token clipping. Attention and head
stay fp32 in v0.1 with a fake-quant path in Python; full integer attention is
future work. Tolerance: embedding roundtrip max error < 0.002, int8 vs fp32
token error < 0.05, exported-model C++ vs Python eval agreement 1e-4 (fp32).

## Inference pipeline

`Evaluator` binds embedding tables and an `IArchitecture`, holds a persistent
`GroupedAccumulator`, and exposes `refresh(board)`,
`updateIncremental(added, removed)`, `evaluate()`. Hot paths use caller-owned
fixed-size buffers; no allocation inside forward. The `.rune` file format is
`RUNE` magic + u32 header length + JSON header + raw little-endian payload,
readable from both Python and C++ (`core/model_io`).

## Limitations

- Vocab sizes are small research placeholders, not a full-size HalfKAv2.
- No search integration yet; match tool uses greedy 1-ply play.
- Mobility counts pseudo-legal moves, including moves that leave check.
- Python training embeds via padded per-group index batches, not incremental.
- Full int8 attention/matmul not yet implemented.
- No phase conditioning, no dynamic geometry, no transformer depth.
