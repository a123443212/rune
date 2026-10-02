# RUNE v0.1 Implementation Report

## Implemented architecture

- Board/position: FEN parse/emit, pseudo-legal generation, make/unmake with
  full state restore, Kiwipete perft(1,2) = 48/2039 and startpos perft(3) = 8902.
- Features: `grouped_hkav2_fullthreats_v01`, 8 groups, vocabs
  256/256/256/128/128/512/512/64, sorted-unique deterministic output,
  exact C++/Python parity tested on 6 positions.
- Accumulator: per-group float tables, incremental add/remove diffs,
  incremental == refresh to 1e-5 over random make/unmake walks; int8 tables
  with per-group scales and int32 accumulation.
- Mixers: RUNE-SFNN (74276 mixer+head params), RUNE-MLP (37156),
  RUNE-ATTN (40388), RUNE-ATTN-GAB (+64 bias params). Attention is
  `A = clamp(QK^T + B, 0, 1)`, `Y = X + AV`, no softmax anywhere.
- Head: 256->128->32->tanh value plus 32->3 WDL logits.
- Loss: `L_value + l1*L_WDL + l2*L_rank`, margin hinge on sibling pairs,
  all coefficients config-driven.
- Dataset: JSONL pipeline with integrity checks, normalized-key dedup,
  quality filter, phase balancing, game-grouped splits, and
  random/stratified/disagreement samplers (disagreement always mixed).
- Export: `.rune` binary (magic + JSON header + LE payload), fp32 and int8
  embedding quantization, readable from Python and C++.
- Inference: persistent-accumulator `Evaluator`, fixed buffers, no hot-path
  allocation; pybind11 module exposes board, features, and full-model eval.

## Known limitations

- Vocab sizes are research placeholders, not full HalfKAv2 scale.
- Attention/head run fp32; only embeddings have an int8 path.
- Match tool is greedy 1-ply; no search integration yet.
- Mobility counts pseudo-legal moves including checking-leaving moves.
- Synthetic demo data is random playouts, unsuitable for real conclusions.

## Benchmark methodology

Matched conditions: same binary, same positions, microbenchmarks per stage
plus a 20k-node incremental walk for NPS. Current numbers (this machine):
feature extract ~5-8 us, accumulator refresh ~2-11 us, incremental update
~0.4-0.8 us, attention ~17-22 us, MLP head ~30 us, full eval ~50-65 us,
walk NPS ~25k nodes/s (includes full feature extraction per node).

## Current performance

- C++: all tests pass (perft, make/unmake FEN roundtrip, incremental
  accumulator, attention-vs-naive, serialization, quant tolerance, model I/O).
- Python: 25 pytest tests pass, including torch-vs-C++ eval agreement at
  1e-4 for all four architectures and export roundtrips.
- No strength comparison is claimed: match smoke test used random weights.

## Reproducibility information

- Configs in `configs/` pin arch, seeds, lambdas, sampling, milestones,
  teacher/feature versions.
- `Trainer.save_checkpoint` stores config, seed, positions seen, params,
  git commit (via `RUNE_GIT_COMMIT`), and timestamp.
- `LearningCurve` writes config copy + `learning_curve.csv` per run.
- Dataset splits are hash-of-game-id based, stable across runs.

## Next experimental priorities

1. Collect teacher-labeled data and run the 100M milestone for all four
   models with fixed loss/data.
2. Toggle ranking loss (H4) and disagreement sampling (H5) one axis at a time.
3. Add search integration for valid match testing before any strength claim.
4. Profile-guided SIMD (AVX2) only after correctness suite stays green.
