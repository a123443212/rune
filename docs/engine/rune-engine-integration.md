# RUNE engine integration

Minimal alpha-beta integration layer, not a search rewrite. No MCTS, no neural policy, no neural move ordering, no RL search.

## Interface

C++ works on `rune::Board` with `loadRuneFile`, `Evaluator::refresh`, `updateIncremental`, `evaluate`, `evaluateBoard`. Rust works on FEN through `Evaluator::evaluate_board` behind `rune-search::AlphaBeta`. Python orchestrates through `training/engine/search.py`. All three share one contract: `load_model`, `evaluate`, `refresh_state`, `make_move`, `unmake_move`, or the closest equivalent the existing runtime already has.

## Search loop

Negamax with alpha-beta, move loop with cutoff counting, per-node-type stats (`root`, `pv`, `cut`, `leaf`), time, NPS, root score, and root move. Node types exist so approximation or adaptive paths can be caught misbehaving in one specific context instead of hiding in an average.

## Lazy evaluation

```text
L0  full evaluation always
L1  existing adaptive threshold routing (v0.4 machinery)
L2  search-aware margin check: skip refinement outside [alpha - margin, beta + margin],
    refine inside, always bounded by max_refine with full-evaluation fallback
```

Thresholds are deterministic, errors are bounded, recursion is bounded, and NaN always refines. L2 never invents unbounded recursive evaluation.

## Caching

Evaluator cache is separate from engine transposition tables and only used when the engine has no evaluation cache of its own. Keys bind board, side to move, castling, en passant, model hash, and mode. Cross-model reuse fails closed.

## Threads

Weights shared immutable, evaluator state thread-local. The C++ stress test runs four threads on one shared pattern and requires bit-identical results. No global mutable state, no corrupted accumulators.

## Compiler in the loop

Generic versus compiled runtimes are benchmarked inside the engine, not beside it. A compiler gain that disappears under full search is reported as exactly that.
