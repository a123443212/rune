# RUNE v0.12 Architecture: engine-native evaluation

v0.11 answered whether the evaluator can be fast. v0.12 answers whether the engine can actually use it.

```text
RUNE model
   ↓
compiled runtime (v0.11 artifact, frozen)
   ↓
engine adapter (C++ / Rust / Python, same contract)
   ↓
minimal alpha-beta integration layer (not a rewrite)
   ↓
real search-node distribution
   ↓
NPS, nodes, time-to-depth, root move, match strength
```

## What changed from v0.11

- New `core/engine/`: canonical scale, WDL helpers, mate guards, eval cache with model-hash identity, minimal negamax alpha-beta with per-node-type stats, lazy L0/L1/L2 switch. No change to `Evaluator`, kernels, compiler, or spec semantics.
- New `crates/rune-search/`: same contract, same cache rule, same lazy rule, shakmaty-backed alpha-beta for research and differential testing. Rust is now a research engine runtime, not just a benchmark.
- New `training/engine/`: contract, scale, cache, lazy, and reference alpha-beta used for orchestration and analysis. Python stays out of the hot path.
- Search respects the existing v0.4 adaptive machinery instead of inventing a parallel lazy system: L1 reuses threshold routing, L2 adds the alpha/beta margin check with bounded refinement and full-evaluation fallback.

## Roles after v0.12

```text
C++:  production engine evaluator, primary strength benchmark
Rust: research engine runtime, search-native validation, differential platform
Python: analysis, training, calibration, dataset work, experiment orchestration
```

## What v0.12 refuses

Claiming improvement from lower MSE, higher WDL accuracy, or higher standalone NPS alone. Anything that does not survive real search is reported as static improvement only.
