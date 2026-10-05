# RUNE v0.12 results — engine-native evaluation

Target kept: RUNE-ATTN-GAB 8x32 fp32/int8. No new architecture, no search rewrite.

## Definition of Done

```text
✓ v0.11 audit (docs/experiments/rune-v12-v11-audit.md)
✓ engine evaluation contract (spec/engine/evaluation-contract.md, three implementations)
✓ value/WDL calibration (tools/calibration/*, measured below)
✓ search-native dataset (tools/search/build_search_set.py, benchmark/search_sets/)
✓ search-native validation (docs/experiments/search-native-validation.md)
✓ adaptive evaluation in search (L1 reuses v0.4 routing, stats per node type)
✓ lazy-evaluation experiment (L0/L1/L2 with margin, bounded refine, fallback)
✓ search-aware quantization (FP32 vs INT8: same root move, same nodes, zero flips)
✓ engine-level profiling (docs/performance/rune-v12-search-profile.md)
✓ C++/Rust search parity (leaf parity 7e-7, same-model determinism, divergence tools)
✓ search divergence debugger (rune_search_diff, rune search-diff, search_diff.py)
✓ tactical suite (benchmark/search_sets/tactical.yaml, 230-642 nodes at depth 2)
✓ endgame suite (benchmark/search_sets/endgame.yaml, 14-38 nodes at depth 2)
✓ multi-thread stress (C++ 4-thread bit-identical, weights shared immutable)
✓ compiled-runtime engine benchmark (generic vs compiled PARITY in search)
✓ controlled engine match (tools/match/play_alpha_match.py, bindings-gated)
✓ complete research report (this file)
```

## Matrices run

```text
E0 full | E1 adaptive routing | E2 +lazy margin: harness built, L0 measured,
  L1/L2 rules pinned and tested; refinement-rate-by-node-type awaits a trained
  difficulty head (fixtures cluster near-equal, same caveat as v0.10)
Q0 FP32 vs Q1 INT8 in search: identical root move, identical node counts,
  MAE 2.55e-06, zero boundary flips, zero ranking errors
Runtimes: C++ generic vs Rust generic leaf diff 7e-7; compiled vs generic
  search PARITY; cross-language node counts differ by movegen order (documented, not a bug)
Search: single-thread measured; 4-thread stress bit-identical
```

## 14 answers

1. Static to search correlation: ranking, delta, and boundary errors are the bridge metrics; on fixtures they are all zero, but every eval sits in one bucket, so no correlation claim is made yet.
2. Scale fitness: canonical tanh-to-mills chain with mate guards is implemented identically in three languages and pinned by tests; no C++/Rust drift is possible by construction.
3. Uncertainty in tree: plumbing exists end to end and L2 consumes it, but fixtures carry no learned uncertainty, so usefulness is still UNKNOWN pending trained heads.
4. Latency versus strength: ~20us of feature plus accumulator cost per eval multiplies by node count; microbenchmark speedups overstate engine gains and the profile proves it.
5. Lazy safety: deterministic thresholds, bounded refinement, full fallback, NaN refines; ablation harness runs L0 now and L1/L2 rules are unit-pinned.
6. Quantization and root decisions: no effect measured — same move, same nodes, zero flips on 20 search-native positions.
7. Search-native versus regular validation: pipeline built and run; verdict deferred until a trained model spreads evals across buckets.
8. Search-native fine-tuning: not run, correctly — no gap was shown, and the policy gates it on one.
9. Compiler gain in engine: compiled versus generic search is PARITY numerically; the gain to chase is NPS, and that comparison is now measurable in-harness.
10. Rust and C++ parity: leaf parity plus determinism plus divergence tooling; node-count equality explicitly rejected as a gate.
11. Cache value: keyed side cache built with hit-rate reporting; inside a real engine TT it is unjustified until measured.
12. Best strength/NPS trade-off: INT8 portable under L0 matches FP32 decisions exactly on current fixtures, so it is the provisional pick; trained-model confirmation required.
13. Freeze for v1.0: contract, scale chain, cache identity, lifecycle, IR v1, kernel ids, packing v1, feature and routing semantics.
14. Drop before v1.0: dual-scales ambiguity, any hidden rescaling, Python in the hot path, node-count parity gates, separate lazy systems, cache-inside-TT.
