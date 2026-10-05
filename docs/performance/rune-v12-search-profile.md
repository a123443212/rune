# RUNE v0.12 search profile

Method: same model, same hardware, ranges over repeats. Evaluator numbers are shared metrics; engine numbers are the verdict.

## Engine bench (C++ mobility stub shown for structure, real evaluator below)

Depth scaling on startpos with the stub evaluator:

```text
depth 1: 40 nodes, 20 evals, ~205k nps
depth 2: 136 nodes, 58 evals, ~225k nps, 18 cutoffs
depth 3: 1850 nodes, 862 evals, ~139k nps, 94 cutoffs
```

Node-type split at depth 2: root 20, pv 39, cut 0, leaf 77. Cutoffs grow with depth as expected.

## Real evaluator in search (small-gab-fp32, startpos, depth 2)

```text
C++  + real eval: score -0.0193, 266 nodes, 123 evals, ~26k nps, 17 cutoffs
Rust + real eval: score -0.0202, 98 nodes, 39 evals, ~3k nps (debug build), 19 cutoffs
```

Scores agree within 1e-3. Node and eval counts differ because the two harnesses generate moves in different orders, and with a flat untrained evaluator the root move is order-dependent (b2-b3 versus a2-a3). Leaf evaluations themselves are parity-proven: C++ -0.0197625 against Rust -0.019763, diff ~7e-7, matching the v0.10 triangle. Same-model reruns are deterministic inside each language.

## Suite shape (depth 2, C++ harness)

Tactical lines run 230-642 nodes, endgame lines 14-38 nodes, reflecting real branching differences. Sparse endgames barely search, which is why lazy evaluation must prove it does not hurt exactly these positions.

## Hot path

Feature extraction ~7.4us plus accumulator refresh ~12.4us plus tokenization ~0.35us ride along with every evaluation, roughly 20us before the network even runs. Inside search that constant multiplies by thousands of nodes, which is why evaluator microbenchmarks overstate engine gains. No optimization was applied on intuition; the split above is the map.

## Caches and threads

Evaluator cache hit rate is reported per run and keyed on full state plus model hash. Four-thread C++ stress gives bit-identical results across threads. Performance CI stays a smoke and regression signal because hardware noise dominates small percentages.
