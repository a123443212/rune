# RUNE v0.4 — v0.3 Audit (evidence-gated, input to v0.4)

Scope: everything v0.3 built, measured against v0.4's question
(same intelligence, less unnecessary computation on CPU). A component
is PROVEN only with a trained comparison under identical conditions.
Bench, parity, and smoke runs measure cost and correctness, never gain.

Current honest status: **no trained runs exist in v0.3 either
(no `runs/` directory; all v0.3 configs were never executed — no
teacher labels exist in this repo state).** Every strength claim about
v0.3 components is therefore UNKNOWN. v0.3's output is scaffolding +
methodology, not empirical winners. v0.4 inherits that honestly.

## 1. What was inspected

- Code: `training/models/dense.py` (VarEmbedder, TokenPool
  none/per_token/shared, ChannelGate, DenseHead, presets A-D,
  ALLOCATION_H1, BUDGET_ALLOCS), `training/models/relational.py`
  (FlexEmbedder, RelationalMixer, RelationalHead),
  `tools/analysis/representation.py` (token-token cosine, channel
  correlation, effective rank, entropy, dead ratio, phase splits,
  pair sensitivity), `training/experiments/screening.py`
  (arch_version 0.3.0 fix, dense passthrough), `training/export/`
  (checksum, INT paths).
- C++: `core/architectures/dense/` (DenseModel, TokenPool,
  ChannelGate, VarAccumulator, DenseEvaluator with
  refresh/updateIncremental/evaluate/currentTokens),
  `benchmarks/stage_bench.cpp` (dense section), bindings
  (dense_bindings), `tests/cpp/test_dense.cpp`, `tests/test_dense.py`.
- Configs: all 11 files under `configs/v03/` (core, A0/A1/A2/A3,
  P0/P1, budget 128/192/320, mixer_check).
- Reports: `rune-v03-v02-audit.md`, `rune-v03-plan.md`,
  `rune-v03-results.md`, `docs/architecture/rune-v03.md`,
  `docs/future-work.md` (v0.3 parked list).

## 2. Verdicts

### KEEP (foundation for v0.4)

| Component | Evidence | v0.4 role |
| --------- | -------- | --------- |
| Grouped feature set + exact incremental accumulator | Incremental == refresh over 60 random moves + 30 unmakes, C++/torch parity 1e-4 | Frozen base. §36 reuse: refinement path must reuse these buffers, never recompute features |
| 8-token semantic contract + `token_dims` serialization + fnv1a checksum + arch version 0.3.0 | Covered by parity/IO/tamper tests | Stable token contract both paths share; package extended with refinement + routing fields (§37) |
| RUNE-03-A uniform 8x32, no pool, no gate | Only fully-compatible, cheapest proven-working form | De-facto A0 full evaluator until ablation promotes a successor (v0.3 never selected one — see UNKNOWN) |
| Screening harness + metrics.json + promotion gate + per-metric reporting | Smoke-run, dense-capable | KEEP, extended with refinement rate, routing mode, avg/worst latency, strength-per-average-compute |
| Stage-bench per-stage splits + dense section | Splits refresh/incremental/pool/gate/forward + param totals | KEEP, extended with difficulty/routing/cheap/refine/precision-conversion splits (§9, §30) |
| Deterministic incremental correctness tests | 60-move + unmake coverage incl. castling/ep/promotion FENs | KEEP, extended to routing determinism: same FEN → same routing → same eval (§25) |
| INT16 embedding path via int32 accumulator | Measured neutral on random weights only | P0 uniform-precision candidate; trained-weight numbers still missing |
| Equal-budget rule + no-composite-score reporting + kill-on-no-gain discipline | Methodology, v0.3's real output | Carried over unchanged (§23, §34, §42) |

### REMOVE (from v0.4 core candidate)

| Component | Evidence | v0.4 role |
| --------- | -------- | --------- |
| Dynamic GAB | Zero trained evidence, measured overhead suspicion since v0.2 | Stays dead. Not a refinement candidate |
| Relational mixer as default path | Still no 25M+ trained comparison after two versions | Experimental comparator for mixer_check only, never the refinement default |
| Ranking loss / disagreement sampling as default-on | Still smoke/pipeline only | Stay gated; adaptive training does not depend on them |
| Any stochastic/random routing in engine inference | Banned by §7/§25 by design | Never implemented, only threshold + optional hysteresis |
| Unbounded or recurrent refinement | Banned by §24 by design | Fixed single refinement step, fixed max cost |
| MoE / Transformer / Mamba / RL / neural search policy / giant uncertainty nets | §40 bans, no v0.3 evidence points at them | Parked in future-work, not implemented |
| Theoretical-FLOPs claims without CPU benchmark | v0.3 lesson (noisy-machine ranges) | §35 formula is bookkeeping only; hardware bench decides (§31) |

### REWORK (idea survives, form changes for adaptive use)

| v0.3 form | Problem for v0.4 | v0.4 form |
| --------- | ---------------- | --------- |
| Dense head T*D→128→32 frozen | Cheap path needs a reduced head; refinement needs its own refined head; one frozen shape cannot serve both | Two-head design: tiny cheap head (usable standalone, §4) + refined head on mixed tokens. Both costed separately |
| TokenPool per_token/shared, ChannelGate | Implemented + parity-tested, never trained → quality UNKNOWN | Reused as cheap-path mixer candidates and refinement substrate; verdicts still pending first 25M runs |
| H1 allocation [28,32,32,28,20,48,40,28] | Hypothesis never executed | Stays hypothesis; difficulty/error analysis (§14) may re-derive it, never intuition |
| Quantization global-only, random-weight numbers | Says nothing about trained nets or easy/hard precision needs | Selective precision legs P0/P1/P2 on trained weights with real target-CPU bench (§15-§16) |
| bench/infer_bench/quant_bench (single-latency view) | Cannot express average cost over a refinement-rate sweep | Average-cost matrix 10/25/50/75/100% + routing-overhead splits + worst-case (§8-§9, §29-§30) |
| play_match.py (no adaptive awareness) | Cannot log refinement %, nodes, avg eval latency under fixed conditions | Match harness logs routing + cost alongside result (§28) |
| representation.py (redundancy-only) | No difficulty notion: no teacher error, no uncertainty, no instability | Extended with difficulty statistics (§14) + oracle routing analysis (§11) + FP/FN breakdown (§13) |
| 8x8 relational interaction matrix | Full-only, no pruning notion | Pruned-interaction experiment with retrain + tolerance check (§19); needs new analysis for interaction contribution |

### UNKNOWN (needs data; v0.4 must produce it, smallest test first)

- The central hypothesis itself: whether easy/hard positions exist in
  a cheaply-routable sense. v0.3 contributes zero evidence here. The
  oracle experiment (§11) is therefore the mandatory first empirical
  step — if even oracle routing shows no trade-off, learned routing
  work stops.
- Cheap-path standalone quality (A1) and refinement gain over cheap
  (A2 vs A1). Separates refinement quality from adaptive savings (§10).
- Oracle ceiling and learned-vs-oracle gap (R0 vs R1).
- Which refinement rate (10/25/50/75%) gives a notable trade-off.
- Whether dynamic routing is actually faster on CPU once branch/cache
  effects are measured (§31) — static routing stays a live alternative
  (§18).
- Selective precision benefit on the real target CPU (INT8 vs
  INT16/FP16 paths; FP16 may not help — §15).
- Interaction pruning viability (I0 vs I1 with retrain).
- Search stability under routing flips; whether hysteresis is needed
  (§26-§27, default off).
- Optimal total dims / allocation (v0.3 ladder never ran).
- Carried UNKNOWNs: static GAB effect, channel-gate effect, shared
  projection effect, ranking/disagreement value.

## 3. Answers to the v0.4 entry questions

1. **V0.3 bottleneck thực sự là gì?** No trained bottleneck exists —
   v0.3 never ran. Methodologically: uniform 8x32 capacity assumption
   untested, per-token sensitivity unmeasured, budget ladder unrun,
   mixer necessity undecided. v0.4 must not treat any v0.3 variant as
   "the full model" without the A-ladder first promoting one.
2. **Representation có redundancy ở đâu?** Unknown empirically. The
   tool to find out exists (`representation.py`, unrun). v0.4 extends
   it rather than guessing.
3. **Adaptive computation có evidence nào không?** None. Zero routing,
   zero difficulty, zero average-cost numbers anywhere in the repo.
   Oracle-first discipline (§1) is not optional.
4. **Component nào chỉ tăng compute?** Dynamic GAB, default QKV mixer,
   uniform 10x40 scaling — same list as v0.2, still valid.
5. **Failed experiments (v0.3):** screening wrote arch_version 0.1.0
   for RUNE-03 (fixed before any run consumed it); dense presets all
   shipped [32]*8 making allocation inexpressible (fixed with
   ALLOCATION_H1/BUDGET_ALLOCS + per-leg configs). Lesson for v0.4:
   keep leg-specific parameters in leg-specific configs, never in
   shared presets.

## 4. Consequence for v0.4

v0.4's A0 is RUNE-03-A uniform 8x32 until the carried-over A-ladder
promotes a successor — the ladder and the adaptive track run on the
same seed/data/positions so results compose. Nothing adaptive enters
the core without beating two gates in order: oracle ceiling exists,
then learned routing approaches it, then real-CPU average cost wins
at equal quality. A smaller, simpler, faster model at equal results
is preferred over a clever router with no measured gain — same
philosophy as v0.3, applied to computation instead of dimensions.
