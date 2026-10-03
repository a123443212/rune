# RUNE v0.5 — v0.4 Audit (evidence-gated, input to v0.5)

Scope: everything v0.4 built, measured against v0.5's question
(value + confidence + stability + adaptive computation at minimal
added compute, never a policy network). A component is PROVEN only
with a trained comparison under identical conditions. Bench, parity,
and smoke runs measure cost and correctness, never gain.

Current honest status: **no trained runs exist in v0.4 either
(no `runs/` directory; teacher labels still missing, so the oracle
experiment, difficulty training, and every A/R/P/I leg never ran).**
v0.4's verified output is: C++ adaptive runtime (all cpp tests pass),
stage-bench splits on random weights (noisy machine), Python adaptive
model + loss + oracle tooling (unexecuted — no torch in this env),
match-harness logging fields, and methodology. Every quality claim
about cheap/refinement/difficulty/precision/pruning is UNKNOWN.

## 1. What was inspected

- Code: `training/models/adaptive.py` (CheapHead, DifficultyHead,
  RefinementBlock static-only, RefinedHead, deterministic
  threshold/hysteresis routing, `infer` modes), `training/losses/
  adaptive.py` (cheap+refined value/WDL, rank on refined, difficulty
  BCE with oracle-or-proxy target), `tools/analysis/oracle_routing.py`
  (ceiling sweep, learned gap, FP/FN buckets, entropy-error
  correlation), `tools/analysis/representation.py` (redundancy +
  unsupervised difficulty proxies, clearly labeled),
  `tools/match/play_match.py` (per-side evals, refinement rate,
  avg latency), `configs/v04/` (A0/A1/A2/A3, P1, I1).
- C++: `core/architectures/adaptive/` (cheap/refine forwards,
  route(), AdaptiveEvaluator reusing VarAccumulator), model_io
  RUNE-04 load/save + checksum + version 0.4.0, bindings, stage-bench
  adaptive section, `tests/cpp/test_adaptive.cpp` (6 tests, passing).
- Reports: `rune-v04-v03-audit.md`, `rune-v04-plan.md`,
  `rune-v04-results.md`, `docs/performance/rune-v04-profiling.md`
  (noisy-machine numbers: cheap 7.6 us vs always 53.2 us incremental,
  route ~1 ns, refined head dominates — ratios only).

## 2. Verdicts

### KEEP (foundation for v0.5)

| Component | Evidence | v0.5 role |
| --------- | -------- | --------- |
| Value-first output discipline (cheap + refined heads, A0–A3 ladder) | Code + passing C++ tests; zero trained evidence | A0 baseline. Uncertainty/stability stay auxiliary; value is never displaced (§3) |
| Deterministic threshold routing, hysteresis default off, same-FEN determinism | Tested (`testAdaptiveRoutingModes`, hysteresis unit test) | Routing substrate R0–R3 build on. Hysteresis ships only on measured instability, unchanged rule |
| Oracle methodology + FP/FN buckets + danger-miss rate | Tool exists, unrun for lack of teacher | Extended with calibration curves, error stratification, parent-child spread; danger misses stay separately reported (§16) |
| WDL entropy as a free signal | Implemented in Python (`wdl_entropy`), zero params | R1 baseline and calibration reference before any learned head is trusted |
| Match harness (evals, refinement %, avg latency, fixed openings, CI reporting) | Code, unrun | Extended with uncertainty/stability columns + search-aware subsets (§17, §25) |
| Checksum/versioning/package discipline (0.4.0) | Roundtrip + tamper tests pass | Package extended with uncertainty fields; v0.1–v0.4 files keep loading |
| Per-metric reporting, no composite score; gated plan order | Methodology | Unchanged (§33). Failed log moves to `docs/experiments/failed/` (§36) |
| Static-only refinement, fixed single step, bounded worst case | Tested | Unchanged; v0.5 adds signals, not compute steps |

### REMOVE (from v0.5 core candidacy)

| Component | Evidence | v0.5 role |
| --------- | -------- | --------- |
| Proxy difficulty supervision presented as difficulty evidence | `AdaptiveLoss` falls back to cheap-vs-refined disagreement when no oracle mask exists — that measures head disagreement, not teacher error | Flagged proxy stays for bootstrapping only; no routing/calibration claim may rest on it (§5, §26) |
| Dynamic GAB, unproven mixer variants as defaults | Still zero trained evidence after three versions | Stay dead/experimental |
| Hysteresis-on, stochastic routing, any same-FEN nondeterminism | Banned by design, tested off | Stay off/banned (§14) |
| Policy network, neural move ordering, search-algorithm changes, MCTS/RL | §39 bans, no v0.4 evidence points at them | Parked in future-work |
| Average-cost win claims without matched quality | Standing rule | Unchanged (§25, §33) |

### REWORK (idea survives, form changes for uncertainty/stability)

| v0.4 form | Problem for v0.5 | v0.5 form |
| --------- | ---------------- | --------- |
| `DifficultyHead` (unbounded linear scalar) | v0.5 needs bounded, quantizable uncertainty `u ∈ [0,1]` with calibration meaning | New bounded uncertainty head alongside difficulty; difficulty keeps routing compat until R-experiments subsume or retire it |
| `AdaptiveLoss` difficulty BCE (oracle-or-proxy) | No regression-to-teacher-error objective; no stability objective | `L0 = Value+WDL`, `L1 = +Uncertainty`, `L2 = +Stability`, each λ-gated with its own ablation; uncertainty target is `\|teacher−student\|` only (§5, §11) |
| Oracle tool (ceiling + entropy correlation) | No calibration curves, error buckets, rank correlation, stratification, or parent-child statistics | Extended calibration + stability analysis tooling (§18–§20); oracle comparison adds v0.5 routing as third curve (§34) |
| Representation difficulty proxies (unsupervised) | Proxies, not calibration | Superseded by supervised calibration once teacher exists; proxies stay labeled as such |
| Match harness (score + cost only) | No uncertainty/stability/search-subset columns | Search-aware subsets with explicit labeling rules; per-subset error, calibration, routing accuracy (§17) |
| Precision metadata (`refine_precision` flag) | Flag only, no bench, no calibration-under-quant check | Real INT8 uncertainty bench; calibration must survive quantization or the head is not deployment-ready (§28) |
| Validation-threshold practice (sweep at inference) | Thresholds never calibrated (no validation data) | Calibration on validation only, operating points 10/25/50/75/100% reported separately (§15) |

### UNKNOWN (needs data; v0.5 must produce it, cheapest test first)

- The central question: can uncertainty predict teacher error, and
  does it beat difficulty for routing (R0 vs R1 vs R2 vs R3).
- Whether any stability signal carries search-relevant information
  beyond value spread; whether child-aware training helps.
- Whether auxiliary objectives preserve tactical sharpness (flatten
  risk is the explicit kill condition, §22).
- Whether quantization breaks calibration.
- Whether stability analysis shows real flips vs legitimate sharp
  changes (§21).
- All carried UNKNOWNs: static GAB, channel gate, shared projection,
  ranking/disagreement value, optimal dims/allocation, selective
  precision benefit, interaction pruning, refinement rates.

## 3. Answers to the v0.5 entry questions

1. **V0.4 bottleneck thực sự là gì?** No trained bottleneck exists —
   v0.4 never ran. Measured (random weights, noisy): refined head
   dominates refinement cost; routing compare negligible; cheap path
   ~7x cheaper than always-refine IF quality holds (unproven).
   Methodologically: difficulty supervision has no teacher ground
   truth, thresholds uncalibrated, stability unmeasured.
2. **Adaptive computation có evidence nào không?** None on quality.
   Cost-side microbench exists; everything routing-quality is UNKNOWN.
3. **Difficulty estimator của v0.4 dùng được cho v0.5 không?** As a
   routing scalar with proxy supervision: usable scaffolding, not
   evidence. Its unbounded form is wrong for calibration; v0.5 adds
   a bounded head rather than reinterpreting difficulty as confidence.
4. **Failed experiments (v0.4):** `loadRuneFile` dispatch bug routing
   RUNE-04 into the dense branch (caught by roundtrip test, fixed,
   lesson: exact arch-dispatch + IO test as guard). Proxy-difficulty
   confusion risk documented in-results (lesson carried into §5/§26
   leakage rules).

## 4. Consequence for v0.5

v0.5's A0 is the v0.4 adaptive model (RUNE-04) with difficulty-only
routing — itself unpromoted, so the v0.4 A-ladder debt carries over:
no adaptive claim stands until 25M+ legs run on teacher data. New
signals enter only through the A0–A3 / R0–R3 / L0–L2 ladders at
matched budgets, smallest scale first, with tactical-sharpness and
calibration-under-quant as kill conditions. Anything that does not
reduce dangerous cheap-path decisions or improve search behavior at
negligible CPU cost leaves the core — same philosophy as v0.3/v0.4,
applied to awareness instead of dimensions or computation.
