# RUNE v0.6 — v0.5 Audit (evidence-gated, input to v0.6)

Scope: everything v0.5 built, measured against v0.6's question
(strong teacher → compact student → efficient INT8 deployment at
minimal quality loss, never a bigger model). A component is PROVEN
only with a trained comparison under identical conditions. Bench,
parity, and smoke runs measure cost and correctness, never gain.

Current honest status: **no trained runs exist in v0.5 either
(no `runs/` directory; the only labels ever produced are
`synth_mlp_v1` on a 30-position curated pool for pipeline smoke
tests).** Consequences for v0.6, stated plainly:

1. There is no trained "best v0.5 candidate", so T0 cannot be
   selected — it must first be produced (a real teacher at
   meaningful scale). Any distillation run before that is
   pipeline validation only.
2. The v0.5 uncertainty head never saw teacher error from a real
   teacher; its calibration is UNKNOWN, so §9 fires by default:
   no uncertainty weighting until the calibration gate passes.
3. The stability head never saw child data (no parent-child
   dataset exists); it is an untrained mechanism, not a signal.

## 1. What was inspected

- Code: `training/models/uncertainty.py` (`SearchAwareModel` =
  adaptive + bounded sigmoid uncertainty + clamped stability head,
  `route_multi` R0–R3, `inference` modes), `training/losses/
  uncertainty.py` (`SearchLoss`: adaptive base + MSE-to-
  `|teacher−student|/2` + masked stability MSE, all λ-gated),
  `training/models/adaptive.py` + `dense.py` (fixed head widths:
  cheap 32, refined 128/32; dims configurable, heads not),
  `training/trainer/trainer.py` (RUNE-05 branch, unc/err Pearson
  tracking in `evaluate`), `training/experiments/screening.py`
  (rune_05_* keys, loss flags, version 0.5.0 records),
  `training/export/export.py` (uw/ub/sw/sb tensors, uncertainty
  flags, RUNE-05 checksum).
- C++: `core/architectures/adaptive/` (uncertainty/stability
  forwards, `routeSearch`, RUNE-05 IO roundtrip + tamper tests
  passing), stage-bench search section (unc+stab 0.39 us vs cheap
  7.8 us on random weights, noisy machine), bindings
  (6-tuple eval, `eval_search`).
- Analysis: `calibration.py` (curve, buckets, Spearman,
  stratification, danger-miss), `stability.py` (spread, flips,
  spikes, sharpness battery, rule-labeled subsets),
  `oracle_routing.py`, `representation.py` — all executed on
  smoke outputs only.
- Configs: `configs/v05/` (A0–A3, R-modes, L0–L2).
- Reports: `rune-v05-v04-audit.md`, `rune-v05-plan.md`,
  `rune-v05-results.md`, `rune-v05-profiling.md`,
  `docs/experiments/failed/` (index + template, v0.4 entries
  referenced).

## 2. Verdicts

### KEEP (foundation for v0.6)

| Component | Evidence | v0.6 role |
| --------- | -------- | --------- |
| Value-first discipline + A-ladder method | Code + passing tests; no trained contradiction | Student quality is value error first; distillation is auxiliary (§19 decides its fate) |
| Bounded uncertainty head (single linear + sigmoid, shared rep) | Code + C++ parity tests + 0.39 us bench; quality UNKNOWN | Teacher-uncertainty source for D2/D3 weighting — after the calibration gate, never before (§9) |
| SearchLoss λ-gating + one-axis ablations + uncorrelated-u kill rule | Methodology, enforced in plan | Direct template for the distillation loss family (§10) and stop rules (§37) |
| Calibration tooling (curve, buckets, rank corr, stratification, danger-miss) | Runs on smoke outputs; logic reviewed, untested on real errors | Gates confidence weighting; extended with student-calibration comparison (§23) |
| Stability analysis tooling + rule-labeled subsets | Same status | Retention/sharpness battery for S-legs (§17, §22); subsets reused for failure analysis (§36) |
| Deterministic routing + validation-only thresholds + operating points | Tested, unchanged rule | Student keeps difficulty/uncertainty routing; thresholds re-calibrated per student, never inherited blindly |
| Checksum/versioning/package discipline | Roundtrip + tamper tests pass | Extended with teacher id/hash, dataset hash, loss config per §32; C++ student loads teacher-free |
| Match harness (evals, refinement %, latency, mean-u, fixed openings, CI) | Code, unrun on real models | Teacher-vs-student engine comparison + delta analysis harness (§24–§25) |
| INT8 export paths + int32 accumulator discipline | Code-level, random-weight numbers only | Q0/Q1/Q2 legs; calibration-under-quant check is a kill condition (§28 carried over) |
| Per-metric reporting, no composite score; failed/ log | Methodology | Unchanged (§38). Compression rejected loudly on kill criteria (§37) |

### REMOVE (from v0.6 candidacy)

| Component | Evidence | v0.6 role |
| --------- | -------- | --------- |
| Uncalibrated-uncertainty weighting | v0.5 calibration UNKNOWN by construction (no real teacher error observed) | Banned until the gate passes; D2 runs only after (§9) |
| Stability-head engine use | No child data exists; head never trained | Analysis-only until spread shows signal with retrain (§12–§13 logic carried over) |
| Proxy-difficulty as evidence | Standing removal since v0.4 audit | Stays out; difficulty weights need the same calibration gate as uncertainty |
| Dynamic GAB, unproven mixers as defaults, policy/search-algorithm changes, MCTS/RL | Standing bans, no new evidence | Stay parked in future-work |
| Test-set threshold tuning; composite scores; "student" at teacher scale | Standing bans | §3 enforces strictly-smaller students; §38 enforces reporting |

### REWORK (idea survives, form changes for distillation)

| v0.5 form | Problem for v0.6 | v0.6 form |
| --------- | ---------------- | --------- |
| `SearchLoss` (task + unc + stab) | No teacher-student terms; no weighting functions; no ranking-distillation split | Distillation loss family: Value/WDLL/Ranking/Distillation/WeightedDistillation components, `L = L_task + α L_distill`, bounded `w(x)` in at least two variants (confidence, difficulty), uniform-weighting control always present (§7, §10) |
| Fixed head widths (cheap 32, refined 128/32) | S1–S4 students cannot be expressed; heads dominate cost (v0.4 bench: refined head ~44 us of ~53 us) | Configurable head widths + token dims in student builders; IO already shape-driven (getTensors/setTensors carry shapes), so artifacts stay self-describing (§4–§5) |
| Uncertainty target `\|teacher−student\|/2` inside task training | Distillation needs the reverse direction: teacher predictions as targets, teacher-vs-ground-truth reported separately | Teacher-labeling pipeline: precomputed teacher value/WDL/uncertainty (+ optional ranking pairs) on the pool with teacher id/hash, storage cost reported; three-way reporting student-vs-teacher, student-vs-target, teacher-vs-target (§11, §31) |
| Difficulty-aware sampling (disagreement sampler, pipeline-only) | Never ran; drops-easy-samples risk unmeasured | Uniform-vs-difficulty sampling experiment keeping easy mass (§15); curriculum only on signal (§35) |
| Oracle routing tool (teacher-error ceiling) | No teacher-vs-student distillation ceiling notion | Extended with distillation-ceiling analysis: what a perfect student could keep at each budget |
| Match harness (score + cost) | No delta analysis, no retention subsets, no teacher-vs-student pairing | Delta test (Δteacher vs Δstudent over legal moves, tactical-weighted) + retention battery (§17, §24–§26) |
| Representation analysis (redundancy view) | No teacher-student representation comparison | Representation-distillation view with explicit alignment (small projection when dims differ), tried only after value distillation signals (§12–§13) |

### UNKNOWN (needs data; cheapest test first, same discipline)

- Teacher selection: no candidates exist, so the first empirical
  question is producing T0, not choosing among finished models.
- Every RQ1–RQ5 answer: transfer limits per budget, weighting
  efficacy, easy/hard asymmetry, cliff location, search preservation.
- Whether representation/layer-wise/ranking distillation pay for
  their complexity; whether QAT beats post-quant; whether INT8
  preserves calibration on a real student.
- Whether delta-aware or curriculum training helps or harms.
- All carried UNKNOWNs (static GAB, gate, shared projection,
  ranking/disagreement value, optimal dims).

## 3. Answers to the v0.6 entry questions

1. **V0.5 bottleneck thực sự là gì?** No trained bottleneck exists —
   v0.5 never ran past smoke. Structurally: fixed head widths block
   student budgets; no distillation terms exist; no teacher-labeled
   pool exists; stability has no data source. These four gaps are
   v0.6's build list, in that order.
2. **Teacher nào đáng dùng?** Undecided and undecidable today. The
   audit forbids defaulting to "RUNE-05 because newest": T0 is the
   first benchmark result of v0.6, selected on validation +
   calibration + cost, not version recency (§1).
3. **Uncertainty v0.5 dùng được cho weighting không?** Mechanism yes,
   permission no — calibration gate first (§9). The head's form
   (bounded, cheap, shared-rep) is exactly what weighting needs;
   its numbers mean nothing yet.
4. **Failed experiments carried:** evaluate-KeyError fix, batch-pad
   fix, test arg-order fix, IO dispatch fix (all in-results, all
   with regression tests). No failed *experiments* exist yet
   because no 25M+ experiment has ever run in this repo — v0.6's
   failure log must become real, not scaffolding.

## 4. Consequence for v0.6

Build order is forced by the audit: (1) student-expressible
architectures (configurable widths) + distillation losses with
uniform control; (2) teacher-labeling pipeline with cost accounting;
(3) smallest-budget D-legs on whatever teacher exists (synth for
pipeline, real T0 the moment it exists); (4) compression curve +
retention + delta + search behavior; (5) INT8/QAT. Any step that
fails its gate stops its track. A rejected compression level is a
result, not a delay — the frontier with holes is the honest
deliverable (§43).
