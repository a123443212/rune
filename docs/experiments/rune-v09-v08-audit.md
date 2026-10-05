# RUNE v0.9 — v0.8 Audit (evidence-gated, input to v0.9)

Scope: everything that produces or consumes teacher supervision,
measured against v0.9's question (how trustworthy is what RUNE
learns from). A component is PROVEN only with a matched-budget,
matched-teacher comparison. Bench, parity, and smoke runs measure
cost and correctness, never label quality.

Current honest status: **the loop machinery runs end-to-end, but
no teacher has ever been characterized.** Every label in the repo
comes from Stockfish d10–d12 on ≤4.5k positions or a synth MLP.
No depth-stability number, no teacher-bias measurement, no
phase-specific error, no consistency check, and no engine match
exist. Teacher quality is 100% UNKNOWN — which is exactly why v0.9
exists.

## 1. What was inspected

- Teacher generation: `tools/dataset/label_engine.py` (Stockfish
  via python-chess, cp→tanh value, hard WDL thresholds, WDL-spread
  uncertainty, per-run meta with positions/sec + nodes total),
  `label_teacher.py` (synth MLP + legacy aliases),
  `label_distillation.py` (torch-teacher labeling).
- Teacher identity: `teacher_id` strings (`stockfish17_d10`,
  `synth_mlp_v1`), `teacher_hash` over (fen, value, id) in
  screening records. **Missing: engine version string, network
  id/hash, per-position nodes/depth reached, threads/hash MB in
  records, search settings — a dataset stamped `teacher=Stockfish`
  cannot say which network or limits produced it (§44 fails).**
- Active loop: `tools/active/` (score/select rounds, coverage,
  retention) + Rust `score`/`select` + schema v2 + round.yaml.
  One full round-0 smoke ran (synth seed → uncertainty select →
  stockfish labels → mixed train). Mechanics proven, learning
  unclaimed.
- Supervision signals: value + hard WDL + sibling ranking (pairs
  need bindings — available now but ungenerated at scale) +
  proxy difficulty (BCE, flagged) + bounded uncertainty head
  (never calibrated on real teacher error).
- Analysis: oracle ceiling, calibration (curve/buckets/Spearman),
  stability (spread/flips/spikes), compression frontier,
  delta analysis — all executed on smoke outputs only.
- Match harness: fixed conditions + CI fields, unrun on real
  models. Distillation losses + student budgets + configs exist
  but never met a characterized teacher.

## 2. Verdicts

### KEEP (v0.9 builds on these unchanged)

| Component | Evidence | v0.9 role |
| --------- | -------- | --------- |
| Closed-loop machinery (rounds, audit trail, round.yaml, reuse split, coverage, retention) | Smoke-validated end-to-end | The harness multi-depth/cascade experiments run inside |
| Teacher cost accounting (positions/sec, nodes, no-free-lunch warning) + teacher_hash | In tooling and records | Extended to per-level cascade budgets and total-compute curves (§23, §50) |
| stm-relative convention + perspective stamping | Forced by data, fixed everywhere | Target schema keeps it; joins reject mismatches |
| Calibration + stability + oracle analysis tooling | Runs on smoke; logic reviewed | Extended with depth-stability, agreement, and teacher-vs-student calibration views |
| Deterministic selection/routing + validation-only thresholds | Tested | Quality-weighted routing joins as one more gated signal, same rules |
| Per-metric reporting, failed log, no-composite-score discipline | Methodology | Unchanged (§60 answers are per-metric or they are not answers) |

### REMOVE (must not shape v0.9 conclusions)

| Component | Evidence | v0.9 role |
| --------- | -------- | --------- |
| "Deeper is better" as an axiom | Zero measurements at any second depth | Replaced by measured convergence curves (§5); depth is a budget knob with diminishing returns (§53) |
| Uncalibrated-uncertainty weighting, proxy-difficulty evidence | Standing bans since v0.4/v0.5 | Stay banned; quality weights need the §18 gate |
| Strict ranking on near-ties | Margin exists but ties unhandled; noise risk unmeasured | Replaced by strict/soft/tie handling with benchmarked thresholds (§27–§28) |
| Silent treatment of teacher disagreement/instability | No categories exist | REVIEW/REJECT/KEEP buckets + validity states (§11, §42) |
| Test-set tuning, composite scores, full-Cartesian runs, policy/search-algorithm changes | Standing bans | Stay banned (§61) |

### REWORK (same need, stronger form)

| v0.8 form | Problem for v0.9 | v0.9 form |
| --------- | ---------------- | --------- |
| Single-depth labels (`--depth N`, one value per position) | No convergence signal; stability unmeasurable | Multi-depth records (low/med/high/ref on a subset) + teacher-delta/WDL-flip/rank-flip metrics (§5–§6) |
| Flat `teacher_id` string | Cannot distinguish network/config/limits; migration unsafe | Full provenance record (§14, §56) + versioned target stores (§36, §45) |
| Hard WDL from value thresholds | Throws away engine WDL probabilities; calibration untestable | Soft WDL distillation vs hard WDL, ablated, with calibration as the metric (§26) |
| Ranking on raw pairs | Near-tie noise forced into strict constraints | Strict/soft/tie triple with configurable margin (§27–§28) |
| `teacher_hash` over (fen, value, id) | Blind to config drift behind the same id | Hash covers full provenance record (§30, §32) |
| Synth/random-init stand-ins | Mechanics-only, correctly labeled | Kept strictly for pipeline tests; never a quality baseline (§48 uses real teachers only) |

### IMPROVE (v0.8-correct, v0.9-hardened)

- Deterministic teacher mode (fixed threads/hash/settings/version/seed) + same-position-same-target consistency checks that distinguish infra bugs from noise (§12–§13).
- Target validity states (VALID/PROVISIONAL/UNSTABLE/INVALID) with config-documented criteria, enforced before materialization (§11, §43–§44).
- Teacher confidence proxy from convergence + agreement + WDL/rank consistency — explainable, no new neural model (§18).
- Cascade routing on deterministic metadata (stable→accept, else escalate; fixed depth ladder, costed per level) (§21–§22, §34).
- Difficulty-aware active sampling extended with teacher-quality signals, ablated per component (§32–§33).
- Rust target commands (validate/stats/verify-targets) + storage efficiency measurement (§35, §39–§41).

### UNKNOWN (cheapest test first, same discipline)

- Label convergence vs depth/time; stability distributions; phase/material-specific teacher error (RQ1).
- Cheap unreliability detection without deep search everywhere (RQ2).
- Whether rich targets (value+WDL+rank+search metadata) beat plain ones at equal positions (RQ3).
- Multi-depth/cascade learning-curve and budget effects (RQ4).
- Target-engineering data efficiency; teacher-vs-architecture bottleneck ranking at current scale (RQ5–RQ6).
- Ensemble value, compression safety, minimum useful teacher effort.
- All carried UNKNOWNs (real-model rounds, 25M+ legs, QAT, pruning, rates, starvation proof).

## 3. Consequence for v0.9

Build order is forced: (1) provenance + multi-depth labeling on a
subset (measure convergence before theorizing); (2) validity +
stability + confidence metadata with cheap gates; (3) cascade on
deterministic routing with per-level budgets; (4) target-form
ablations (soft WDL, near-tie ranking, quality weighting) at fixed
architecture and positions; (5) student legs + search-relevant
validation + total-compute curves. The first empirical output is a
teacher-stability report, not a model — if labels don't converge
where it matters, that finding outranks any architecture work and
the v1.0 default stays simple.
