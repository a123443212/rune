# RUNE v0.4 Results (running log)

Status: audit + scaffolding complete, no trained runs yet. Nothing
here is a conclusion until 25M+ comparisons with identical
seed/data/positions exist, plus real-CPU average-cost benches.

## Completed (code-level, no strength claims)

- v0.3 audit: `docs/experiments/rune-v04-v03-audit.md`
  (KEEP / REMOVE / REWORK / UNKNOWN).
- Architecture + plan: `docs/architecture/rune-v04.md`,
  `docs/experiments/rune-v04-plan.md`.
- Adaptive model (Python, C++ parity pending bindings build):
  `training/models/adaptive.py` — cheap path, refinement block,
  difficulty scalar, deterministic threshold routing with optional
  hysteresis (default off).
- Oracle + difficulty tooling: `tools/analysis/oracle_routing.py`
  (ceiling sweep, learned-vs-oracle gap, FP/FN by phase/material/
  tactics), difficulty stats extension point in `representation.py`.
- Configs: A0/A1/A2/A3, routing sweep, P0/P1/P2, I0/I1 under
  `configs/v04/`.
- Match/bench harness fields: refinement %, avg/worst latency,
  nodes searched (play_match.py logs per-side evals, refinement
  rate, avg latency us; RUNE-04 loads via AdaptiveModel binding).
- C++ adaptive runtime verified: full `rune_tests` build passes
  (incl. 6 new adaptive tests: configure/routing/hysteresis/
  incremental/IO/accounting); `rune_stage_bench` adaptive section
  reports cheap/always/adaptive splits + route cost + equality
  checks (see profiling log for first noisy-machine numbers).

## Pending (in plan order)

1. Oracle ceiling on teacher-labeled pool (needs teacher labels —
   currently the repo's hardest blocker, same as v0.2/v0.3).
2. A0 25M full baseline (RUNE-03-A until ladder promotes).
3. A1/A2 legs, then A3 sweep at 10/25/50/75/100%.
4. R0 vs R1, P-legs on target CPU, I-legs with retrain.
5. Stability analysis, 250M confirmation, controlled match.

## Interpretation rules

Gates in order: oracle ceiling → learned approach → CPU average-cost
win at equal quality. Capacity wins labeled capacity. Danger misses
(hard→cheap on tactics/king-danger) reported separately, never
averaged away. Promotion human, per-metric, never a single score.

## Failed experiments / fixes

- `loadRuneFile` routed RUNE-04 into the dense branch (shared
  `if (out.isDense)` condition after a broad replace), crashing on
  `arch.substr(8)`. Caught by the new `testAdaptiveModelIO`
  roundtrip test. Fixed by scoping the condition back to dense-only.
  Lesson: keep arch-dispatch conditions exact; the IO roundtrip test
  is the guard.
- (log here; keep leg params in leg configs, never shared presets)
