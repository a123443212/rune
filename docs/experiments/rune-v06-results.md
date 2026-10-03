# RUNE v0.6 Results (running log)

Status: audit + scaffolding complete, no teacher and no D-leg runs
yet. Nothing here is a conclusion until T0 exists and matched
direct-vs-distilled comparisons run.

## Completed (code-level, no strength claims)

- v0.5 audit: `docs/experiments/rune-v06-v05-audit.md`
  (KEEP / REMOVE / REWORK / UNKNOWN).
- Architecture + plan: `docs/architecture/rune-v06.md`,
  `docs/experiments/rune-v06-plan.md`.
- Distillation losses: `training/losses/distillation.py`
  (Value/WDLL/Ranking/Distillation/WeightedDistillation,
  `L = L_task + α L_distill`, bounded confidence/difficulty
  weights, uniform control).
- Student builders: configurable token dims + head widths for
  dense and adaptive families (`training/models/students.py`,
  S1–S4 presets as compute targets).
- Teacher labeling: `tools/dataset/label_distillation.py`
  (teacher value/WDL/uncertainty + optional ranking pairs,
  teacher id/hash, cost accounting).
- Analysis: `tools/analysis/compression.py` (frontier curves),
  retention battery, `delta_analysis.py` (Δteacher vs Δstudent),
  failure-analysis doc skeleton.
- Configs: teacher + S-budgets + D-legs + Q-legs under
  `configs/v06/`.
- C++ student support where it fell out of shape-driven IO
  (see profiling log); fixed-head kernels otherwise untouched
  until profiling justifies work (§30).
- Student head scaling in C++ (dense headH1/H2, adaptive
  cheapHidden/refH1/refH2) with IO roundtrips: full `rune_tests`
  build passes (2 new tests: dense S2 + adaptive S2 save/load and
  eval agreement); match harness loads RUNE-03 students (incl.
  custom head widths) and logs mean uncertainty for RUNE-05.

## Pending (in plan order)

1. T0 teacher at meaningful scale (first empirical result).
2. Calibration gate → D-legs A0–A3 at S2.
3. Budget sweep, retention, delta, sampling, Q-legs.
4. Search-behavior preservation, 250M, deployment selection.

## Interpretation rules

Three-way reporting (student/teacher/target) always. Rejected
compression levels are results. Promotion human, per-metric,
never a single score.

## Failed experiments / fixes

- Trainer `unpack` routed len-5 teacher batches into the context
  slot (`wdl.to` on a dict). Caught by the first distill smoke.
  Fixed: teacher batches are len-6 with explicit None ctx.
- Distill metric merge wrote `dist_*`-prefixed keys while `tot`
  expected unprefixed ones (`distill` read 0.0). Fixed the merge
  mapping; smoke re-run shows consistent distill MSE ≈ MAE².
- (log here AND mirror to `docs/experiments/failed/` per v0.5 §36)
