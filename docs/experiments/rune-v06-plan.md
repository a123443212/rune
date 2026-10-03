# RUNE v0.6 Experiment Plan

Audit input: `docs/experiments/rune-v06-v05-audit.md`.
Architecture: `docs/architecture/rune-v06.md`.

## Order (gated — each step can stop its track)

1. Produce T0: train the teacher candidate at meaningful scale and
   benchmark it (validation + calibration + cost). No T0, no D-legs
   beyond pipeline smoke.
2. Calibration gate on teacher uncertainty. Fail → D2/D3 wait;
   D0/D1 proceed.
3. Core ablation at one fixed student (S2 first): A0 direct vs A1
   value-distill vs A2 confidence-weighted vs A3 +ranking.
   Architecture frozen across the loss ladder (§34).
4. Budget sweep S1–S4 on the winning D-leg → compression curve
   (size vs quality, cost vs quality).
5. Retention + delta analysis per budget (quiet/tactical/king-attack/
   imbalance/endgame/queenless; Δteacher vs Δstudent over legal
   moves, tactical-weighted).
6. Sampling: uniform vs difficulty-aware (easy mass kept); curriculum
   only on signal.
7. Representation/layer-wise distillation only on prior signal,
   with retrain + instability watch.
8. Q0/Q1/Q2 on promoted students (INT8 error, calibration drift,
   latency, NPS, bytes, match).
9. Search-behavior preservation: teacher engine vs student engine
   (nodes, NPS, latency, tactics, stability, match). Static metrics
   never substitute for this.
10. 250M confirmation for deployment candidates only.

## Mandatory matrices (controlled subsets, never full Cartesian)

```
T0  benchmarked teacher
S0..S4 budgets (100/75/50/33/25% compute targets)
D0 direct / D1 value / D2 value+WDL / D3 uncertainty-weighted / D4 +ranking
Q0 FP32 / Q1 INT8 / Q2 QAT-INT8
```

## Metrics per leg (no composite score)

Quality (value/WDL/rank error, three-way teacher/target splits),
uncertainty calibration (student and drift), params/bytes/ops,
latency/NPS/memory, refinement behavior, match with uncertainty,
teacher generation + labeling + storage + training costs separately.

## Stop rules (kill loudly, log to failed/)

- Distillation ≤ direct at matched budget → distillation leaves
  (technique charm is not evidence, §19).
- Tactical sharpness loss, endgame collapse, or bad calibration →
  reject that compression level (§37).
- No NPS/bytes win despite fewer params (bad layout) → not a
  successful compression (§29).
- Search degradation beyond static-metric prediction → reject.
- Uncalibrated-teacher weighting, easy-starved sampling, unstable
  layer-wise terms → removed on measurement.
