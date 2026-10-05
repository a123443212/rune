# RUNE v0.9 Experiment Plan

Audit input: `docs/experiments/rune-v09-v08-audit.md`.
Architecture: `docs/architecture/rune-v09.md`.

## Order (gated — each step can stop its track)

1. Multi-depth subset labeling (low/med/high/ref) → convergence
   report first. No convergence where it matters outranks everything.
2. Deterministic-teacher consistency checks (same position, same
   config → same target); infra bugs fixed before any learning claim.
3. Validity + stability buckets; cheap unreliability detectors.
4. Target-form ablations at fixed architecture AND fixed positions:
   T0 current vs T1 stability-filtered vs T2 quality-weighted vs
   T3 cascade/multi-condition (§47–§48).
5. Soft-vs-hard WDL; strict/soft/tie ranking with benchmarked
   margins (§26–§28).
6. Cascade budget experiments T0–T3 on equal teacher compute (§49).
7. Student legs (25/50/100M rapid, 250M candidates) + search-
   relevant validation (deltas, ranking preservation, tactics,
   NPS/nodes/latency/match) (§46, §51).
8. Depth-vs-improvement diminishing returns → minimum useful
   teacher effort (§52–§54).

## Matrices (controlled subsets, never full Cartesian)

```
T0 current / T1 stability-filtered / T2 quality-weighted / T3 cascade
L0 value+WDL / L1 +rank / L2 quality-weighted
S0 random-stratified / S1 v0.8-active / S2 active+quality
```

## Metrics (no composite score)

Teacher stability distributions, bucket rates, agreement rates,
student validation + calibration, three-way splits, per-level
teacher budgets, total-compute-to-quality curves, search behavior.
Small-match differences reported with uncertainty, never as
conclusions.

## Stop rules

- Stability filtering eats tactical knowledge → reject the level.
- Quality weighting ≈ uniform → remove the complexity.
- Cascade overhead > saved reference compute → all-expensive wins.
- Deeper teacher adds nothing the student absorbs → stop deepening.
- Quantization/compression breaks labels, calibration, or ranking
  → not deployment-ready.
