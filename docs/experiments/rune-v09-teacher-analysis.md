# RUNE v0.9 Teacher Analysis (running log)

First empirical output of v0.9: how stable, consistent, and
phase-dependent teacher labels are — before any student trains
on them. A non-converging teacher outranks architecture work.

## Multi-depth convergence

- 150-position subset, Stockfish 17.1 deterministic mode
  (Threads=1, Hash=16, Clear Hash per position), depths 6/10/14:
  WDL flip rate **6.7%** across levels; cascade simulation
  (accept ≤0.1 delta at low) accepts **139/150**, escalates 11,
  residual unstable 6 → **77% cheaper than all-high** at
  illustrative per-level costs (0.02/0.05/0.05s).
- Reading: most positions converge early; a small unstable tail
  drives escalation. Thresholds and costs are illustrative —
  production values need the T-leg measurements.

## Consistency (deterministic mode)

- 50 positions × depth 10, repeated: **without** Clear Hash,
  exact-cp match **2%**, mean delta **32cp** (hash-state carryover
  between analyses in one session). **With** per-position Clear
  Hash: **100% exact match, 0 delta, 0 WDL flips**.
- Verdict: discrepancy was infrastructure, proven by the fix.
  Deterministic labeling clears hash per position; the
  consistency tool gates any host before production labeling.

## Phase/material error profile

- (pending: where the teacher is stable, where it flips)

## Calibration of confidence proxies

- (pending: convergence/agreement/WDL-consistency vs actual
  student error, by subset)
