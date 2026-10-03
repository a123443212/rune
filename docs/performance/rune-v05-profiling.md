# RUNE v0.5 Profiling

Target: added intelligence at ~zero added compute. Theory plans,
hardware decides.

## Required splits (ns/eval or cycles/eval, quiet machine)

```
v0.4 baseline (cheap / refine / route, already logged in v0.4)
uncertainty head alone (dot + sigmoid)
multi-signal routing decision (R0 vs R1 vs R2 vs R3)
stability scalar handling (if A3 ships it to engine)
INT8 uncertainty path (conversion + drift check)
full v0.5 eval at refinement rates 10/25/50/75/100%
worst-case single eval (refinement + all signals, cold + warm)
```

## Calibration-under-quant protocol (§28)

Report FP32 vs INT8: numerical error on `u`, calibration-curve drift,
bucket-reassignment rate. A head that drifts loses deployment status
regardless of FP32 quality.

## Reporting

Per-metric only: aux-head latency, routing latency per R-mode,
avg/worst latency, refinement rate, NPS, nodes, match with
uncertainty. No composite score.

## Log

- 2026-10-03, same noisy host, random weights, `rune_stage_bench`
  search section (8x32): uncertainty+stability forwards 0.39 us vs
  cheap forward 7.8 us (~5% added); multi-signal routing compare
  ~0.002 us; full search eval from refresh 57.1 us (includes
  ~13.9 us refresh; refinement taken on random weights).
  Reading: auxiliary signals cost ~nothing next to heads; routing
  stays branch-predictable single compares. Quality/calibration
  verdicts need trained weights — cost side alone never promotes.
- (quiet-machine numbers pending; v0.4 baseline numbers live in
  `docs/performance/rune-v04-profiling.md`)
