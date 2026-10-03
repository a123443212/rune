# RUNE v0.3 Experiment Plan

Audit input: `docs/experiments/rune-v03-v02-audit.md`.
Architecture: `docs/architecture/rune-v03.md`.

## Research questions

- RQ1: can semantic tokens carry more information without
  meaningfully more inference cost? (A0 vs A1, then A2)
- RQ2: can redundancy between tokens be reduced? (redundancy report
  + P0 vs P1)
- RQ3: can freed capacity be replaced with useful capacity?
  (H1 allocation + pooling, analysis re-derived)
- RQ4: can equal-or-better accuracy/strength come from a smaller or
  faster model? (budget ladder 128/192/256/320 + efficiency ratios)

## Order

1. Representation analysis on A0 (random-init first for pipeline
   validation, then trained 25M): `tools/analysis/representation.py`.
   Find where capacity is wasted before spending training budget.
2. A0 vs A1 at 25M. Reallocation isolated, nothing else moves.
3. Winner of (2) vs A2 at 25M/50M. Pooling isolated.
4. A2 vs A3 at 25M/50M. Gate isolated, kill-on-no-gain.
5. P0 vs P1 at 25M/50M. Shared projection isolated.
6. Budget ladder 128/192/256/320 on the simplest winning form.
7. Mixer check: best dense form vs `rune_rel_s` 8x32 at 25M/50M.
8. Only promoted candidates: 250M confirmation, per-token quant
   sensitivity, quiet-machine benchmark, engine match with CI.

Each experiment states hypothesis, controlled variable, fixed
variables, metric, and expected interpretation before running. No
brute force, no NAS, no hundred-variant sweeps.

## Mandatory ablation ladder (§23)

```
A0  v0.2 representation (a0_baseline.yaml)
A1  reallocated dims, nothing else (a1_realloc.yaml)
A2  A1 + token pooling (a2_pool.yaml)
A3  A2 + lightweight gate (a3_gate.yaml)

P0  per-token projection (p0_pertoken.yaml)
P1  shared projection (p1_shared.yaml)
```

Total budget controlled in every leg. 500M/1B runs do not exist in
v0.3 except as a template for a promoted, confirmed candidate.

## Metrics per leg (no composite score)

Validation (value/WDL loss, WDL acc, rank acc), params total and
trainable, param bytes, activation bytes, per-stage latency split
(feature / accumulator / pool / gate / mixer / head), NPS, model size,
INT8 effect per token, match strength only with uncertainty. Plus
efficiency ratios reported separately: strength/param,
strength/byte, strength/cost.

## Learning-curve discipline (§21)

Rapid 25M/50M/100M for all legs. 250M for promoted candidates only.
Never train every variant to 1B. Data pipeline frozen during arch
legs; the only data question allowed is whether a better
representation needs less data, tested after the arch winner exists.

## Stop rules

- Gate fails twice at matched conditions: removed, not optional.
- Shared fails to match per-token: stays experimental, needs a
  param-matched control before any claim.
- 320 wins by capacity alone: recorded as capacity, not architecture.
- Mixer check ties: representation wins, mixer leaves the core.
