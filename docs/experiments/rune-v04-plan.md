# RUNE v0.4 Experiment Plan

Audit input: `docs/experiments/rune-v04-v03-audit.md`.
Architecture: `docs/architecture/rune-v04.md`.

## Order (gated — each step can stop the track)

1. Oracle routing (§11). Offline, teacher/student error decides who
   needs refinement. Establishes the theoretical ceiling. No ceiling
   → no learned-routing investment.
2. A0 vs A1 at 25M. Cheap-path standalone quality. A weak cheap path
   poisons everything downstream.
3. A1 vs A2 (always-refine) at 25M/50M. Refinement gain isolated
   from routing savings (§10).
4. A2 vs A3 adaptive at 25/50/75/100% refinement budgets, 25M/50M.
   Routing savings isolated.
5. R0 oracle vs R1 learned at matched refinement rates. Difficulty
   estimator accuracy, plus FP/FN breakdown by phase/material/
   tactics/king-danger/queen/endgame/boundary (§13).
6. P0 vs P1 vs P2 on trained weights, real target CPU.
7. I0 vs I1 with retrain + tolerance check; static patterns vs
   dynamic selection.
8. Search-stability analysis: single-move pairs, routing-flip rate,
   EASY/HARD oscillation check. Hysteresis only if instability is
   measured.
9. Promoted candidates only: 250M, quiet-machine bench, worst-case
   bench, controlled engine match (§28).

## Mandatory matrices (controlled legs, never full Cartesian)

```
A0  v0.3 full (a0_full.yaml — RUNE-03-A until ladder promotes)
A1  cheap only (a1_cheap.yaml)
A2  cheap + refinement always on (a2_refine_always.yaml)
A3  cheap + refinement adaptive (a3_adaptive.yaml, threshold sweep)

R0  oracle routing (offline analysis, never in engine)
R1  learned difficulty routing

P0  uniform precision / P1 selective refinement / P2 critical-ops

I0  full 8x8 interactions / I1 pruned (retrain mandatory)
```

## Metrics per leg (no composite score)

Quality (value/WDL loss, WDL acc, rank acc), refinement rate,
avg latency, worst-case latency, cheap/refine/route latency splits,
params + bytes (base / refinement / routing separately), NPS, nodes
searched, match result with uncertainty. Central metric:
strength per average compute at fixed time/hardware/threads/hash.

## Training discipline

Loss stays `L_value + λ1 L_WDL + λ2 L_rank`. `L_compute` only with
a with/without experiment stating its purpose. Distillation
(cheap ← full) allowed with teacher cost reported, never called
free. Refinement training compares independent vs distilled-cheap vs
joint, keeps complexity only on measured benefit. 25/50/100M rapid,
250M candidates only, microbenchmarks before large training spend.

## Stop rules

- Oracle shows no trade-off → adaptive track stops, document why.
- A2 ties A1 → refinement leaves; A1 becomes the efficiency story.
- A3 matches quality but real-CPU avg cost does not win (branch/cache
  effects) → dynamic routing stays experimental, static patterns only.
- Hard→cheap miss rate unacceptable on tactics/king-danger →
  thresholds move or track stops; danger misses are never averaged away.
- Selective precision or pruning without notable gain → removed.
