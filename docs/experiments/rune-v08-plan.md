# RUNE v0.8 Experiment Plan

Audit input: `docs/experiments/rune-v08-v07-audit.md`.
Architecture: `docs/architecture/rune-v08.md`.

## Order (gated — each step can stop its track)

1. Seed model on random/stratified (existing smoke path at real
   scale) → seed checkpoint (cold-start requirement, §17).
2. Single-signal ablations A0–A5, one signal at a time, equal
   labels (B0/B1/B2 baselines first). Rarity/diversity costed.
3. Multi-signal A5 + diversity A3 only after a single signal shows
   life. Uniform-weight control always runs.
4. Coverage/collapse watch from round 1 (tactical/phase drift is
   a kill signal, not a tuning knob).
5. Replay/forgetting gates (T0/T1/T2 + probe MSE) before any
   multi-round claim.
6. Diminishing-returns curves → stopping analysis (§47–§49).
7. Sibling-group selection only where child data exists, with the
   individual-selection control beside it.
8. 250M+ only on measured signal; engine behavior last.

## Matrices (controlled subsets, never full Cartesian)

```
B0 random / B1 stratified / B2 disagreement
A0 uncertainty / A1 rarity / A2 multi / A3 multi+diversity
T0 scratch / T1 continue / T2 mixed+replay
```

## Metrics (no overall score, §55)

Quality vs labels, vs teacher compute, vs total compute, vs
rounds; gain/round, gain/1M labels; static + dataset +
infra + engine metric families (§54). Danger misses and
coverage drift reported separately, never averaged away.

## Stop rules

Marginal gain ≈ 0, budget/pool exhausted, diversity collapse,
or target hit — policy configurable, decision human. A strategy
is promoted only on: measured curve gain + equal-budget win +
sane teacher cost + intact coverage + sufficient throughput.
Static-loss wins on one round promote nothing (§61).
