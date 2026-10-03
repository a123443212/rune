# RUNE v0.5 Experiment Plan

Audit input: `docs/experiments/rune-v05-v04-audit.md`.
Architecture: `docs/architecture/rune-v05.md`.

## Order (gated — each step can stop its track)

1. Uncertainty calibration (L1 at 25M): does `u` predict teacher
   error (rank correlation, error buckets, calibration curve)? No
   relation → uncertainty leaves routing and possibly the core.
2. A0 vs A1 at 25M/50M. Prediction gain isolated (value error
   must not regress; tactical sharpness checked, §22).
3. R0 vs R1 vs R2 at matched refinement rates. Routing gain
   isolated from prediction gain. Danger-miss rate is the priority
   metric.
4. A2 (uncertainty routing) vs v0.4 difficulty routing + oracle.
   Three-curve comparison: theoretical ceiling vs deployable (§34).
5. Stability track: child-spread statistics → A3 only if spread
   carries search-relevant information beyond value. R3 last.
6. Quantization: INT8 uncertainty bench (numerical + calibration).
7. Tactical-sharpness battery + search-stability analysis
   (flips, spikes, oscillation; legitimate sharpness vs instability).
8. Promoted candidates only: 250M, quiet-machine bench, controlled
   match on search-aware subsets.

## Mandatory matrices (controlled legs, never full Cartesian)

```
A0  v0.4 (difficulty routing)
A1  + uncertainty head (prediction)
A2  + uncertainty-aware routing
A3  + stability signal

R0  difficulty / R1 uncertainty / R2 both / R3 + stability

L0  Value+WDL / L1 +Uncertainty / L2 +Stability
```

## Metrics per leg (no composite score)

Value/WDL error, rank acc, uncertainty calibration (curve, buckets,
rank corr), error stratification (low/med/high buckets), routing
accuracy + danger-miss rate, refinement rate, avg/worst latency,
params/bytes (aux heads reported separately), NPS, nodes, match
result with uncertainty. Subsets: quiet, tactical, king attack,
material imbalance, endgame, queenless, high/low mobility —
labeled by explicit rules, never intuition.

## Stop rules

- Uncertainty uncorrelated with teacher error → out of routing;
  core membership re-argued from scratch.
- Tactical flattening (reduced swing separation, suppressed king
  attacks) → offending objective removed or down-weighted, re-tested.
- Quantization breaks calibration → head not deployment-ready.
- Routing wins microbench but loses real CPU / engine behavior →
  removed (same rule as v0.4 §31).
- No search-behavior improvement at negligible cost → signals stay
  analysis-only, core stays v0.4.
