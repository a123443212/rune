# RUNE v0.9 Results (running log)

Status: audit + target infrastructure complete; first multi-depth
measurements pending. Nothing here is a conclusion until
matched-budget, matched-teacher comparisons run.

## Completed (measured)

- v0.8 audit: `docs/experiments/rune-v09-v08-audit.md`
  (KEEP / REMOVE / REWORK / IMPROVE / UNKNOWN).
- Architecture + pipeline + plan: `docs/architecture/rune-v09.md`,
  `docs/data-pipeline/rune-v09-targets.md`,
  `docs/experiments/rune-v09-plan.md`.
- Teacher tooling: multi-depth labeling (`label_engine.py`
  `--depths`), deterministic mode + consistency check
  (`tools/dataset/check_teacher.py`), full provenance records.
- Target schema v2: validity states, multi-depth fields, soft
  WDL, strict/soft/tie ranking pairs, quality metadata
  (`training/datasets/targets.py`, loss support, masked missing).
- Rust: `verify-targets`, `target-stats` (+ noise triage),
  cascade budget scheduler stub with measured per-level costs,
  join/lookup paths extended (stale/mismatch detection).
- Configs: T0–T3, L0–L2, S0–S2 under `configs/v09/`.
- Tests green (Rust + pytest); failed/v09/ open.
- First measurements: 150-pos multi-depth set (d6/d10/d14,
  deterministic) → 6.7% WDL flips, cascade 139 accept / 11
  escalate / 77% cheaper (illustrative costs); consistency
  2%→100% exact match via Clear Hash; soft-WDL + quality-weight
  + tie-margin training path smoke-tested (20 steps, nonzero
  distill terms).

## Pending (in plan order)

1. Multi-depth subset measurements → convergence report.
2. Consistency verification on the labeling host.
3. T1/T2/T3 ablations at fixed architecture/positions.
4. Soft WDL + near-tie ranking legs.
5. Cascade budget experiments on equal teacher compute.
6. Student legs, search validation, diminishing returns.

## Interpretation rules

Convergence first, claims later. Capacity wins labeled capacity.
Danger misses and unstable buckets reported separately, never
averaged away. Promotion human, per-metric, never a single score.
