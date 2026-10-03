# RUNE v0.5 Results (running log)

Status: audit + scaffolding complete, no trained runs yet. Nothing
here is a conclusion until 25M+ comparisons with identical
seed/data/positions exist, plus calibration and CPU benches.

## Completed (code-level, no strength claims)

- v0.4 audit: `docs/experiments/rune-v05-v04-audit.md`
  (KEEP / REMOVE / REWORK / UNKNOWN).
- Architecture + plan: `docs/architecture/rune-v05.md`,
  `docs/experiments/rune-v05-plan.md`.
- Uncertainty model (Python, C++ parity pending bindings build):
  `training/models/uncertainty.py` — bounded head on shared
  representation, stability helpers, multi-signal routing modes.
- Loss ladder: L0/L1/L2 in `training/losses/uncertainty.py`
  (λ-gated, ablated one at a time).
- Analysis: `tools/analysis/calibration.py` (curves, buckets,
  rank corr, stratification), `tools/analysis/stability.py`
  (parent-child spread, flip/spike detection, sharpness battery).
- Configs: A0/A1/A2/A3, R-modes, L0/L1/L2 under `configs/v05/`.
- C++ uncertainty mirror + bench split + tests: full `rune_tests`
  build passes (4 new tests: bounds, routing modes, RUNE-05 IO
  roundtrip, param accounting); stage bench reports unc/stab
  forward, multi-signal routing, and full search-eval splits
  (see profiling log for first noisy-machine numbers).
- Teacher pipeline unblocked at smoke scale: torch CPU installed,
  synth teacher (`synth_mlp_v1`) labels a 30-position curated pool
  end-to-end (clean 30/30, split 20/5/5); one smoke screening runs
  all four arch families train → export → metrics; oracle,
  calibration, and representation tools execute on real outputs.
  Pure-torch pytest: 41 passed, 11 skipped (bindings-only skips).
- Failed-experiment home: `docs/experiments/failed/` (§36).

## Pending (in plan order)

1. Teacher labels (repo-wide blocker since v0.2) → L1 25M.
2. Calibration verdict → routing legs R0–R3.
3. A1/A2/A3, stability track, INT8 calibration bench.
4. Sharpness + search-stability, 250M, controlled match.

## Interpretation rules

Calibration before routing. Danger misses reported separately, never
averaged away. Tactical flattening is a kill condition. Promotion
human, per-metric, never a single score.

## Failed experiments / fixes

- Trainer `evaluate` KeyError on adaptive/search loss keys (tot only
  knew value/wdl/rank; AdaptiveLoss/SearchLoss return cheap_*/ref_*
  keys). Never ran before — caught by the first smoke screening.
  Fixed by widening tot and guarding missing keys.
- `representation.py` batch builder crashed on variable feature
  counts (`torch.cat` along mismatched dims). Fixed with per-group
  max-len padding mirroring `RuneDataset.collate`.
- `test_search_loss_gating` passed stability args positionally into
  rank/oracle slots. Fixed with keywords; code was correct.
- (log here AND mirror to `docs/experiments/failed/` per §36)
