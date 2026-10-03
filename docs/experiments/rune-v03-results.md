# RUNE v0.3 Results (running log)

Status: scaffolding complete, no trained runs yet. This file grows
per milestone. Nothing here is a conclusion until 25M+ comparisons
with identical seed/data/positions exist.

## Completed (code-level, no strength claims)

- v0.2 audit: `docs/experiments/rune-v03-v02-audit.md`
  (KEEP / REMOVE / REWORK / UNKNOWN).
- Dense paths C++ + Python + bindings parity (1e-4) for A/B/C/D,
  incremental == refresh over 60 random moves + 30 unmakes,
  export checksum + tamper rejection, per-tensor IO tests.
- Stage bench dense section: full refresh/incremental, pool, gate,
  model forward splits plus param/dim totals.
- Representation tooling: `tools/analysis/representation.py`
  (token-token cosine, channel correlation, effective rank, entropy,
  dead ratio, phase splits, position-pair sensitivity).
- Configs: A0/A1/A2/A3, P0/P1, budget 128/192/320, mixer check.
- Screening runner records `architecture_version 0.3.0` for RUNE-03
  and passes token_dims/pooling/gate through.

## Pending (in plan order)

1. Random-init representation report (pipeline validation only).
2. A0 25M baseline + analysis.
3. A1/A2/A3/P0/P1 legs at 25M/50M/100M.
4. Budget ladder + efficiency ratios.
5. Mixer check, 250M confirmation, per-token quant, quiet-machine
   bench, engine matches.

## Interpretation rules

One axis varies per comparison; fixed axes recorded in every
metrics.json. Promotion is human, per-metric, never a single score.
Small match differences reported with uncertainty, never as
conclusions. Capacity wins are labeled capacity, not architecture.

## Failed experiments / fixes

- Screening wrote `architecture_version 0.1.0` for RUNE-03 models.
  Fixed to `0.3.0` (this cycle, before any 25M run consumed it).
- Dense presets shipped uniform [32]*8 for all variants, so no
  allocation experiment was expressible. Fixed with `ALLOCATION_H1`
  / `BUDGET_ALLOCS` plus per-leg configs.
