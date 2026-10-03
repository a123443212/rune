# RUNE v0.8 Results (running log)

Status: audit + loop + selection + analysis complete; mechanics
validated at small scale with stand-in models. No learning claim
exists until real-model rounds run at equal budgets.

## Completed (measured)

- v0.7 audit: `docs/experiments/rune-v08-v07-audit.md`
  (KEEP / REMOVE / REWORK / EXTEND / UNKNOWN).
- Loop: `tools/active/` (score_candidates, rounds orchestrator
  with seed/continue/mixed + replay, coverage, retention) wired
  to screening (`init_ckpt` + cumulative positions), match, and
  calibration tooling.
- Selection: Rust `score`/`select` (bounded heaps, two-level
  merge, diversity caps + refill, stratified floor, reuse split,
  audit sidecar, `round.yaml`); schema v2 (round/method/score,
  v1 still reads, v99 rejected); tie-breaks by identity hash.
- Measured: two-level EXACT at K′≥shard, 65–70% at K′=400–1000;
  select determinism (identical manifest hash, identical audit
  order); single/multi-shard paths agree by construction.
- Full round-0 smoke (seed synth model → uncertainty scoring →
  select 200 → stockfish labeling → mixed train → coverage +
  retention): mechanics end-to-end, artifacts complete
  (round.yaml, audit, coverage, retention probe table).
- Rust: 15 unit + 6 integration green, clippy/fmt clean.
  Python: 74 pytest green (bridges need built bindings).
- Failure log: teacher key-set split (unified with legacy
  aliases), batch-shape collision (len-6 explicit-None form),
  distill metric prefix mismatch, YAML-vs-JSON config load,
  game-less round pools (game_hash carried in audit),
  coverage reader on binary shards, PGN fixture bugs.

## Pending (in plan order)

1. Seed model at real scale → scored candidates with a real
   student (not a stand-in).
2. A0–A5/B0–B2 equal-label legs; coverage/collapse watch.
3. Replay/forgetting gates; diminishing-returns curves.
4. Sibling-group legs where child data exists.
5. 250M+, engine behavior, promotion decision.

## Interpretation rules

Stand-in runs prove mechanics only and are labeled as such.
Learning claims need real models, equal budgets, and total-cost
curves. Same discipline as ever.
