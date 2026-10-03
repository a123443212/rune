# RUNE Data Pipeline: Active Learning (v0.8)

How data flows through one closed-loop round, and which language
owns each step (§56).

## Round dataflow

```
candidate shards (.rune-data, any schema v1/v2)
  │ Python: score_candidates.py (torch inference, needs a real model;
  │         synth/random-init checkpoints are mechanics-only stand-ins)
  │         → scores.jsonl {identity|fen, disagreement, uncertainty,
  │            instability, rank_disagreement}
  ▼
Rust: score (identity join, mismatch counted; rarity from two
      metadata passes; normalized weighted final; components sidecar)
  │──▶ score_components.jsonl (audit input, per-sample components)
  ▼
Rust: select (per-shard heaps → global merge → diversity caps →
      stratified floor → requested/reused split)
  │──▶ selected shards (schema v2: round, method, score)
  │──▶ selection_audit.jsonl (per-sample: id, method, components,
  │      final, round, teacher/student versions)
  │──▶ round.yaml (§59: round, dataset, model, selection, teacher,
  │      system — git commit included)
  ▼
Python: label_engine.py on requested FENs only (reused skip the
        teacher entirely); staleness = teacher-id mismatch
  ▼
materialize round pool (mixed/replay per config, immutable file)
  ▼
screening train (scratch / init_ckpt continue / mixed) → checkpoint
  ▼
coverage.py (pool vs dataset vs batch) + retention.py (probe MSE
across checkpoints) → next round rescores with the new model
```

## Division of labor (measured reasons only)

- Rust: streaming, heaps, joins, caps, manifests, hashes. It never
  sees a neural network.
- Python: inference, training, curves, calibration, match analysis.
- C++: final evaluator + engine integration + inference bench.

## Scale notes (honest)

Validated to ~25k positions end-to-end. Memory bounds: one shard
+ K′×shards winners + K heap + scores map for the join step
(the scores join is the loosest bound — documented; sharded
sidecars are the escape hatch, not yet needed). Billion-scale
claims require the §31–§33 validation runs, which do not exist yet.
Two-level approximation is measured (65–97% agreement sweeping
K′), never assumed.
