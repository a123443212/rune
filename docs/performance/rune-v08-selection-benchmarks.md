# RUNE v0.8 Selection Benchmarks

Host: shared Linux box. Noisy numbers are shapes, never absolutes.

## Selection matrix (4.5k candidates, 3 shards, budget 500)

```
global single-pass:      baseline (exact by construction)
two-level K′=400:        65–70% agreement with global
two-level K′=1000:       ~70% agreement with global
two-level K′≥shard:      EXACT (identical order + manifest hash)
select determinism:      identical hash + audit order across runs
```

Reading: K′ is a real approximation knob with a measured curve,
converging to exact — not a hope. Diversity caps + stratified
floor apply once, globally, after the merge (applying them per
shard made agreement non-monotonic in K′ — found, fixed, tested).

## Throughput (this host, release)

```
score (join + rarity passes):  not separately timed — shares the
                               from-jsonl path (~10k/s scale); dedicated
                               score/select split timing is scheduled
select (heap + caps + floor):  dominated by shard reads (~900k/s)
audit sidecar + round.yaml:    negligible vs selection
```

## Infrastructure metrics status (§54)

Candidate records/sec, selection records/sec, memory (RSS per
report), I/O: measured. Teacher lookup throughput: measured at
label time (Stockfish path, ~150/s single-thread). Batch delivery
latency distribution, GPU starvation, end-to-end training
throughput with prefetch: scheduled with harness ready — the same
honest gap as v0.7, narrowed but not closed.

## Starvation verdict

Not measured against a live trainer. Unchanged rule: no promotion
on microbenchmarks.
