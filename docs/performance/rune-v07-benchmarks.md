# RUNE v0.7 Benchmarks

Host: shared Linux box, 8 torch threads visible. Noisy-machine
numbers are RATIOS and shapes, never absolutes. Quiet-machine
re-runs required before any publishable claim.

## Pipeline matrix (§55)

P0 Python baseline (4465-pos real pool):
`extract 4,434 | clean 46,897 | epoch-collate ~640 pos/s`

P1–P4 Rust release (same pool unless noted):
`extract 88,844 | ingest(pool) 78,508–104,300 | dedup 1.4–1.7M |
full-chain end-to-end verified FEN-equal to Python`

25k-position PGN ingest, threads 1/2/4/8:
`55,410 | 102,742 | 169,964 | 265,429 pos/s` (4.8x, sublinear —
threading helps parse+extract, merging stays serial; deterministic
manifest hash identical all four runs).

## Storage matrix (§46)

```
S0 raw:     407–409 B/pos, write 678–756k/s, read 888–980k/s
S1 deflate: ~51 B/pos (8x smaller), write 87–100k/s, read 861–887k/s
S2 packed:  not implemented — S1 reads already saturate; packed only
            if a future bottleneck says so
```

Decode cost is ~zero at these sizes; write-side compression pays
8x disk for ~7x slower writes. Choice is workload-dependent, hence
both ship and the decision stays per-dataset.

## Read matrix (§10)

`R0 buffered 980k/s vs R1 mmap 801k/s vs R2 streaming structs
inherent in all commands` — warm cache, small shards: no winner.
R1 path ships so the large-shard/cold-cache experiment can run;
declaring mmap "better" now would be fabrication.

## Starvation (§38–§39)

Not yet measured against a live trainer — the honest gap. Bridge
epoch delivery (~726/s, Python-loop-bound) vs trainer consumption
(~430/s S2) suggests headroom at small scale, but the §39 metrics
(active%, wait time, batch latency at 25M+ with prefetch) are
scheduled work, explicitly not claimed here.

## Success metrics status (§56)

 Records/sec, GB/hour, RAM (RSS per report), bytes/pos: measured.
 CPU utilization, batch latency distribution, GPU starvation,
 end-to-end training throughput: scheduled with harness ready.
