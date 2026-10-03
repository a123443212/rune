# RUNE v0.6 Profiling

Target: fewer params must mean less time and less memory on real
hardware — layout and kernels decide, not parameter counts.

## Required splits (ns/eval or cycles/eval, quiet machine)

```
per student budget S1–S4: full eval latency (refresh + incremental)
teacher labeling throughput (positions/sec, per-precision)
INT8 vs FP32 student eval (error, latency, NPS, bytes together)
hot activation memory + cache footprint per budget
packed-INT8 / fused-op variants only where profiled useful (§30)
engine walk NPS: teacher engine vs student engine
```

## Reporting

Per-metric only: quality, size, latency, NPS, memory, training
cost (teacher + labeling + storage + student, separately), search
behavior. Plus the compression frontier plots (size-vs-quality,
cost-vs-quality). No composite score.

## Log

- 2026-10-03, smoke scale (30 curated positions, random-init
  RUNE-05 teacher, pipeline validation only — no quality claims):
  teacher labeling 101–107 pos/sec with cost sidecar
  (teacher id/hash/positions/sec); S2 dense student distills
  end-to-end (train → export → metrics) with three-way reporting
  (student-vs-teacher MAE 0.087, teacher-vs-target MAE 0.130 on
  random weights, as expected divergent); student `.rune` carries
  token_dims sum 128, heads 64/16, 16-char checksum.
  Student params (dense, incl. embeddings): S1 72028 (~69%),
  S2 43412 (~41%), S3 30832 (~29%), S4 19404 (~18%) vs 105252
  teacher-scale. C++ `rune_tests` all pass incl. student-width
  roundtrips; pure-torch pytest 48 passed, 11 skipped
  (bindings-only skips — no Python headers on this host).
