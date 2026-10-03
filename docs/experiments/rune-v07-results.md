# RUNE v0.7 Results (running log)

Status: audit + engine + bridge + parity complete on pools up to
~25k positions. Scale validation to 100M+ and trainer-starvation
proof are scheduled, not claimed.

## Completed (measured, this host)

- v0.6 audit: `docs/experiments/rune-v07-v06-audit.md`
  (KEEP / REMOVE / REWORK / MOVE TO RUST / UNKNOWN) with a
  measured P0 baseline (extract 4.4k/s, clean 47k/s, epoch
  collate ~640/s, labeling ~150/s, S2 consume ~430/s).
- `rune-data` CLI (13 subcommands) + `rune-data-core` lib, zero
  warnings; 12 unit + 4 integration tests green.
- Exact Python parity: features 4302/4302, final sets 4288==4288
  FEN-equal stage by stage, torch batches bit-identical.
- Determinism: identical manifest hashes across 1/2/4/8 threads.
- Release throughput: extract 88.8k/s (20x), ingest 78–265k/s
  (1–8 threads, 4.8x sublinear), dedup 1.4–1.7M/s, raw write
  678–756k/s @ ~408 B/pos, raw read 888–980k/s, deflate 87–100k/s
  write @ ~51 B/pos (8x smaller) with 861–887k/s reads (decode
  ≈ free), external dedup 300–383k/s, mmap ≈ buffered (~0.8–1.0M/s
  warm cache — no winner declared).
- Python bridge (`tools/data_bridge/reader.py`) + pytest
  (`tests/test_data_bridge.py`, 4 tests incl. tamper rejection).
- Failure log: EP-range bug, throughput-metric bug, width-
  truncation in a test, PGN fixture bugs — all fixed with tests.

## Pending (in plan order)

1. 100M+ scale validation (external-dedup crossover, shuffle
   randomness quality, cold-cache storage comparison).
2. Trainer-starvation proof: Rust bridge → real training run,
   active%/wait-time metrics (§39).
3. pyo3 contiguous-buffer bridge if epoch delivery binds (§24–§25).
4. Quiet-machine publishable numbers for the frontier table.
5. First T0-scale dataset build (25M+) through the engine.

## Interpretation rules

Microbenchmarks never promote alone; end-to-end trainer
throughput without starvation is the only promotion test.
Rejected levels, corrupt inputs, and mismatch counts are results.
Same discipline as ever: per-metric, human promotion, no scores.
