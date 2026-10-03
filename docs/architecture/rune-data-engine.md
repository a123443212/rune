# RUNE Data Engine (v0.7)

Division of labor, enforced, not aspirational:

```
Rust        → Data Engine (parse, validate, dedup, filter, shard, serve batches)
Python/Torch → Training (consumes batches, owns loss/optimizer)
C++         → Chess runtime / inference (owns the evaluator hot path)
```

Rust never touches the evaluator, search, or NNUE path in v0.7.
Python never re-extracts features per epoch in the v0.7 path.
C++ and Rust meet only through dataset/model artifacts.

## Pipeline

```
PGN / JSONL pool
  ↓ ingest (streaming, bounded memory, 1..N deterministic threads)
validation (king counts, back-rank pawns, value/WDL ranges — same rules as Python)
  ↓
feature/metadata extraction (exact port of grouped_hkav2_fullthreats_v01)
  ↓
canonical identity: FNV-1a over normalized FEN (strict mode available)
  ↓
dedup: in-memory HashSet fast path + partitioned external path (sort-free,
      hash-partitioned spill files, deterministic re-merge by sort)
  ↓
filter (composable: teacher presence, value range, phases, imbalance, min ply)
  ↓
phase balance (same cap rule as Python) → sampling (uniform/stratified) →
game-aware split (hash-bucketed game ids, leakage-safe)
  ↓
sharding (identity-sorted, fixed per-shard count, manifest + per-shard checksums)
  ↓
compressed .rune-data → Python bridge (pre-extracted features, torch batches)
```

Every stage runs standalone via `rune-data <stage>`. Deterministic
mode: same input + config + seed = same bytes (manifest hash covers it).

## Why Rust wins here (measured, this host)

| Stage | Python (P0) | Rust release | Ratio |
| ----- | ----------- | ------------ | ----- |
| feature extraction | 4,434 pos/s | 88,844 pos/s | ~20x |
| pool ingest (parse+extract) | — | 78,508–104,300 pos/s | — |
| per-epoch collate | ~640 pos/s | bridge ~726 pos/s (Python-loop-bound; pyo3 is the follow-up) | ~1x |
| dedup (in-mem) | inside 47k/s clean | 1.4–1.7M pos/s | ~30x+ |
| shard write raw / read | — | 678k / 888k–970k pos/s | — |
| deflate write / read | — | 87–100k pos/s @ 51 B/pos (8x smaller) / 861–887k pos/s | decode ≈ free |
| PGN ingest 8 threads | — | 265k pos/s (55k 1-thread → 4.8x, sublinear, honest) | — |
| teacher labeling (Stockfish d10) | 146 pos/s | n/a (engine-bound, unchanged) | — |

mmap vs buffered reads are tied at shard scale (~0.8–1.0M pos/s
warm cache); no winner declared — large-shard/cold-cache comparison
is future work, and the mmap path ships for that experiment.

## Non-goals (parked, not forgotten)

pyo3 zero-copy bridge, external-shuffle randomness study,
near-duplicate/fuzzy analysis beyond statistics, full NUMA tuning,
QAT data paths. Each has a measured reason to exist later and a
cold, explicit NOT NOW in `docs/future-work.md`.
