# RUNE v0.7 — v0.6 Audit (evidence-gated, input to v0.7)

Scope: the entire data path from source positions to trainer
batches, measured against v0.7's question (more data, faster,
cheaper, reproducible — training compute, not preprocessing/I-O,
must be the limit at 500M–1B+ scale). Bench, parity, and smoke
runs measure cost and correctness, never quality. No strength
claims exist anywhere in this audit: there are still no 25M+
trained runs in the repo.

Current honest status: the Python data path is complete and
correct (clean/dedup/balance/split/sampling/teacher-labeling all
run end-to-end; full pytest 70/70 with bindings), but every stage
is single-threaded Python with per-position object overhead, no
binary dataset format exists (JSONL pools on disk), and scale has
never exceeded ~4.5k positions outside synthetic screening configs.

## 1. Measured P0 baseline (this host, 4465-pos real pool)

```
extract_features (Python, per position, one-time):  ~4,434 pos/s
clean_pipeline (integrity+dedup+quality+balance):  ~46,897 pos/s
epoch collate (RuneDataset, per epoch, repeated):  ~600-660 pos/s
teacher labeling (Stockfish d10, 1 thread):        ~146-153 pos/s
trainer consumption (S2 dense, 100k run):          ~430 pos/s
```

Reading: the per-epoch collate loop (~600 pos/s) is slower than
feature extraction itself and sits within 1.4x of trainer
consumption (~430 pos/s) — at larger models/widths the trainer
will outrun supply, and every epoch re-pays full Python scan
cost. JSONL pools are text: re-parsed per stage, no random
access, no checksums, no sharding, no mmap path.

## 2. What was inspected

- Code: `training/datasets/pipeline.py` (integrity, dedup by
  normalized FEN key, quality, phase balance, game-grouped
  split, random/stratified/disagreement samplers),
  `training/datasets/rune_dataset.py` (pre-extract + per-batch
  Python collate — the measured bottleneck),
  `training/datasets/siblings.py` (bindings-gated child
  enumeration), `training/samplers/disagreement.py`,
  `tools/dataset/` (build/label/siblings tooling, synth +
  Stockfish teachers, stm-relative convention),
  `training/export/export.py` (model artifacts only — no
  dataset format), screening record fields (dataset_hash,
  teacher_hash, sampling config).
- Data reality: `/tmp/rune-real` (4465 Stockfish-labeled real
  positions, ephemeral); no versioned dataset exists in-repo;
  `data/` absent; runs/ gitignored and ephemeral.

## 3. Verdicts

### KEEP (Python/training side, unchanged)

| Component | Evidence | v0.7 role |
| --------- | -------- | --------- |
| Pipeline semantics (integrity rules, normalized-key dedup, quality filter, phase balance, game-grouped split) | Correct, tested, smoke-validated | Normative spec: the Rust engine must reproduce these exact semantics, verified by cross-checks |
| stm-relative value convention + `value_perspective` stamping | Forced by real-teacher data, fixed across labelers + match harness | Recorded per-record in the binary format; join/migrate must reject mismatches |
| Teacher id/hash + dataset hash + sampling config in run records | Implemented in screening | Extended into dataset manifests (§31); training declares dataset version (§48) |
| Disagreement sampler logic | Pipeline-only, unmeasured value | Reference semantics for the Rust sampler; worth re-measuring only after throughput exists |
| Model-artifact export (`.rune`) | Roundtrip + tamper tests pass | Untouched: dataset/model artifacts stay separate (§59) |

### REMOVE (do not carry into the data engine)

| Component | Evidence | v0.7 role |
| --------- | -------- | --------- |
| Per-epoch Python feature-list rescanning | Measured ~600 pos/s, 7x slower than extraction | Replaced by pre-encoded binary records + contiguous batch buffers |
| JSONL text pools as a training input | Re-parsed per stage, no checksums/shards/random access | Replaced by versioned `.rune-data`; JSONL stays as interchange/debug only |
| In-RAM full-dataset assumptions (shuffle-all, hash-all) | Never tested past 4.5k; design assumes fit-in-RAM | Replaced by streaming + external/partitioned algorithms with bounded memory |
| Silent fallbacks on corrupt/empty data | Past bugs (empty splits, sys.path) needed guards | Explicit detect/report/fail; recovery mode counts skips (§29) |

### REWORK (same semantics, new implementation)

| Python form | Problem at scale | v0.7 form |
| ----------- | ---------------- | --------- |
| `extract_features` per position | 4.4k pos/s single-thread; attacks recomputed per call | Rust extractor (shakmaty movegen) + benchmark P1 vs P0; Python stays as cross-check oracle |
| `deduplicate` (in-RAM set of FEN strings) | Memory grows with dataset; string keys wasteful | Hash-based dedup with in-memory fast path + partitioned external path; Zobrist/canonical identity, strict vs normalized modes (§12–§14) |
| `phase_balance` / samplers (list slicing, full shuffle) | Full-RAM shuffle; no streaming | Deterministic reservoir/buffer/shard shuffles with randomness-quality benchmarks (§22–§23) |
| `split_by_game` (hash over game ids) | Fine logically, unmeasured at scale | Deterministic game/source sharding with manifests (§11, §17) |
| Teacher labels inline in JSONL records | Re-generated or duplicated per experiment; no join checks | Label store separable from positions + deterministic join with mismatch detection (§33–§34) |
| Manual `report.json` per screening run | No per-stage pipeline reports | `report.json` + `report.md` per pipeline run with reject reasons counted (§21) |

### MOVE TO RUST (the v0.7 build list, in pipeline order)

PGN parse → position extraction → validation → feature/metadata
extraction → canonical identity + hashing → dedup (in-mem +
external) → quality/teacher-availability filtering → phase
balance/statistics → deterministic sampling/shuffling → sharding
+ manifest + checksums → compressed `.rune-data` → Python/Torch
batch bridge with prefetch. Plus: `benchmark` matrix (P0–P4, S0–S2,
R0–R2, threads 1–16+), corruption handling, and the four docs.
Anything here without a measured bottleneck behind it stays
Python until profiling says otherwise (§1 rule).

### UNKNOWN (needs measurement, cheapest test first)

- Rust-vs-Python throughput ratios on real data (RQ1).
- Largest scale fitting bounded memory; external-dedup crossover
  point (RQ2, §14).
- Deterministic sharding + shuffle randomness quality at scale
  (RQ3, §23).
- Whether the trainer starves and at what batch/worker point
  (RQ4, §38–§39).
- Compression ratio vs decode-throughput tradeoff; mmap vs
  buffered vs streaming winner per storage type (RQ5, §9–§10).
- Worker scaling curve and NUMA relevance on the target host
  (§27–§28).
- Whether near-duplicates matter (statistics first, §15).

## 4. Consequence for v0.7

Build order is forced: (1) format v1 + streaming ingest with
determinism + checksums; (2) dedup/filter/stats/shard with
Python-parity cross-checks on every semantic; (3) Python bridge
+ prefetch; (4) benchmark matrix vs P0 numbers above; (5) docs.
No component earns "keep forever" status from microbenchmarks
alone — end-to-end trainer throughput without starvation is the
only promotion test (§39, §57). Anything unmeasured stays
explicitly UNKNOWN, including all five RQs at the time of writing.
