# RUNE Data Pipeline (v0.7)

Every stage is independently runnable, streaming-first, and
bounded-memory. Python semantics are the normative spec; Rust
proves equivalence by cross-check, not by assertion.

## Stage order and commands

```
from-jsonl / ingest (PGN)          parse + extract + validate
  → dedup [--external --partitions]  hash dedup (in-mem or partitioned spill)
  → filter [--require-teacher --min-abs-value --phases --max-imbalance --min-ply]
  → balance [--max-ratio]            phase-cap rule, same as Python
  → sample [--n --mode uniform|stratified --seed]
  → split [--train-frac --val-frac]  game-hash buckets, leakage-safe + fallback
  → shard [--per-shard --compression]  identity-sorted, manifest + checksums
```

Plus: `validate` (re-checks any artifact), `stats` (JSON counts,
phase/value histograms, dup rates, teacher coverage),
`inspect` (human dump incl. `--features`), `join-labels`
(deterministic identity join with missing/duplicate/mismatch
counts — never silent), `export-labels` (teacher sidecar for reuse),
`benchmark` (extract/pool/dedup/write/read/external/mmap splits).

## Semantics locked to Python (verified, not claimed)

- Final record set of the full Rust chain == Python
  `clean_pipeline` output on the 4465-pos pool: **4288 == 4288,
  FEN sets exactly equal**, stage counts identical at every step
  (53 dups, 124 min-ply drops).
- Feature vectors: **4302/4302 exactly equal**; torch batches
  **bit-identical** to `RuneDataset` collate on 512 shared records.
- Game split keeps the hash-bucket + fallback semantics; sampling
  uses an explicit splitmix-style RNG (seeded, no hidden randomness).
- Caught and fixed along the way: EP file range excluded 'a'
  (Rust rejected 3 valid FENs); throughput metric divided by output
  instead of input; Python batch-test widths truncated group 6.

## Determinism

Same input + config + seed = same manifest hash, verified across
1/2/4/8 ingest threads (`04bfd4e…` four times). No HashMap iteration
in output paths (sorts where order matters), no locale/time
dependence, FNV-1a everywhere for identity, explicit seeded RNG
only. External dedup re-merges by deterministic sort.

## Memory and scale posture

Ingest streams games; dedup external spills by hash partition;
shards bound per-file size; the reader never loads more than one
shard + one batch. Current validated scale is ~25k positions;
100M+ validation (external-dedup crossover, shuffle quality,
cold-cache mmap) is scheduled work with the harness already in place
(`benchmark`, partitioned dedup, per-shard caps).

## Teacher labels (§33–§35)

Labels ride inline (`has_teacher` flag) or join deterministically
from JSONL sidecars with mismatch accounting. Disagreement metadata
(level for sampling experiments) is produced by the existing Python
tooling on top of exported labels — no model training inside Rust.
Re-generating teacher outputs is never hidden: labeling throughput
and engine costs stay in run metas, and manifests pin which labels
a dataset carries.
