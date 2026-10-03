# RUNE v0.8 — v0.7 Audit (evidence-gated, input to v0.8)

Scope: everything v0.7 built, measured against v0.8's question
(train → find weakness → label useful data → train again, spending
teacher compute where it buys the most knowledge). A component is
PROVEN only with a matched-budget comparison. Bench, parity, and
smoke runs measure cost and correctness, never learning value.

Current honest status: **the Data Engine moves, validates, dedups,
and serves data with proven parity and throughput, but it has never
selected anything.** No candidate scoring, no top-K, no diversity
filter, no coverage guard, no label-store lookup, no round
versioning, and no active round has ever run. Disagreement sampling
exists only as a Python pipeline function with zero measured
learning value. Every RQ1–RQ6 answer is UNKNOWN.

## 1. What was inspected

- Rust: `rune-data-core` (board/features exact port, FNV-1a
  identity strict/normalized, format v1 with checksums + offsets,
  in-memory + partitioned-external dedup, composable filters,
  phase balance, uniform/stratified samplers, hash-bucketed
  game split, identity-sorted sharding, manifest hashing,
  label join with mismatch accounting, PGN ingest 1–8 threads
  with identical manifest hashes), `rune-data` CLI (14
  subcommands), 12 unit + 4 integration tests, zero clippy
  warnings.
- Format/bridge: `.rune-data` v1 layout doc, Python reader
  (bit-identical batches vs `RuneDataset` collate on shared
  records, tamper rejection tested).
- Python: `training/samplers/disagreement.py` (mixture sampler,
  pipeline-only), `tools/analysis/oracle_routing.py` (ceiling +
  danger-miss buckets), `calibration.py`, `representation.py`
  difficulty proxies, `tools/match/play_match.py` (fixed openings,
  CI reporting, refinement/latency/uncertainty logging).
- Numbers: extract 88.8k/s (20x), ingest 55–265k/s (4.8x @8T),
  dedup 1.4–1.7M/s, raw R/W 678–980k/s @408B, deflate 51B/pos
  with ~free decode, bridge epoch delivery ~726/s tied with
  Python collate (both Python-loop-bound — pyo3 pending).
- Scale reality: validated to ~25k positions; 100M+ paths exist
  but are unrun; trainer-starvation proof explicitly scheduled.

## 2. Verdicts

### KEEP (v0.8 builds on these unchanged)

| Component | Evidence | v0.8 role |
| --------- | -------- | --------- |
| Exact Python parity (features 4302/4302, final sets FEN-equal, bit-identical batches) | Tested, incl. committed fixtures | Any selection operating on Rust metadata inherits trust; parity tests gate every semantic change |
| Determinism (manifest hash stable across thread counts; seeded RNG only) | Tested | Round reproducibility: same pool + model + score + seed = same selection (§34) |
| Format v1 + manifest/checksum discipline + corruption rejection | Roundtrip + tamper tests | Extended with active metadata (§35), never redesigned mid-stream |
| Stage-wise CLI + per-stage reports with reject reasons | Working, reported | Selection becomes stages (score → select → audit), same report discipline |
| Teacher cost accounting + stm convention + failed log | In tooling and docs | Extended to round-level budget ledgers (§13–§14) |
| Match harness with fixed conditions + per-metric reporting | Code, unrun on real models | Equal-budget active comparisons (§51) |
| No-composite-score, no-test-tuning, kill-loudly discipline | Methodology | Unchanged (§55, §61) |

### REMOVE (must not leak into the active loop)

| Component | Evidence | v0.8 role |
| --------- | -------- | --------- |
| Disagreement-as-default-answer | Zero measured learning value; v0.4–v0.6 never validated it | Demoted to B2 baseline; must beat B1 before any claim (§2–§3) |
| Silent fallbacks, composite winner scores, full-Cartesian runs | Standing bans | Stay banned (§55, §60) |
| Test-set threshold/weight tuning; magic unnormalized scores | Standing bans | Weights configurable, components exposed, normalized (§7) |
| Production dependency on selection metadata | §62 ban | C++ evaluator never sees active machinery |

### REWORK (same idea, new obligations)

| v0.7 form | Problem for v0.8 | v0.8 form |
| --------- | ---------------- | --------- |
| Disagreement sampler (pipeline util) | Single signal, no ablation, no danger-miss accounting | One of five scored signals (S1–S5) with per-signal ablations A0–A5 and FP/FN-by-bucket reporting |
| Representation difficulty proxies (unsupervised) | Not error, not calibrated | Replaced by supervised signals (teacher error, calibrated u) wherever labels exist; proxies stay labeled |
| Oracle routing tool (static pool ceiling) | No round dynamics, no rescoring | Per-round disagreement evolution + learned-vs-oracle gap tracking (§21, §34-v0.5-sense) |
| Match harness (score + cost) | No equal-budget active framing | Reports labels/compute/wall-time alongside quality; stopping-criteria inputs (§12, §49) |
| Phase stats only | Coverage needs material/composition/safety/eval-range/source axes | Coverage tracker per round vs pool vs batch (§9, §23) |

### EXTEND (new v0.8 construction on the v0.7 base)

- Candidate streaming scorer: per-record S1–S5 from precomputed
  teacher/student outputs, stored separately, never a magic scalar
  without components (§6–§7).
- Bounded top-K (heap) + two-level shard-local→global selection
  with approximation-quality checks (§31–§33).
- Diversity filter (hash exclusion → metadata buckets → cheap
  distance), costed against its compute (§11).
- Minimum-coverage guard + random/stratified floor ratio,
  configurable (§10).
- Label-store lookup + reuse (dedupe teacher spend, count
  reused vs requested) + staleness detection (§19–§20, §28).
- Round versioning (model/dataset/round/teacher/score/config
  ids, immutable artifacts, audit trail per sample) (§4, §18, §36).
- Teacher budget ledger (labels, compute, wall time, storage)
  and total-cost accounting (§13–§14).
- Replay + forgetting test harness (T0/T1/T2, retention on
  quiet/endgame/common) (§39–§41).
- Diminishing-returns curves (gain/round, gain/1M labels,
  stopping policy) (§47–§49).
- Sibling-group selection where child data exists
  (bindings-gated; individual-selection control always runs)
  (§26–§27).
- Rust selection engine for metadata-only paths; neural
  inference stays Python/C++ (§29–§30, §56).

### UNKNOWN (cheapest test first, same discipline)

- Which signal (if any) beats stratified at equal labels (RQ1–RQ2).
- Whether labeling cost actually falls at equal quality (RQ3).
- Learning-curve deltas, collapse/forgetting behavior, stopping
  points (RQ4–RQ5).
- Whether a smaller information-dense dataset exists (RQ6).
- Rust selection throughput at billion-scale; starvation verdict;
  optimal budgets, ratios, and replay needs.
- All carried UNKNOWNs (teacher at scale, QAT, pruning, rates).

## 3. Consequence for v0.8

Build order is forced: (1) scoring + selection + audit trail on
the existing engine, ablated signal by signal against B0–B2 at
equal labels; (2) diversity + coverage guards with collapse
monitoring; (3) round loop with versioning + budget ledgers +
replay/forgetting gates; (4) diminishing-returns + stopping
analysis. Rust does metadata-only selection; Python does inference
and analysis; the boundary moves only on measured reason. If no
strategy beats stratified at equal total cost, v0.8 ends with that
negative result and the simple pipeline stays — the spec demands
exactly this honesty (§66).
