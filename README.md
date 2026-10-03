# RUNE — Relational Unified Neural Evaluator

Research framework for CPU-efficient neural chess evaluation:
sparse grouped features, incremental accumulators, compact
architectures, adaptive computation, and a high-throughput data
engine — developed version by version, each gated on measured
evidence, never on novelty.

House rules (stable across all versions):

- No claim without a matched-condition experiment behind it.
- Per-metric reporting; no composite scores, no hidden capacity wins.
- Smaller, simpler, faster wins ties against cleverer.
- Failed experiments are logged, never deleted.

## Subsystems

| Direction | Role | Lives in |
| --------- | ---- | -------- |
| C++ | Chess runtime and inference (board, features, accumulators, architectures, SIMD, model I/O, benchmarks, tests) | `core/`, `benchmarks/`, `bindings/`, `tests/cpp` |
| Python / PyTorch | Training and research (features mirror, models, losses, datasets, trainer, export, screening, analysis, match harness) | `training/`, `tools/`, `tests/test_*.py` |
| Rust | Data engine (parse, validate, dedup, filter, shard, serve batches, `.rune-data` format) | `rust-data/`, `tools/data_bridge/` |

The three meet only through artifacts: datasets (`.rune-data`,
manifests) and models (`.rune`, checksums). No subsystem reaches
into another's internals.

## Versions

Work proceeds as `rune-vNN` generations. Each generation owns:

- `docs/architecture/rune-vNN.md` — what was built and why
- `docs/experiments/rune-vNN-plan.md`, `rune-vNN-results.md` — running log
- `docs/experiments/rune-vNN-vMM-audit.md` — evidence-gated audit of the previous generation
- `docs/performance/rune-vNN-*.md` — measured numbers with machine caveats
- `configs/vNN/` — experiment configs for that generation
- `docs/experiments/failed/` — dead ends, kept readable

To find the current state, read the newest
`docs/experiments/rune-vNN-results.md` (or `git log --oneline`).
Any document that names a "latest version" instead of pointing
here is stale by definition.

## Build

```bash
cmake -S . -B build && cmake --build build -j4
./build/rune_tests          # C++ suite (needs no extra setup)

pip install -r requirements.txt
pytest tests/               # Python suite (some tests need the bindings below)

cargo build --workspace --manifest-path rust-data/Cargo.toml
cargo test --workspace --manifest-path rust-data/Cargo.toml
```

Python bindings (`rune_bindings`) need CPython headers. Without
root, fetch and unpack them locally instead of `apt install`:

```bash
apt-get download libpython3.14-dev   # adjust version to `python3 --version`
dpkg-deb -x libpython3.14-dev_*_amd64.deb /tmp/pyheaders
# then compile with -I/tmp/pyheaders/usr/include/python3.14
# -I/tmp/pyheaders/usr/include (see repo history for the full g++ line)
```

## Reproduce an experiment

```bash
python tools/dataset/build_real_pool.py --out data/pool.jsonl --games 200
python tools/dataset/label_engine.py --pool data/pool.jsonl --out data/labeled.jsonl \
    --engine /path/to/stockfish --depth 12 --copy-to-ground-truth
python tools/screening/run_screening.py --config configs/vNN/<leg>.yaml --pool data/labeled.jsonl
./rust-data/target/release/rune-data benchmark --pool data/pool.jsonl
```

Every run records config, seed, dataset/teacher/model hashes,
hyperparameters, and position counts in `metrics.json`; see the
current generation's plan doc for interpretation rules.

## Key conventions (decided once, followed everywhere)

- Evaluation values are **side-to-move-relative** (negamax).
  Records stamp `value_perspective`; mismatches are rejected, never coerced.
- Determinism: same input + config + seed = same bytes
  (manifest/dataset/model hashes prove it).
- Teacher supervision is never free: generation, labeling, and
  storage costs are reported alongside any claim built on them.
