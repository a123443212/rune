# RUNE — Relational Unified Neural Evaluator

Research framework for CPU-efficient neural game evaluation: chess, shogi,
xiangqi, and go. Small networks over sparse features with incremental
updates, developed generation by generation (configs `v01`–`v17`), each
gated on measured evidence, never on novelty.

## Contract

These hold for every generation. Changing one is a breaking change,
not a new experiment.

- Three subsystems meet only through artifacts: datasets
  (`.rune-data` plus manifests) and models (`.rune` plus hashes).
  No subsystem reaches into another's internals.
- Evaluation values are **side-to-move-relative**. Records stamp
  their perspective; mismatches are rejected, never coerced.
- Determinism: same input plus config plus seed yields the same
  bytes, proven by manifest, dataset, and model hashes.
- Loaders fail closed on anything they do not recognize: bad magic,
  bad FEN/SFEN, out-of-vocab indices, shape mismatches, checksum
  mismatches. Exact tolerances live in `spec/`, never in prose.
- No claim without a matched-condition experiment behind it:
  one changed variable at a time against a re-run control.
- Per-metric reporting. No composite scores, no hidden capacity wins.
- Smaller, simpler, faster wins ties against cleverer.
- Failed experiments are logged, never deleted.
- Teacher supervision is never free: generation, labeling, and
  storage costs are reported alongside any claim built on them.

## What is in this repo

- `core/` — native C++ runtime: board rules, feature sets,
  accumulators, architectures (MLP, attention, dual, multi-head,
  relational, dense, adaptive), SIMD kernels, model I/O (format 2),
  search, benchmarks. Tests in `tests/cpp/`.
- `training/` — Python/PyTorch mirror: per-game feature code
  (`training/games/`), models (`training/models/`), losses, datasets
  and sharding (`training/datasets/`), trainer, export to `.rune`
  (`training/export/`), experiment screening. Tests in `tests/`.
- `crates/` — Rust workspace: `rune-spec`, `rune-model`, `rune-kernel`,
  `rune-runtime` (reference + compiled evaluators), `rune-search`
  (alpha-beta, MCTS), `rune-ir`/`rune-compiler`, `rune-cli`,
  `rune-uci`, and the data engine (`rune-data-core`, `rune-data-cli`).
- `configs/` — experiment generations `v01`–`v17`; newest dir is current.
- `spec/` — normative contracts: `VERSIONS.md`, `games/`, `features/`,
  `model-format/`, `ir/`, `quantization/`, plus golden fixtures in
  `spec/test-vectors/` that every implementation must satisfy.
- `tools/` — parity and analysis scripts: `diff/` (python vs C++ vs
  Rust value checks), `bench/`, `analysis/`, `dataset/`, `compile/`.
- `bindings/` — pybind11 bridge used by the Python parity tests.

Games and their feature versions: chess
(`grouped_hkav2_fullthreats_v02`, 9 groups), shogi (`shogi_raw_v01`),
xiangqi (`xiangqi_raw_v01`), go (`go_planes_v01`/`go_planes_v02`;
v02 token groups 0–1 use disjoint `ci*361+sq` encoding).

## Build and test

Each subsystem builds with its own toolchain and tests itself:

```bash
cmake -S . -B build -DRUNE_BUILD_BINDINGS=OFF && cmake --build build -j4
ctest --test-dir build

pip install -r requirements.txt
pytest tests/

cargo test --workspace --locked
```

`scripts/python_test.sh` runs the Python suite the way CI does.
For Python↔C++ parity tests, build with bindings on
(`-DRUNE_BUILD_BINDINGS=ON`, needs matching CPython headers and
pybind11) so `rune_bindings` imports instead of skipping.

Built CLIs document themselves. When in doubt, ask them:

```bash
cargo run -p <cli-crate> -- --help
python <tool-script> --help
```

## Reproduce an experiment

Every generation follows the same stages. Find the current
generation's config dir (`ls configs`, newest is current), then walk
the stages:

1. Build a position pool from games.
2. Label it with an engine teacher at recorded depths,
   deterministically, keeping provenance and hashes.
3. Train and screen from the generation config: same seed,
   same data, one variable per leg against a re-run control
   (`training/experiments/screening.py`, per-leg flags such as
   `pair_legs` override the global `architecture` section).
4. Shard the set with the data engine: validate, dedup on full
   state, hash-shard, manifest.
5. Export the model (`training/export/export.py`) and verify payload
   and header hashes; never load checkpoints with bare
   `torch.load` — always `weights_only=True`.
6. Check parity (`tools/diff/`) then play engine matches at fixed
   nodes and at fixed time.

Every run records config, seed, dataset, teacher, and model
hashes, hyperparameters, and position counts in `metrics.json`.
Interpretation rules live with the generation config, not here.

## Map

Do not document the current state here — it rots. Derive it:

```bash
git log --oneline -10        # what changed lately; newest is current
ls configs                   # experiment generations; newest dir is current
cat spec/VERSIONS.md         # what the version numbers mean
ls spec/test-vectors         # golden fixtures every implementation must satisfy
```

## Conventions

Decided once, followed everywhere. Details and tolerances are
normative in `spec/`; this list is only an index:

- Negamax values, stamped perspective, reject on mismatch.
- Deterministic bytes from input plus config plus seed.
- Priced teachers: report generation, labeling, and storage.
- Evidence-gated versions: novelty alone promotes nothing.
- Fail loud: null components, bad indices, bad shapes, bad
  positions, and bad time controls are errors, never silent zeros
  or silent fallbacks.
