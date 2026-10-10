# RUNE — Relational Unified Neural Evaluator

CPU-efficient neural evaluation for chess, shogi, xiangqi, and go.
Small networks over sparse features with incremental updates, built in
three matching implementations (C++, Python/PyTorch, Rust) that are
kept in parity against golden fixtures. Research proceeds generation
by generation (`configs/v01`–`configs/v17`); nothing promotes on
novelty alone — only on measured wins against a re-run control.

## How it works

A position becomes a sorted list of `(group, index)` sparse features
(9 groups for chess). An accumulator sums one embedding row per active
feature into 8 tokens, a mixer (attention / gated relational / dual /
multi-head / dense pooling / adaptive refine) mixes the tokens, and a
value+WDL head scores them. C++ and Rust reuse the same math with
incremental refresh (`refresh` vs `updateIncremental`); Python is the
training reference. `tools/diff/` proves the three agree within the
tolerances in `spec/`.

Supported games and feature sets:

| Game    | Feature version              | Notes                                    |
|---------|------------------------------|------------------------------------------|
| Chess   | `grouped_hkav2_fullthreats_v02` | 9 groups, 8 tokens × 32 dims          |
| Shogi   | `shogi_raw_v01`              | SFEN input                               |
| Xiangqi | `xiangqi_raw_v01`            | FEN input                                |
| Go      | `go_planes_v01` / `go_planes_v02` | 9/13/19 boards; v02 groups 0–1 use disjoint `ci*361+sq` |

Architectures: `RUNE-MLP`, `RUNE-SFNN` (+ compact/small variants),
`RUNE-ATTN` / `RUNE-ATTN-GAB` / `RUNE-ATTN-DUAL` / `RUNE-ATTN-MH4`,
`RUNE-REL-02` (6/8/10 tokens), `RUNE-REL-LITE`, dense `RUNE-03-*`
and adaptive `RUNE-04`/`RUNE-05` (C++ training path).

## Layout

- `core/` — C++ runtime: rules, features, accumulators, architectures,
  SIMD kernels, model I/O (`.rune` format 2), search, benchmarks.
  C++ tests live in `tests/cpp/`.
- `training/` — PyTorch side: game mirrors, models, losses, datasets
  and hash-sharding, trainer, `.rune` export, screening harness.
  Python tests live in `tests/`.
- `crates/` — Rust workspace: `rune-spec`, `rune-model`, `rune-kernel`,
  `rune-runtime` (reference + compiled evaluators), `rune-search`
  (alpha-beta + MCTS), `rune-ir` / `rune-compiler`, `rune-cli`,
  `rune-uci`, data engine (`rune-data-core`, `rune-data-cli`).
- `configs/` — experiment generations; the newest directory is current.
- `spec/` — normative contracts (`VERSIONS.md`, `games/`, `features/`,
  `model-format/`, `ir/`, `quantization/`) and golden fixtures in
  `spec/test-vectors/` that all three runtimes must satisfy.
- `tools/` — `diff/` (cross-runtime parity), `bench/`, `analysis/`,
  `dataset/`, `match/` (engine games), `search/`, `inspect/`.
- `bindings/` — pybind11 bridge backing the Python parity tests.

Subsystems meet only through artifacts: `.rune-data` + manifests for
data, `.rune` + payload/header hashes for models.

## Quickstart

```bash
pip install -r requirements.txt
cmake -S . -B build -DRUNE_BUILD_BINDINGS=OFF && cmake --build build -j4
ctest --test-dir build
pytest tests/
cargo test --workspace --locked
```

`scripts/python_test.sh` runs the Python suite the way CI does. For the
Python↔C++ parity tests, rebuild with `-DRUNE_BUILD_BINDINGS=ON`
(requires matching CPython headers and pybind11) so `rune_bindings`
imports instead of skipping. Checkpoints must always be loaded with
`torch.load(..., weights_only=True)`.

## Using it

```bash
cargo run -p rune-cli -- --help
python tools/diff/cross_check.py --help
python tools/diff/run_diff.py --help
```

Typical loop: build a pool → label with a teacher → train/screen from
a generation config → hash-shard the dataset → export `.rune` →
parity-check across runtimes → engine matches at fixed nodes and
fixed time. UCI is served by `rune-uci` (`position`/`go`/`stop`);
time control is side-aware and bad positions are reported, never
silently kept.

## Experiment rules

- One changed variable per leg against a re-run control; per-leg flags
  (e.g. `pair_legs`) override the global `architecture` section.
- Per-metric tables, no composite scores; capacity and teacher costs
  (generation, labeling, storage) are reported with every claim.
- Every run records config, seed, dataset/teacher/model hashes, and
  position counts in `metrics.json`. Failed runs stay in the log.

## Map

State rots here, so derive it instead:

```bash
git log --oneline -10
ls configs
cat spec/VERSIONS.md
ls spec/test-vectors
```
