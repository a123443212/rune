# RUNE v0.1 — Relational Unified Neural Evaluator

Modular neural chess-evaluation research framework: sparse grouped features,
incremental accumulators, CPU-friendly inference, and controlled
attention-vs-MLP ablations.

## Layout

- `core/` — C++20 runtime: board, position, features, accumulators,
  architectures (`base`, `mlp`, `attention`), inference, SIMD helpers, model I/O
- `training/` — PyTorch research side: features mirror, models, losses,
  datasets, trainer, export/quantization, learning-curve experiments
- `tools/` — dataset builder, benchmark, match runner
- `benchmarks/` — C++ microbenchmark + NPS walk
- `bindings/` — pybind11 module `rune_bindings`
- `tests/` — C++ (`tests/cpp`) and pytest suites
- `configs/` — versioned (`v01/`, `v02/`) and `rune_v01` experiment configs
- `docs/` — architecture spec, experiment plan, future work

## Build

```bash
cmake -S . -B build
cmake --build build -j4
./build/rune_tests
./build/rune_bench
```

Python: `pip install -r requirements.txt`, then `pytest tests/`.

## Reproduce an experiment

```bash
python tools/dataset/build_dataset.py --out data/synth.jsonl --games 500
python -c "from training.experiments.learning_curve import LearningCurve; ..."
python tools/benchmark/bench.py --arch RUNE-ATTN-GAB
```

Every run records config, seed, dataset/model identifiers, hyperparameters,
and position counts; see `docs/experiments/v01.md` for interpretation rules.
No strength claims without matched-condition match tests.
