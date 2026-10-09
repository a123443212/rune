# RUNE — Relational Unified Neural Evaluator

Research framework for CPU-efficient neural chess evaluation.
One idea, many generations: small sparse-feature networks with
incremental updates, developed version by version, each gated on
measured evidence, never on novelty.

This file states only what does not change. Anything with a version
number, a size, or a "latest" in it lives elsewhere — see Map below.
If this file ever names one, that is a bug: fix this file, not reality.

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
- Loaders fail closed on anything they do not recognize. Exact
  tolerances live in `spec/`, never in prose.
- No claim without a matched-condition experiment behind it:
  one changed variable at a time against a re-run control.
- Per-metric reporting. No composite scores, no hidden capacity wins.
- Smaller, simpler, faster wins ties against cleverer.
- Failed experiments are logged, never deleted.
- Teacher supervision is never free: generation, labeling, and
  storage costs are reported alongside any claim built on them.

## Map

Do not document the current state here — it rots. Derive it:

```bash
git log --oneline -10        # what changed lately; newest is current
ls configs                   # experiment generations; newest dir is current
cat spec/VERSIONS.md         # what the version numbers mean
ls spec/test-vectors         # golden fixtures every implementation must satisfy
```

Source layout, by role rather than by name:

- Native runtime and inference: board, features, accumulators,
  architectures, SIMD, model I/O, benchmarks, and its tests.
- Training and research in Python and PyTorch: feature mirrors,
  models, losses, datasets, trainer, export, screening, analysis,
  match harness, and its tests.
- A single Rust workspace holding both the inference runtime
  (spec, model, kernel, runtime, search, IR, compiler, CLI) and
  the data engine (parse, validate, dedup, filter, shard, batches).

## Build and test

Each subsystem builds with its own toolchain and tests itself.
Concrete target names change; these shapes do not:

```bash
cmake -S . -B build && cmake --build build -j4
ctest --test-dir build

pip install -r requirements.txt
pytest tests/

cargo build --workspace
cargo test --workspace --locked
```

Built CLIs document themselves. When in doubt, ask them:

```bash
cargo run -p <cli-crate> -- --help
python <tool-script> --help
```

Native Python bindings need CPython headers matching
`python3 --version`. Without root, fetch and unpack the matching
headers package locally instead of installing system-wide.

## Reproduce an experiment

Every generation follows the same stages. Find the current
generation's config dir via Map, then walk the stages:

1. Build a position pool from games.
2. Label it with an engine teacher at recorded depths,
   deterministically, keeping provenance and hashes.
3. Train and screen from the generation config: same seed,
   same data, one variable per leg against a re-run control.
4. Shard the set with the data engine: validate, dedup,
   filter, shard, manifest.
5. Export the model and verify payload and header hashes.
6. Play engine matches at fixed nodes and at fixed time.

Every run records config, seed, dataset, teacher, and model
hashes, hyperparameters, and position counts in `metrics.json`.
Interpretation rules live with the generation config, not here.

## Conventions

Decided once, followed everywhere. Details and tolerances are
normative in `spec/`; this list is only an index:

- Negamax values, stamped perspective, reject on mismatch.
- Deterministic bytes from input plus config plus seed.
- Priced teachers: report generation, labeling, and storage.
- Evidence-gated versions: novelty alone promotes nothing.
