# RUNE v0.2 Results (running log)

## Hypothesis / setup / metrics

Each experiment states its hypothesis in the plan doc. Setup is fixed by
config: shared clean pool, fixed game-grouped split and seed, one teacher
version, identical recipe except the varied axis. Metrics per milestone:
value/WDL loss, WDL acc, rank acc, grad stability, throughput, params, size,
NPS/latency split (accumulator vs mixer vs head), quant consistency.

## Results so far (code-level evidence only, no trained models yet)

- Parity: torch vs C++ eval agreement 1e-4 for all rel variants
  (6/8/10 tokens, 24/32/40 dims, clip/hard_sigmoid, alpha in {0.5, 0.75, 1.0},
  static/dynamic). Context vectors bit-exact. Flex incremental == refresh.
- Export/load: `.rune` roundtrip fp32/int8/int16 for MLP and REL; C++ loads
  trained-checkpoint exports (smoke) and evaluates within tolerance.
- Quant on random weights: int16 ~ fp32-exact (diff ~3e-8), int8 diff ~1e-6,
  sign/WDL consistency 1.0 on 4 positions. Says nothing about trained nets.
- Inference split (noisy machine, ranges): accumulator ~13-16 us, head
  ~33-40 us, v0.1 attention mixer ~15-57 us, rel static mixer ~20-52 us,
  rel dynamic mixer higher with large variance. 6-token strictly cheaper,
  10x40 strictly more expensive, matching parameter counts.
- Smoke training: rel static/dynamic train, checkpoint, export, and report
  through the screening pipeline at 400/800 positions.

## Interpretation

Nothing here supports or refutes RQ1-RQ4. The only substantive finding is
negative in the small: the dynamic-bias path costs more than its op count
suggests on this machine, so B2 enters screening under suspicion and dies
on NPS regression without metric gain. Head cost (not attention) remains
the eval bottleneck, unchanged from v0.1.

## Limitations

No teacher labels, no 25M+ runs, no engine matches, one noisy shared
machine for all latency numbers. Every table above needs a quiet-machine
re-run and trained weights before it means anything about strength or data
efficiency.

## Failed experiments / fixes

- `split_by_game` starved val/test on small pools (all 17 games hashed to
  train), producing silent zero metrics. Fixed with a deterministic fallback
  plus a fail-fast guard. Lesson: metrics pipelines must assert non-empty
  evaluation sets.
- An early `sys.path` mistake in `tools/screening/*` (one extra `..`) was
  masked by pytest's path setup and only caught when running CLIs standalone.
  Lesson: smoke-test tools outside pytest too.
