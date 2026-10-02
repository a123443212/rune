# v0.2 Audit: gain thật vs compute (evidence-gated)

Rule: a component has "gain" only with a trained comparison under identical
conditions. Microbench, parity, and smoke runs measure cost and correctness,
never gain. Status levels: PROVEN (trained comparison), MEASURED (bench
numbers only), UNPROVEN (no data either way).

## Verdict table

| Component | Gain? | Compute cost (measured) | Verdict |
| --------- | ----- | ----------------------- | ------- |
| Attention core (QKT+gate+Vmix) | UNPROVEN, no trained run | ~2.5 us, tiny | EXPERIMENTAL: cheap but purposeless until A0-vs-A2 trains |
| QKV projections | UNPROVEN, and it is the mixer cost | ~15-57 us (noisy), dominant mixer cost | EXPERIMENTAL: the most expensive unproven part |
| Static GAB | UNPROVEN | ~0.2-1 us, +64 params, negligible | KEEP AS OPTION: free to keep, must still prove effect |
| Dynamic GAB | UNPROVEN + over-budget suspicion | clearly above op-count expectation, +128 params | STRIP FROM CORE: measured cost, zero benefit evidence, pre-registered kill rule fires |
| Gate registry / alpha | UNPROVEN | ~0 | KEEP AS OPTION: flexibility with no cost |
| 6-token compact | UNPROVEN quality | strictly cheaper (fewer rows, 99k vs 108k params) | FIRST CHALLENGER: cheapest RQ1 test, MLP-only, no attention confound |
| 10-token / 24-40 dims | UNPROVEN | 10x40 strictly most expensive | EXPERIMENTAL, lowest priority |
| Ranking loss (true siblings) | UNPROVEN (128-pos smoke only) | small extra forward per step | EXPERIMENTAL, L1 only on candidates |
| Disagreement sampling | UNPROVEN, labeling cost unmeasured | pipeline only | EXPERIMENTAL, stage-gated after S0 |
| INT16 embeddings | MEASURED neutral (diff ~3e-8, consistency 1.0) + ~30% smaller files | same int32 accum path | KEEP: closest thing to a proven gain (efficiency, not strength) |
| Head 256->128 | N/A (fixed for clean comparisons) | dominant eval cost ~33-40 us | FROZEN, redesign parked |

## Answers to the six questions

- **Component nào có gain thật?** None on strength. INT16 is the only
  efficiency gain with measurements (accuracy-neutral, smaller). Everything
  else is unproven either way.
- **Component nào chỉ tăng compute?** Dynamic GAB (measured overhead, no
  benefit); QKV projections (cost without proven gain); 10x40 (cost without
  proven gain).
- **Attention có thật sự cần không?** Unknown. Its necessity is exactly what
  A0-vs-A2 at 25M+ must decide. The core must not assume it in the meantime.
- **GAB có tác dụng không?** Unknown. Static stays as a zero-cost option;
  dynamic leaves the core.
- **Ranking loss có giúp learning curve không?** Unknown beyond smoke.
  Stays L1-on-candidates.
- **Disagreement sampling có đáng tiền không?** Unknown; price (labeling)
  and value (gains) both unmeasured. Stays stage-gated.

## Consequence

v0.3 Core = the cheapest proven-working configuration plus measurement
harness, with every unproven component behind an explicit experimental
opt-in. Nothing is deleted (deletion would destroy the experiments that
can answer the questions); defaults are narrowed instead.
