# RUNE v0.4 Profiling

Target: real-CPU average cost. Theory plans, hardware decides.

## Required splits (ns/eval or cycles/eval, quiet machine)

```
feature extract / accumulator refresh / accumulator update
cheap token formation / cheap mixer / cheap head
difficulty scalar / threshold routing (branch taken / not taken)
refinement block / refined head
precision conversion (where applicable)
full adaptive eval at refinement rates 10/25/50/75/100%
worst-case single eval (refinement path, cold + warm cache)
```

Adaptive-100% must equal always-refinement numerically.

## CPU behavior checklist (§31)

- Branch predictability of the routing compare at each rate.
- Cache locality: token/intermediate buffer reuse between cheap and
  refinement (no recompute, shared scratch where sound).
- Instruction footprint and SIMD utilization per stage.
- NPS end-to-end (search walk) vs microbench eval rate — report both;
  a router that wins microbench but loses the walk loses.

## Reporting

Per-metric only: avg latency, worst-case latency, refinement rate,
NPS, nodes searched, match result with uncertainty. Plus the
bookkeeping formula `C_avg = (1-r)*C_cheap + r*C_refine + C_route`
with measured (not assumed) components. No composite score.

## Log

- 2026-10-03, Tsukuba shared host (NOISY, single run, random weights —
  ratios only, never absolutes):
  `rune_stage_bench` adaptive section (8x32, cheap pooling none):
  cheap refresh 13.9 us / cheap incremental 7.6 us /
  always incremental 53.2 us / adaptive-forced-cheap 7.8 us /
  adaptive-forced-refine 53.1 us / cheap forward 7.8 us /
  refine forward 51.8 us / route compare ~0.001 us.
  Correctness: adaptive-low == cheap, adaptive-high == always.
  Params: 116585 total / 76197 cheap-path (incl. embeddings).
  Reading: routing overhead negligible vs paths; refined head
  (~128-wide, cf. rel_head 43.8 us) dominates refinement cost —
  same head-first-layer bottleneck as v0.2/v0.3. Bookkeeping
  C_avg at r=10%: ~12.2 us vs 53.2 us always (~4.4x IF quality
  holds — unproven, needs trained weights + match).
- Quiet-machine protocol from future-work still pending before any
  publishable claim.
