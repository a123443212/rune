# v0.1 Audit (input to v0.2)

Measured on this machine via `rune_stage_bench` / `infer_bench.py`. Absolute
numbers are machine-specific; ratios are the evidence.

## Inference bottleneck

| Stage | Cost (us) | Share of full eval |
| ----- | --------- | ------------------ |
| feature extract (refresh) | 5-8 | dominant in search walk |
| accumulator refresh / update / tokens | 2-16 / 0.4-0.8 / 0.44 | small |
| QKV projections (attention) | 15-19 | largest mixer cost |
| QKT + GAB bias + clip + Vmix + residual | ~2.5 total | negligible |
| head 256->128->32 | 30-56 | dominant in eval |
| full eval incremental, MLP / ATTN-GAB | ~45 / ~50-65 | - |

Conclusion: the attention core is cheap. The bottlenecks are the head first
layer and the QKV projections. GAB costs ~0.2 us and 64 params. NPS ~25k in
the random walk is dominated by full feature extraction per node (no
move-derived incremental path yet).

## Architecture bottleneck

- v0.1 attention is one fixed block: single gate (clip), alpha = 1, static
  8x8 bias, fixed 8x32 tokens. No evidence it helps: no trained comparison
  exists (only random-weight smoke).
- Static GAB cannot adapt to phase/material; whether that matters is RQ2/RQ4
  territory, untested.
- Token grouping (8 groups) and dim 32 were chosen, never ablated (RQ1).

## Data bottleneck

- No teacher labels exist; all runs used synthetic random playouts.
- `split_by_game` could starve val/test on small pools (fixed with
  deterministic fallback in this cycle).
- Phase balance truncates heavily when endgames are rare (observed 6886 -> 720).
- Disagreement sampling exists but never ran against a real teacher.

## Loss bottleneck

- Ranking loss uses within-batch pseudo-siblings, not true parent/child
  pairs with recorded teacher ordering. Provenance gap: pairs cannot be
  traced to a parent position. v0.2 must generate real sibling pairs.
- Loss API is a monolith (`RuneLoss`); toggling components works via config
  but there is no per-component class structure.

## What v0.2 upgrades (and why)

1. Gated relational mixer with configurable gate + residual alpha: QKV and
   the attention core are cheap, so a slightly richer mixer is affordable.
2. Dynamic GAB from a tiny context vector: static bias may be too rigid;
   keep the dynamic path small enough to preserve NPS.
3. Token count/dim ablations on a layout-driven flex path: grouping was
   never tested; the flex path leaves v0.1 code untouched.
4. True sibling-pair ranking data with provenance + component loss API.
5. INT16 quantization step + full FP32/INT16/INT8 comparison.
6. Head cost is noted but the head stays fixed in v0.2 to keep mixer
   comparisons clean; head redesign is future work.
