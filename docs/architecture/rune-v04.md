# RUNE v0.4 Architecture: Adaptive Computation + Selective Precision

Question: same intelligence, less unnecessary computation. Easy
positions should be cheap; difficult positions receive computation
selectively. One base path, one refinement path. Not MoE.

## Pipeline

```
Sparse features (frozen)
-> incremental var-width accumulator (frozen, reused by both paths)
-> group accumulator segment
-> cheap token formation (none or shared pool, cheapest viable)
-> cheap reduced mixer (identity or tiny shared linear)
-> tiny cheap head -> preliminary value/WDL + difficulty scalar
        |
   difficulty < threshold --> cheap output (EASY)
        |
   difficulty >= threshold -> refinement block -> refined head (HARD)
```

## Cheap path

The cheap path is a usable evaluator on its own (A1 baseline):

```
tokens -> cheap formation -> cheap mixer -> cheap head (T*D -> 32 -> value+WDL)
```

No large side network decides routing. The difficulty signal comes
from the cheap path's own outputs: WDL uncertainty, value margin,
plus one tiny learned scalar head on the shared representation
(a single linear layer to 1 unit — overhead negligible by
construction, costed in microbenchmarks).

## Refinement path

Receives cheap-path tokens and intermediate buffers directly (§36 —
no feature recomputation). One fixed relational block (static bias
only, no dynamic GAB), then a refined head:

```
cheap tokens -> relational refinement (static GAB, alpha fixed)
             -> refined head (T*D -> 128 -> 32 -> value+WDL)
```

Fixed single step. Fixed maximum cost. No recurrence, no
"refine until good enough".

## Difficulty and deterministic routing

Routing signals, cheapest first:

- WDL entropy of the cheap output (free, no extra params).
- Learned difficulty scalar (one linear layer, trained against
  oracle labels offline — never in the engine loop as a classifier).
- Optional activation statistics / token disagreement (analysis
  only until proven worth their cost).

Engine routing is a pure threshold compare, config-controlled:

```
difficulty < T --> cheap only
difficulty >= T --> refinement
```

Optional hysteresis (`T_high` enter, `T_low` exit) exists in code
but defaults OFF and ships only if routing-flip instability is
measured (§27). Same FEN always gives same routing and same eval.

## Precision

P0 uniform INT8 embeddings (existing int32 accumulator path).
P1 INT8 base + higher-precision refinement (refinement matmuls in
INT16/FP32 while cheap stays INT8). P2 INT8 base + higher precision
only for critical ops named by sensitivity analysis. No FP16 on CPU
unless the target CPU benefits — decided by benchmark, not fashion.

## Interaction pruning

The 8x8 relational matrix is analyzed for low-contribution
interactions (contribution measured by ablation with retrain, never
correlation alone). I1 masks a fixed subset to zero with predictable
shape — no dynamic allocation, no sparse formats in the hot path.
Static semantic patterns (king-threats, pawn-minor, rook-queen) are
benchmarked as alternatives to dynamic selection (§18).

## Cost model (bookkeeping, not evidence)

```
C_avg = (1-r) * C_cheap + r * C_refine + C_route
```

`r` is the measured refinement rate. This formula plans sweeps
(10/25/50/75/100%); the CPU benchmark confirms them. Adaptive-100%
must equal always-refinement numerically (implementation check).

## Serialization

One package carries: base weights, refinement weights, routing
thresholds (+ optional hysteresis), precision metadata, arch version
`0.4.0`, feature version, checksum. Loader rejects incompatible
packages explicitly. v0.1–v0.3 single-path files keep loading.

## Explicit non-goals

MoE, full Transformer, Mamba/SSM, recurrent multi-step refinement,
large expert routing, RL, neural search policy, giant uncertainty
networks, composite quality scores. Worst-case cost is bounded by
construction and reported next to every average.
