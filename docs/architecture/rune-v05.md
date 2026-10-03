# RUNE v0.5 Architecture: Search-Aware Evaluation

Question: value + confidence + stability + adaptive computation at
minimal added compute. More aware, not larger. Never a policy network.

## Pipeline

```
Sparse features (frozen)
-> incremental accumulator (frozen, reused)
-> cheap token formation
-> cheap head -> value/WDL (primary output, usable standalone)
             -> difficulty scalar (v0.4 routing compat)
             -> uncertainty scalar u in [0,1] (NEW, tiny)
-> deterministic routing from {difficulty, uncertainty, stability}
-> refinement block (static, single step) -> refined value/WDL
```

## Uncertainty head

One linear layer from the shared cheap representation to a single
scalar, bounded to [0,1] (sigmoid), trained to regress
`|teacher_value − student_value|`. Cost: one dot product + sigmoid —
measured in microbenchmarks, expected negligible next to the cheap
head. One form only (bounded scalar); no variance/confidence zoo.

The v0.4 unbounded difficulty scalar stays for routing backward
compatibility until the R-ladder retires or keeps it on evidence.

## Stability signal (analysis-first, optional in engine)

Offline definition, never computed by search in production:

```
stability ≈ spread of values over legal children
ΔV = max_child − min/typical_child
```

Children come from the dataset's parent-child records or
teacher-derived neighbors. The engine sees only a distilled scalar
threshold (if the A3 leg proves it), never a child enumeration.

## Routing

Signals enter one cheap deterministic decision:

```
R0: difficulty only (v0.4 baseline)
R1: uncertainty only
R2: difficulty + uncertainty (two-threshold OR / linear combo)
R3: + stability scalar
```

All thresholds calibrated on validation, never test. Operating
points 10/25/50/75/100% reported separately with quality, avg
latency, worst-case latency, refinement rate. Hysteresis stays off
unless flip instability is measured.

## Training targets without leakage

- Uncertainty target: `|teacher − student|` computed at train time
  from teacher labels. Production sees only student representation.
- Stability target: child-value spread from recorded
  parent-child pairs. No search at inference, no future results in
  the runtime.
- Distillation (cheap ← full) allowed with teacher-generation cost
  reported separately, never called free.

## Loss ladder (each λ configurable, each leg ablated alone)

```
L0 = Value + WDL
L1 = L0 + λ3 Uncertainty
L2 = L1 + λ4 Stability
```

Rank loss stays as before where already gated. No compute loss
unless a with/without experiment states its purpose.

## Cost and memory posture

Uncertainty head, stability handling, and routing are each
microbenchmarked. Added intelligence must cost ~nothing: no large
temporary tensors, shared scratch, bounded worst case. INT8
uncertainty is benched for numerical drift AND calibration drift —
a head that loses calibration under quant is not deployment-ready.

## Serialization

Package adds uncertainty weights + routing config (signals used,
thresholds, calibration summary) + arch version `0.5.0`. Loader
rejects mismatches. v0.1–v0.4 files keep loading.

## Explicit non-goals

Full move policy, neural move ordering, RL, alpha-beta replacement,
MCTS, large Transformer, MoE, recurrent search networks, learned
search trees, composite scores, test-set threshold tuning.
