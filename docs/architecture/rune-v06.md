# RUNE v0.6 Architecture: Confidence-Weighted Distillation

Question: strong teacher → compact student → efficient INT8
deployment at minimal quality loss. Less model, same useful chess
knowledge. Never a bigger model.

## Teacher–student setup

```
Teacher (T0: first benchmarked teacher, never "newest by default")
  value / WDL / uncertainty (+ optional ranking pairs)
        ↓ precomputed labels with teacher id + hash + cost accounting
Student (strictly smaller in ≥1 of params / dims / cost / bytes)
  same features (attribution stays clean)
  + configurable token dims, head widths, mixer presence
```

## Student budgets (compute targets, not hard gates)

```
S0  teacher reference (100%)
S1  ~75%   S2  ~50%   S3  ~33%   S4  ~25%
```

Via smaller token dims, narrower heads, reduced/absent refinement.
Feature representation frozen wherever possible.

## Distillation targets (one axis at a time)

```
D0  direct training (no teacher — the control that can kill the track)
D1  teacher value                         D2  + teacher WDL
D3  uncertainty-weighted value/WDL        D4  + ranking distillation
```

Representation/layer-wise distillation only after value legs
signal, with explicit alignment (small projection when dims
differ, never a large one).

## Weighting functions (bounded, at least two hypotheses)

```
confidence-weighted:  w from teacher uncertainty (high-conf ↑ weight)
difficulty-weighted:  w from teacher error/uncertainty (hard ↑ weight)
uniform:              always present as the control
```

No assumption about which direction helps — that is the experiment
(§7). Uncertainty weighting is gated on the v0.5 calibration check;
uncalibrated uncertainty never weights anything.

## Loss (independent module, every coefficient configurable)

```
ValueLoss / WDLLoss / RankingLoss / DistillationLoss / WeightedDistillationLoss
L = L_task + α L_distill
```

Three-way reporting always: student-vs-teacher, student-vs-target,
teacher-vs-target — so distillation gain and teacher bias never mix.

## Deployment target

INT8 student via existing embedding-quant paths; QAT only if
post-quant degradation is significant; mixed precision only on
measured benefit. Calibration-under-quant is a kill condition for
any uncertainty-carrying student.

## Serialization

Student is a standalone `.rune` (same format family, teacher-free
loader): arch id/version, feature version, training config hash,
dataset id/hash, teacher id/hash, quantization format, checksum.
Reproducibility fields per §32 are mandatory in every run record.

## Explicit non-goals

Policy networks, RL, MCTS, giant Transformers, MoE, recurrent
search nets, neural move ordering, composite scores, test-set
threshold tuning, "students" at teacher scale.
