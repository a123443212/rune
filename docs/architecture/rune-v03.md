# RUNE v0.3 Architecture: Information-Dense Representation

Question: more chess-relevant information per dimension, per model
byte, per unit of inference cost. Not a bigger model.

## Pipeline

```
Sparse features (frozen grouped_hkav2_fullthreats_v01)
-> incremental var-width accumulator (exact, int32 path kept)
-> group accumulator segment (width = group width)
-> learned token pooling (none / per-token / shared)
-> bounded channel gate (optional, tiny)
-> dense head (T*D -> 128 -> 32 -> tanh value + WDL)
```

No attention before tokenization. No dynamic allocation. No MLP
before the token exists. The mixer question is deliberately deferred
to one experiment (`mixer_check.yaml`): representation + simple path
vs representation + v0.2 relational mixer.

## Why feature -> token is the focus

v0.2 measured where eval time goes (head first, QKV second, attention
core negligible) but never measured what each token carries. Uniform
8x32 assumes every semantic group deserves equal capacity. The audit
(`docs/experiments/rune-v03-v02-audit.md`) shows that assumption was
never tested. v0.3 tests it at fixed total budget, so a win cannot be
a hidden capacity increase.

## Token formation

`VarEmbedder` accumulates per-group embeddings exactly like v0.2,
then `TokenPool` maps each group segment to its token:

- none: identity, group width must equal token dim.
- per_token: one DxD linear + bias per token, clipped to [0,1].
  This is P0 and the A2/A3 pooling leg.
- shared: one shared 32x32 linear plus per-token scale/bias, clipped.
  This is P1. Shared width 32 forces uniform dims at total 256;
  variable allocation lives on the per-token path.

`ChannelGate` multiplies each channel by a bounded function of itself
(`x * clip(a*x + b)`), 2 params per channel, init near identity
(a=0, b=1). Kill-on-no-gain: it ships only if A3 beats A2 at matched
cost.

## Variants

| Id | Dims | Pooling | Gate | Reads as |
| -- | ---- | ------- | ---- | -------- |
| RUNE-03-A | uniform or H1 | none | off | A0 baseline / A1 reallocation |
| RUNE-03-B | uniform or H1 | per_token | off | A2 pooling / P0 |
| RUNE-03-C | H1 | per_token | on | A3 gate |
| RUNE-03-D | uniform 32s | shared | off | P1 |

Arch version `0.3.0`, serialized spec carries arch id, arch version,
feature set, token layout (`token_dims`), pooling, gate flag,
quantization format, and fnv1a checksum. C++ rejects mismatches and
tampered payloads loudly.

## Budgets and the equal-budget rule

Total representation dims are the controlled variable: 128 / 192 /
256 / 320 legs, plus H1 (256 total, uneven). Any comparison claiming
an architecture gain must show equal representation budget, equal-ish
parameter budget (pool/gate params counted and reported), and an
equal inference benchmark. Per-eval reporting is per-metric only:
params, bytes, activation bytes, per-stage latency, NPS, validation
metrics, match strength with uncertainty. No composite score.

## Quantization posture

FP32 reference trains; INT16/INT8 cover embeddings via the existing
int32 accumulator path. Mixer/head stay fp32 until per-token
sensitivity analysis says which token needs more precision, and only
then do scale-granularity tests (global / per-layer / per-token /
per-group), kept small. Every precision change reports numerical
error, latency, NPS, size, and engine impact together.

## Explicit non-goals

No Transformer, Mamba/SSM, MoE, recurrence, self-supervised or
contrastive losses, bilinear blocks, giant heads, routing, or search.
Head shape stays frozen. Dataset and architecture are never redesigned
in the same main experiment.
