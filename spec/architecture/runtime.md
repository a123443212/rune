# Architecture specification (RUNE v0.10 runtime slice)

Canonical table. `architecture-spec.json` next to this file is the
machine-readable form; this text explains it.

## Supported runtime architectures

| arch id | tokens | dim | attention | bias | gate | head | status |
| ------- | ------ | --- | --------- | ---- | ---- | ---- | ------ |
| RUNE-SFNN | 8 | 32 | none | none | clip | sfnn 256->32 | active |
| RUNE-MLP | 8 | 32 | none | none | clip | value_wdl 128->32 | active |
| RUNE-ATTN-GAB | 8 | 32 | gated_linear | learned gab | clip/hard_sigmoid/screlu | value_wdl 128->32 | active |
| RUNE-ATTN-SOFT | 8 | 32 | softmax_scaled | learned gab | softmax | value_wdl 128->32 | active |
| RUNE-REL-02 | 6/8/10 | 24/32/40 | gated_relational | static/dynamic | clip/hard_sigmoid/screlu | value_wdl 128->32 | active |
| RUNE-ATTN | 8 | 32 | gated_linear | none | clip/hard_sigmoid/screlu | value_wdl 128->32 | frozen, load only |
| RUNE-ATTN-MH4 | 8 | 32 | multi_head | per-head gab | clip/hard_sigmoid/screlu | value_wdl 128->32 | frozen, load only |
| RUNE-MLP-S | 8 | 32 | none | none | clip | value_wdl 64->16 | frozen, load only |
| RUNE-SFNN-C | 8 | 32 | none | none | clip | value_wdl 128->16 | frozen, load only |
| RUNE-ATTN-DUAL | 8 | 32 | gated_linear_x2 | none | clip/hard_sigmoid/screlu | value_wdl 128->32 | frozen, load only |
| RUNE-REL-LITE | 6 | 24 | gated_relational | static | clip/hard_sigmoid/screlu | value_wdl 128->32 | frozen, load only |
| RUNE-03-<variant> | 8 | per-group token_dims | pooled | none | clip | dense head_h1->head_h2 | frozen, screening only |
| RUNE-04 | 8 | 8..64 uniform | adaptive cheap+refine | threshold routing | clip | cheap_hidden + ref_h1->ref_h2 | frozen, screening only |
| RUNE-05 | 8 | 8..64 uniform | adaptive + uncertainty | threshold routing | clip | RUNE-04 + uncertainty head | frozen, screening only |

## Mixer (relational / attention, normative order)

Per token i: `Q[i]=Wq*x[i]+bq`, `K[i]=Wk*x[i]+bk`, `V[i]=Wv*x[i]+bv`
with row-major matVec. Scores `S[a][b]=dot(Q[a],K[b])+gabS[a][b]`
plus optional dynamic bias `delta=dot(dynU[a],ctx)*dot(dynW[b],ctx)`
clamped to [-0.25,+0.25]. Gate elementwise per model gate:
clip01, hard_sigmoid `clamp(0.2*s+0.5,0,1)`, or screlu `clip01(s)^2`. Mix `Y=S@V` (S is TxT, V is TxD).
Residual `out=x+alpha*Y`.

## Softmax mixer (RUNE-ATTN-SOFT, normative order)

Per token i: `Q[i]=Wq*x[i]+bq`, `K[i]=Wk*x[i]+bk`, `V[i]=Wv*x[i]+bv`
with row-major matVec. Scores `S[a][b]=dot(Q[a],K[b])/sqrt(D)+gab[a][b]`
with `D=32`, `scale=1/sqrt(D)` in float32. Weights `W[a]=softmax(S[a])`
rows (max-subtraction, `exp`, normalize; uniform row on non-finite sum).
Mix `Y=W@V`. Residual `out=x+Y` (`alpha=1.0`). Header `gate` is
`"softmax"` and is ignored by gated paths. Export order matches
RUNE-ATTN-GAB (`wq,bq,wk,bk,wvv,bvv,gab` then head).

## Dual mixer (RUNE-ATTN-DUAL, normative order)

Two gated_linear layers in sequence. Layer 1 uses `wq,bq,wk,bk,wvv,bvv,gab`
applied per mixer order with `alpha=1.0` and no context. Layer 2 uses
`wq2,bq2,wk2,bk2,wvv2,bvv2,gab2` applied to layer 1 output with same gate
and `alpha=1.0`. Tensors `gab,gab2` are `[8,8]` zeros when unused.
Export order lists layer 1 tensors then layer 2 tensors then head.

## Lite relational (RUNE-REL-LITE, normative order)

Same mixer and head order as RUNE-REL-02 with fixed `tokens=6, dim=24`,
static `gabS` only, no `dynU/dynW`. Context input is accepted but ignored
for bias. Head is `value_wdl 128->32` on flattened `6*24=144` inputs.

## Multi-head mixer (RUNE-ATTN-MH4, normative order)

4 heads, head dim 8 (4x8=32). Per head h: `Q_h[i]=Wq_h*x[i]+bq_h`,
`K_h[i]`, `V_h[i]` with shapes [8,32]/[8]. Scores
`S_h[a][b]=dot(Q_h[a],K_h[b])+gab_h[a][b]`, gate elementwise per model gate,
`O_h=S_h@V_h`. Concat heads row-major per token to 32 dims,
`Y=Wo*concat+bwo`, residual `out=x+Y`. Tensors
`wq_h0..gab_h3, wo, bwo` in head order, then value head tensors
(plain or `_b0..2` bucketed).

## Head (normative order)

Flatten mixed tokens row-major to length T*D. Single path (default,
`head_pair=false`): `h1=clip01(W1*flat+b1)`, `h2=clip01(W2*h1+b2)`,
`value=tanh(dot(wvo,h2)+bvo)`, `wdl=Wwdl*h2+bwdl` linear (logits, no
softmax inside runtime).

Pair path (opt-in, `head_pair=true`, `head="value_wdl_pair"`,
`arch_version` 0.2.0 / 0.2.1 for REL-02): `pre=W1*flat+b1` (no clip),
`c=clip01(pre)`, `s=c*c` (screlu, exact `c*c` float32
round-to-nearest-even per numerical contract), `h1pair=concat(c,s)`
row-major `[clip|sqr]` length 2*H1, `h2=clip01(W2*h1pair+b2)` where
`W2` is `[H2, 2*H1]`, then value/wdl as above. Loaders infer the path
from `w2` element count (`H2*H1` single vs `H2*H1*2` pair) and must
reject any other width. Both paths are bit-deterministic given
identical input bytes; cross-path equality is not required.
Uncertainty/stability heads, when present, are extra linear rows
after wdl and must be compared separately.

SwiGLU path (opt-in, `head="value_swiglu"`, `arch_version` 0.3.0,
incompatible with `head_pair`): `g=Wgate*flat+bgate`,
`u=Wup*flat+bup`, `h=silu(g)*u` with `silu(x)=x/(1+exp(-x))`,
`h2=clip01(W2*h+b2)`, then value/wdl as above. Loaders infer the path
from tensor presence (`wgate`/`wup`): swiglu heads carry
`wgate,bgate,wup,bup,w2,b2,wvo,bvo,wwdl,bwdl` (or `wv`/`bv` aliases on
RUNE-MLP/RUNE-SFNN) with optional `_b{b}` bucket suffixes.

## Head buckets (normative)

A model carries 1 or 3 heads (`head_buckets` header field, default
1). With 3 buckets, bucket `b` is selected by game phase
(`phase = 0/1/2` from the feature spec): `b = clamp(phase, 0, 2)`.
Bucket `b` owns full head tensors named `w1_b{b}`, `b1_b{b}`,
`w2_b{b}`, `b2_b{b}`, `wvo_b{b}`, `bvo_b{b}`, `wwdl_b{b}`, `bwdl_b{b}`,
applied in the normative head order above. Models without bucketed
tensors behave as bucket count 1 shared across phases. Export order
places bucketed head tensors after the arch tensors in bucket order
`b = 0, 1, 2`. Bucket selection is part of evaluation: same weights
in all buckets must reproduce single-head outputs exactly.

## Routing (RUNE-04/05, normative)

Cheap path always runs. Difficulty score from cheap pooling is
compared against thresholds with `>=` on float32: score >= tHigh ->
refine; score < tLow (when present) -> accept cheap; else refine iff
score >= threshold. NaN score must take the refine path. Routing
decision must be bit-deterministic given identical input bytes.

## Export order

`emb0..emb8` then arch tensors in `export_order` per arch id,
then bucketed head tensors (`*_b0`, `*_b1`, `*_b2`) when
`head_buckets` is 3.
Order is part of the file hash. Runtimes must reject out-of-order
or missing tensor names.
