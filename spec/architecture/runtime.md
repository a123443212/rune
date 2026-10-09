# Architecture specification (RUNE v0.10 runtime slice)

Canonical table. `architecture-spec.json` next to this file is the
machine-readable form; this text explains it.

## Supported runtime architectures

| arch id | tokens | dim | attention | bias | gate | head |
| ------- | ------ | --- | --------- | ---- | ---- | ---- |
| RUNE-SFNN | 8 | 32 | none | none | clip | sfnn 256->32 |
| RUNE-MLP | 8 | 32 | none | none | clip | value_wdl 128->32 |
| RUNE-ATTN | 8 | 32 | gated_linear | none | clip | value_wdl 128->32 |
| RUNE-ATTN-GAB | 8 | 32 | gated_linear | learned gab | clip | value_wdl 128->32 |
| RUNE-ATTN-MH4 | 8 | 32 | multi_head | per-head gab | clip | value_wdl 128->32 |
| RUNE-REL-02 | 6/8/10 | 24/32/40 | gated_relational | static/dynamic | clip/hard_sigmoid | value_wdl 128->32 |
| RUNE-03-<variant> | 8 | per-group token_dims | pooled | none | clip | dense head_h1->head_h2 |
| RUNE-04 | 8 | 8..64 uniform | adaptive cheap+refine | threshold routing | clip | cheap_hidden + ref_h1->ref_h2 |
| RUNE-05 | 8 | 8..64 uniform | adaptive + uncertainty | threshold routing | clip | RUNE-04 + uncertainty head |

## Mixer (relational / attention, normative order)

Per token i: `Q[i]=Wq*x[i]+bq`, `K[i]=Wk*x[i]+bk`, `V[i]=Wv*x[i]+bv`
with row-major matVec. Scores `S[a][b]=dot(Q[a],K[b])+gabS[a][b]`
plus optional dynamic bias `delta=dot(dynU[a],ctx)*dot(dynW[b],ctx)`
clamped to [-0.25,+0.25]. Gate elementwise: clip01 or
hard_sigmoid `clamp(0.2*s+0.5,0,1)`. Mix `Y=S@V` (S is TxT, V is TxD).
Residual `out=x+alpha*Y`.

No softmax, no scaling by sqrt(d), no layer norm in this generation.

## Multi-head mixer (RUNE-ATTN-MH4, normative order)

4 heads, head dim 8 (4x8=32). Per head h: `Q_h[i]=Wq_h*x[i]+bq_h`,
`K_h[i]`, `V_h[i]` with shapes [8,32]/[8]. Scores
`S_h[a][b]=dot(Q_h[a],K_h[b])+gab_h[a][b]`, gate clip01 elementwise,
`O_h=S_h@V_h`. Concat heads row-major per token to 32 dims,
`Y=Wo*concat+bwo`, residual `out=x+Y`. Tensors
`wq_h0..gab_h3, wo, bwo` in head order, then value head tensors
(plain or `_b0..2` bucketed).

## Head (normative order)

Flatten mixed tokens row-major to length T*D. `h1=clip01(W1*flat+b1)`,
`h2=clip01(W2*h1+b2)`, `value=tanh(dot(wvo,h2)+bvo)`,
`wdl=Wwdl*h2+bwdl` linear (logits, no softmax inside runtime).
Uncertainty/stability heads, when present, are extra linear rows
after wdl and must be compared separately.

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
