# Quantization rules (normative)

Supported precisions: `fp32`, `int8`, `int16`. Precision is a model
property (`quantization` header field) applying to embedding tables
only. Mixer and head weights are always float32.

## Scale

Per-group symmetric scale, one float32 per group g:

```text
maxAbs[g] = max |w| over embedding table g
scale[g]  = maxAbs[g] / bound, bound = 127 (int8) or 32767 (int16)
scale[g]  = 1.0 if maxAbs[g] <= 0
```

Scales are stored in header `scales: {"emb0": s0, ...}` as decimal
floats and re-parsed with strtod-grade precision. C++ and Rust must
use the stored scale for dequantisation, never recompute it.

## Quantize

```text
q = round_half_away_from_zero(w / scale[g])
q = clamp(q, -bound, +bound)
```

`round_half_away_from_zero` is normative: 0.5 -> 1, -0.5 -> -1,
1.5 -> 2. This matches C `lround` and Rust `f32::round`. Python
`np.round` (banker's) is NOT conforming; the reference generator
uses `floor(x+0.5)` for positives and `ceil(x-0.5)` for negatives.

Stored dtype: int8 (`i8`) or int16 little-endian (`<i2`).

## Dequantize

```text
w' = float(q) * scale[g]
```

float() is exact for int8/int16 magnitudes. Accumulate in int32,
convert to float32 once per (group,dim) cell, multiply by scale.

## Accumulator contract

- int8/int16 refresh: int32 zero, add signed rows for added features,
  subtract for removed features.
- tokens(): dequant then clip01 per element.
- Dequantised embedding tables (for testing) must equal
  `float(q)*scale` bit-identically up to one float32 rounding of the
  multiply; the multiply itself is IEEE-754 round-to-nearest-even.

## Golden coverage

Each precision needs vectors for: known table, quantized bytes,
scales, a startpos accumulator, and expected tokens. int8 and int16
boundaries (+-127, +-32767, zero table -> scale 1.0) are mandatory.
