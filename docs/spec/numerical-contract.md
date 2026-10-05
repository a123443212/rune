# Numerical contract (normative, per operation)

Exact mode means bit-identical float32 (or integer) equality.
Tolerant mode means bounded absolute difference with a stated eps.
No other comparison is allowed in differential testing.

## quantize(w, scale)

Exact. `q = half_away(w/scale)` then saturate to [-bound,bound].
Division is float32, rounding operates on the real quotient as if
infinite precision: implementations compute `w/scale` in float32
(or float64 with a final half-away step — both give the same q for
all in-range values; the tie case q.5 is decided away from zero).
NaN input quantizes to 0. +inf/-inf saturate to +-bound.

## dequantize(q, scale)

Exact up to one IEEE multiply: `w = float(q)*scale` in float32
round-to-nearest-even.

## clip01(x)

Exact. `x<0 -> 0`, `x>1 -> 1`, else x unchanged bit-identically,
including signed zero, which passes through unchanged (-0.0 stays
-0.0 in all three implementations, since -0.0 < 0.0 is false),
and NaN (propagate).

## hard_sigmoid(s)

Exact. `clamp(0.2*s+0.5,0,1)` evaluated as `t = fma(0.2,s,0.5)`
where available, else `(0.2*s)+0.5`, then clip. Because 0.2 is not
exact binary, both orderings are pinned: compute the product first
in float32, then add 0.5 in float32. FMA vs separate mul+add differ
in the last ulp, so: reference uses separate mul then add, SIMD may
use fma only inside tolerant mode with eps 1e-6. Exact vectors avoid
the boundary.

## matVec / matMul / QK^T

Tolerant, eps 1e-5 absolute per element for T*D <= 320 and K <= 64,
eps 2e-5 for larger dense heads, both stated for in-network
operating magnitudes (|mat| <= 0.1, |vec| <= 1.0, the range of
exported weights and clipped tokens). Accumulation order is
unspecified across kernels; summation is left-to-right over k in
the reference. Optimized kernels may reassociate and may use one
FMA per product-sum; those two together are the only allowed
tolerance sources. Outside operating magnitudes the bound scales
with the dot magnitude (measured: 3.82e-05 worst at 128x256 with
|mat|,|vec| <= 1.0, identical bit-for-bit in C++ and Rust lanes,
see docs/performance/rune-v10-benchmarks.md). End-to-end evals stay
within the head eps below because real magnitudes apply.

## tanh (value head)

Tolerant, eps 2e-6 absolute. Implementations may use std::tanh,
libm tanh, or a polynomial; all must stay within eps on [-8,8] and
saturate to +-1 outside.

## gate alpha residual `x + alpha*y`

Tolerant, eps 1e-6. Single float32 multiply then add, left to right.

## dynamic bias delta clamp

Exact clamp bounds +-0.25 inclusive. Inner dots are tolerant
(matVec rule); the clamp comparison itself is exact.

## routing compare

Exact. float32 `>=` with NaN -> refine. No epsilon around the
threshold; threshold bytes are part of the model.

## accumulator int32 sum

Exact. Signed 32-bit wrap is a load-time rejection (fan-in guard),
never a runtime behaviour to compare.

## Policy

One global epsilon is forbidden. Each stage declares exact or its
own eps in `docs/spec/numerical-contract.md` and in the diff tool
config. Widening an eps to hide a bug is a spec violation.
