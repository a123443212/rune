# RUNE v0.10 final report

## 1. Semantic parity?

Yes for the shipped slice. Features, accumulator
(refresh==incremental), tokens, Q/K/V, scores, gates, mixed,
head, value/WDL all match across Python/C++/Rust within
per-stage tolerances; exact stages (features, quant, routing,
clip) are bit-identical. Max C++↔Rust value diff on 9
positions is 4.9e-07.

## 2. Same `.rune`?

Yes. Format 2 files load in all three with the same
`model_hash` (FNV-1a64 over payload). No `-cpp`/`-rust` split.
Legacy format 1 still loads.

## 3. Same numerical behavior?

Yes within contract. Half-away rounding is pinned (C `lround`,
Rust `round`, Python floor/ceil — `np.round` banned). Clip,
saturation, int32 accumulation, threshold `>=`, NaN→refine are
exact. MatVec/head/tanh are tolerant with stated eps.

## 4. What is not bit-exact and why?

MatVec/matMul (8-way reassociation + one FMA per product-sum in
SIMD lanes), tanh libm differences, alpha residual ordering.
Measured: SIMD lanes differ from scalar by 1.67e-06 worst at
in-network magnitudes, 3.81e-05 worst at out-of-range O(1)
magnitudes — identical digit-for-digit in C++ and Rust, so it
is an algorithm property, not a language one. Each stage keeps
its own eps; widening eps to hide bugs is a spec violation
(the contract now states operating magnitudes explicitly).

## 5–6. How fast is Rust vs C++ and vice versa?

Scalar vs scalar, release, same hardware/model: Rust ~64-71us,
C++ ~63-75us — a tie both ways across runs. SIMD vs SIMD:
Rust ~15-18us, C++ ~19-20us — also a tie within machine noise
(±20% at 30-40ms totals); no language verdict is drawn from
either. The speedup belongs to the kernel: ~3.7x C++, ~4.4x
Rust, from the same AVX2+FMA matVec both implement
bit-identically. Debug Rust is 12x slower and irrelevant.

## 7. Memory?

No advantage claimed. Both hold one immutable model plus
per-thread evaluator state. Rust clones tables once per
evaluator; C++ shares pointers. Footprint difference is noise
at v0.10 sizes.

## 8. Data engine bottleneck?

None found at v0.10 scale, so no streaming/mmap rewrite was
made. Data/runtime separation is now structural
(`rune-runtime` vs `rune-data`), ready for future profiles.
Repaid debt: 7 legacy integration tests were failing on a
fixture (`two_games.pgn`) that was never committed — fixed with
a real 2-game fixture, 9/9 green, CI restored to full legacy
workspace tests.

## 9. SIMD bottleneck?

Head (128×256 + 32×128) dominates mixer at 8×32. AVX2 matVec
helps the head most (~7x: 30-36us → 3.6-5.9us); QKᵀ stays scalar
in both. Measured in `rune-v10-benchmarks.md`, including the
adaptive cascade cost model (cheap 7.7us + rate × refine 45.1us).

## 10. Serialization stable?

Yes for shipped archs/precisions. Every fixture carries
`model_hash` verified on load; malformed inputs fail closed
(fuzzed in Rust loader tests, guarded in C++).

## 11. Did diff find a real bug?

Yes, two. A global-epsilon prototype hid a head transpose slip;
per-stage diff caught it. Quantization `.5` ties also
diverged (`np.round` vs `lround`) before the half-away pin.
Third, found while repaying v0.10 debts: the Rust evaluator
silently ran RUNE-04 refinement weights as a plain attention
net (all tensor names happened to match). Fixed with an arch
allowlist; `model-info` still loads everything, evaluation of
unimplemented archs fails closed.

## 12. Is the spec a real source of truth?

Yes. `spec/` decides disagreements; `architecture-spec.json`
is checked in; vectors are the oracle. Three implementations
already defer to it.

## 13. Rust role going forward?

Secondary + research runtime + data infra + cross-validation.
Production stays C++ until Rust proves a measured win with
SIMD. No rewrite pressure.

## 14. C++ next?

Pool the head matVec under AVX2 fully, then re-run the
transparent bench. No kernel duplication without profiles.

## 15. What is frozen for v1.0?

Feature IDs + version, tensor layout, quant rules, model
framing + hash, routing comparison, tolerance policy, vector
format. Architecture table may still grow new ids, but
existing ids are frozen.
