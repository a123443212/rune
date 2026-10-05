# RUNE IR v1 (normative)

IR version 1.0. Independent of model format version (2), architecture versions, data schema version (2).

Canonical op order: FeatureUpdate, AccumulatorUpdate, Tokenize, Q, K, V, Score, Bias, Gate, Mix, Residual, HeadH1, HeadH2, Value, WDL. Adaptive adds Route.

Each op declares inputs, outputs, shapes, dtype. Each tensor declares shape, dtype, layout, constant flag, lifetime [birth, death]. Each kernel entry declares kernel id, shape key, dtype, packing, ISA, fusion group. Memory declares arena bytes, alignment 32, buffer offsets, reuse strategy.

Verifier rejects: bad ir_version, missing op, unsupported isa/dtype/quant, tokens/dim out of range, malformed shape. No silent fallback.
