# Versioning (RUNE v0.12)

Four independent versions. Bumping one never implies the others.

- `RUNE-12 runtime spec` — this directory. Bumped when any
  normative semantics change. v12 adds the `RUNE-ATTN-SOFT`
  architecture (softmax-scaled mixer, `Softmax` IR op as a `Gate`
  alternative), the `value_swiglu` head (`arch_version` 0.3.0), and
  marks `RUNE-ATTN`, `RUNE-ATTN-DUAL`, `RUNE-ATTN-MH4`,
  `RUNE-REL-LITE`, `RUNE-MLP-S`, `RUNE-SFNN-C` frozen (load only).
  Readers accept `RUNE-10` and `RUNE-11` artifacts unchanged.
- `RUNE model format version` — `format` header field, currently 2.
  Bumped only for framing/header changes. Readers accept 1 and 2.
- `RUNE architecture version` — per arch id (`0.1.0`, `0.2.0`, ...).
  Bumped for weight-shape or forward changes of that arch only.
- `RUNE data schema version` — `.rune-data` store version, owned by
  the data engine, currently 2. Runtime only checks feature_version
  compatibility, never the full data schema.

`feature_version` (`grouped_hkav2_fullthreats_v02`) is independent of
all four and changes only with extraction semantics.
