# Versioning (RUNE v0.10)

Four independent versions. Bumping one never implies the others.

- `RUNE-10 runtime spec` — this directory. Bumped when any
  normative semantics change.
- `RUNE model format version` — `format` header field, currently 2.
  Bumped only for framing/header changes. Readers accept 1 and 2.
- `RUNE architecture version` — per arch id (`0.1.0`, `0.2.0`, ...).
  Bumped for weight-shape or forward changes of that arch only.
- `RUNE data schema version` — `.rune-data` store version, owned by
  the data engine, currently 2. Runtime only checks feature_version
  compatibility, never the full data schema.

`feature_version` (`grouped_hkav2_fullthreats_v02`) is independent of
all four and changes only with extraction semantics.
