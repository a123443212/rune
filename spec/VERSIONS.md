# Versioning (RUNE v0.11)

Four independent versions. Bumping one never implies the others.

- `RUNE-11 runtime spec` — this directory. Bumped when any
  normative semantics change. v11 adds the ResNet op registry
  (`StemConv`, `Conv2D`, `ResidualAdd`, `Relu`, `GlobalPool`,
  `Flatten`, `FeaturePlanes`, `PolicyLogits`, `Policy`), the dual
  value/policy head, the MCTS `GameState`/`PolicyNet` contract, and
  the `go` game (`go_planes_v01`). Readers accept `RUNE-10`
  artifacts unchanged.
- `RUNE model format version` — `format` header field, currently 2.
  Bumped only for framing/header changes. Readers accept 1 and 2.
- `RUNE architecture version` — per arch id (`0.1.0`, `0.2.0`, ...).
  Bumped for weight-shape or forward changes of that arch only.
- `RUNE data schema version` — `.rune-data` store version, owned by
  the data engine, currently 2. Runtime only checks feature_version
  compatibility, never the full data schema.

`feature_version` (`grouped_hkav2_fullthreats_v02`) is independent of
all four and changes only with extraction semantics.
