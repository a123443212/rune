# Games (multi-game contract)

RUNE is game-agnostic below the feature boundary and game-specific
above it. The network stack (embeddings, mixer, head) consumes only
`(group, index)` feature lists plus a context vector; it never sees
rules. Each game owns its board, notation, extractor, and teacher.

## Registered games

| `game` | notation | feature_version | groups | context dim |
| ------ | -------- | --------------- | ------ | ----------- |
| `chess` | FEN | `grouped_hkav2_fullthreats_v02` | 9 | 17 |
| `shogi` | SFEN | `shogi_raw_v01` | 9 | 12 |

`shogi_raw_v01` is a raw adapter: board pieces by relative color,
hands by count, checkers, occupancy, and a global group. It shares
the 8-token layout with chess. Semantic upgrades (drops-aware
structure, king-zone analogues) promote through new versions, never
by editing this one.

## Rules

- Every `.rune` file carries `game`. Absent means `chess`.
- `game` is part of the header cover (`header_hash`) like any other
  header field. Unknown `game` fails closed on load.
- A runtime built for one game rejects other games' models, even
  when architecture and shapes match. Native extractors ship per
  game: chess and shogi both evaluate end-to-end in Python, Rust,
  and C++, proven by the shared golden fixture
  (`spec/test-vectors/shogi/eval.json` plus
  `spec/test-vectors/models/shogi-mlp-fp32.rune`).
- Records carry `game` plus `state` (notation string). Legacy
  records with `fen` and no `game` mean chess.
- New games start as raw adapters over `(square, piece)`-style
  features, then earn semantic extractors through matched-condition
  experiments like any other change.
