# Go planes v02 (normative)

`feature_version` `go_planes_v02`. Supersedes `go_planes_v01` for new
artifacts. Readers accept v01 unchanged. v01 models never read v02
planes.

## State

Extended GRID: `<grid> <stm> [ko komi move_no pass_no]`.

- `grid`: rows joined by `/`, cells `./X/O`, sizes 9/13/19 only.
- `stm`: `b`/`X` black to move, `w`/`O` white to move.
- `ko`: `-` or square `0..n*n-1` forbidden by simple ko.
- `komi`: float `-50..50`, default `7.5`.
- `move_no`: `1..10000`, default `1`.
- `pass_no`: `0..500`, default `0`.

Missing trailing fields take defaults. Unknown sizes and stones
fail closed. `normalize` keeps `grid stm [ko]`.

## Planes (ResNet input, NCHW)

`PLANES_V02 = 8`, layout plane-major `p*n*n+sq`, float32 `0.0/1.0`
except plane 7 which broadcasts side to move:

| plane | meaning |
| ----- | ------- |
| 0 | own stones |
| 1 | opponent stones |
| 2 | empty |
| 3 | stones with <=1 liberty |
| 4 | stones with exactly 2 liberties |
| 5 | stones with >=3 liberties |
| 6 | ko square (single 1.0) |
| 7 | side to move broadcast |

Liberties are computed on 4-connected groups. Empty board yields
plane 2 all ones, planes 0/1/3/4/5/6 all zeros.

## Context (12 floats, all clamped 0..1)

Order: `stm, us_stones, them_stones, komi, move_no, pass_no, ko,
lib1, lib2, lib3, empty, phase`.

- `us/them/empty`: stone fractions over `n*n`.
- `komi`: `komi/15`.
- `move_no`: `move_no/200`.
- `pass_no`: `pass_no/2`.
- `ko`: `1.0` when ko present else `0.0`.
- `lib1/lib2/lib3`: fractions of own stones with `<=1`/`==2`/`>=3`
  liberties, `0.0` when no own stones.
- `phase`: `game_phase/2` with `game_phase` from move number:
  `move_no <= n*n/3` is 0, `<= 2*n*n/3` is 1, else 2.

## Token features (shared 8-token stack)

Same 9 groups and vocabs as v01:
`[361,361,361,64,64,361,361,64,18]`.
Sorted unique `(group,index)` list:

- group 0: `(ci*181+sq)%361` per stone, `ci` relative color.
- group 6: `sq%361` occupancy per stone.
- group 1: low-liberty stones (`<=2`), same key as group 0.
- group 5: atari stones (`==1`), `sq%361`.
- group 2: ko square `ko%361` when present.
- group 5 extra: `360-(ko%361)` when ko present.
- group 7: `stm`, `2+komi_bucket`, `10+move_bucket`,
  `16+min(pass_no,2)`, `19` when ko, `20+phase_bucket`.
  `komi_bucket = clamp01((komi+5)/20)*7.99` floored to `0..7`.
  `move_bucket = (move_no-1)//20` capped `0..5`.
  Phase bucket from move bucket: `<=1` is 0, `<=3` is 1, else 2.
- group 8: `min(atari,8)` plus `9+(ko%9)` or `9` without ko.

`phase_from_ids` reads max valid `20..22` in group 7, defaults 0.

## Rules (all three runtimes agree)

- Capture: stones with no liberties after placement are removed
  before suicide check. Opponent groups first, own group last.
- Suicide: placement leaving own group with zero liberties is
  illegal, even in corners.
- Ko: square stored as `new_ko` after a single-stone capture by a
  single-stone placement with one liberty is forbidden next move.
- Legal moves: every empty non-ko square passing capture/suicide
  plus pass `-1`. Empty board has `n*n+1` legal moves.
- Scoring: area `black-white-komi` with Tromp-Taylor territory
  (empty regions touching one color only).

## Parity

Python (`training/games/go_v02.py`), Rust (`crates/rune-runtime/src/go/`),
and C++ (`core/go/go_state.*`, `go_rules.*`, `go_planes_v02.*`,
`go_token_v02.*`) evaluate the golden states in
`spec/test-vectors/resnet/eval_v02.json` identically within
`1e-4` value and exact legal-move sets.
