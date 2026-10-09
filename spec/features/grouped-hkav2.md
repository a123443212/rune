# Feature specification — grouped_hkav2_fullthreats_v02

`feature_version = "grouped_hkav2_fullthreats_v02"`.
Any model or dataset carrying another string is incompatible and
must be rejected, never coerced.

v02 changes over v01: side-to-move-relative orientation (mirror
fold + us/them colors), en-passant file token, halfmove bucket
token. Vocab sizes are unchanged. v01 files are rejected by v02
loaders.

## Orientation (normative)

Let `us` be the side to move (`"w"` when stm==0 else `"b"`).
Let `kus` be the first square holding a `us` king scanning 0..63,
or -1 when absent. `fold = 1` when `kus >= 0` and `file(kus) < 4`,
else `0`. `rel(sq) = sq` when fold is 0, else
`rank(sq)*8 + (7-file(sq))`.

Color index is relative: `ci = 0` when the piece color equals `us`,
else `1`. The `stm` bit stays absolute (it names which absolute
color is `us`).

## Groups

| group | name | vocab | index meaning (`S = rel(sq)`) |
| ----- | ---- | ----- | ----------------------------- |
| 0 | pawn_structure | 256 | 0..127: ci*64+S per pawn square; 128..159: file(S)*4+rank(S)/2 file buckets |
| 1 | king_zone | 256 | 0..127: ci*64+S king square; 128..191: 128+ci*32+rel(nsq)/2 for 8 neighbours |
| 2 | minor_pieces | 256 | 0..127: relative knights ci*64+S; 128..255: bishops 128+ci*64+S |
| 3 | rooks | 128 | ci*64+S, indices >= 128 unused |
| 4 | queens | 128 | ci*64+S, indices >= 128 unused |
| 5 | threats | 512 | 0..383: victim_type*64+rel(vsq) per attacked victim; 384..447: 384+file_rel(asq)*8+rank(asq) attacker coarse |
| 6 | mobility | 512 | 0..383: piece_type*64+S occupancy; 384..447: 384+stm*32+min(pseudo_legal_count,31) |
| 7 | global | 64 | 0: stm; 2+castle_mask (0..15 -> 2..17); 18+bucket (piece_count-2)/2 clamped 0..15; 34+phase (0..2); 37+file_rel(ep) when ep set (37..44); 45+min(halfmove/20,4) (45..49) |
| 8 | pawn_pairs | 4560 | unordered pawn pairs, triangular index (see below) |

`piece_type` order is pawn=0 knight=1 bishop=2 rook=3 queen=4 king=5.
`sq = rank*8+file`, file 0=a, rank 0=first rank. Colors: white=0 black=1.

## Extraction order and dedup

1. Compute `us`, `kus`, `fold` first.
2. Iterate squares 0..63 ascending, emitting per-piece entries in the
   fixed order pawn, king, knight, bishop, rook, queen, occupancy(group 6),
   with relative colors and folded squares.
3. Threat loop: victim square outer 0..63, attacker square inner 0..63,
   skipping empty and same-color pairs, testing `piece_attacks`.
   Same-color comparison uses absolute colors.
4. Mobility: pseudo-legal move count of side to move, promotions count
   as 4, en-passant captures count, castling counts iff pieces between
   are empty (legality beyond that is not checked — count is pseudo-legal
   by definition).
5. Global entries, including EP file and halfmove bucket.
6. Sort ascending by (group,index), dedup exact duplicates.

Output is a sorted unique list of (group:u8, index:u16). Out-of-vocab
indices must never be produced; loaders must validate group < 9 and
index < vocab[group].

## Pawn pairs (group 8, normative)

Collect every pawn on ranks 2..7 (squares 8..55). For each pawn take
`pid = ci*48 + (S-8)` with relative color `ci` and folded square `S`.
For every unordered pair with `a < b` emit `(8, b*(b-1)//2 + a)`.
Pawns off ranks 2..7 contribute no pairs. Vocabulary is 4560
(`95*94//2+94 = 4559` max). Pairs are color-aware through `ci`:
own-own, enemy-enemy and tension pairs all count.

Group 8 folds into token 0 in every token layout: accumulators add
group-8 rows to the token-0 slot, never to a ninth token.

## Attack definition (normative)

- Pawn: one step diagonally forward (white +rank, black -rank).
- Knight: (1,2) leaps. King: Chebyshev distance <= 1.
- Sliders: bishop diagonal only, rook straight only, queen both.
  Path squares strictly between from and target must all be empty.
  Endpoints are not block-tested here; occupancy is handled by the
  empty/same-color skips above.

## Phase

Count non-pawn non-king pieces N. Phase = 0 if N>=12, 1 if N>=6,
else 2.

## Context vector (normative, CONTEXT_DIM=12)

`[stm, phase/2, pawns/16, minors/8, rooks/4, queens/2, shield/8,
total/32, castle_mask/15, ep_set, in_check, halfmove/100]`
with every entry clamped to [0,1]. Counts are absolute over both
colors. `shield` counts own non-king pieces around the `us` king.
`ep_set` is 1 when an en-passant square is set. `in_check` is 1
when any enemy piece attacks `kus` (0 when the king is absent).

## Test coverage required

Golden vectors must include quiet move, capture, promotion,
castling rights change, en passant square set, king move, a
piece exchange, black to move (mirror check), and a fifty-move
clock position, each with full sorted feature lists.
