# Feature specification — grouped_hkav2_fullthreats_v01

`feature_version = "grouped_hkav2_fullthreats_v01"`.
Any model or dataset carrying another string is incompatible and
must be rejected, never coerced.

## Groups

| group | name | vocab | index meaning |
| ----- | ---- | ----- | ------------- |
| 0 | pawn_structure | 256 | 0..127: color*64+sq per pawn square; 128..159: file*4+rank/2 file buckets |
| 1 | king_zone | 256 | 0..127: color*64+sq king square; 128..191: 128+color*32+nsq/2 for 8 neighbours |
| 2 | minor_pieces | 256 | 0..127: white/black knight color*64+sq; 128..255: bishop 128+color*64+sq |
| 3 | rooks | 128 | color*64+sq, indices >= 128 unused |
| 4 | queens | 128 | color*64+sq, indices >= 128 unused |
| 5 | threats | 512 | 0..383: victim_type*64+vsq per attacked victim; 384..511: 384+file*8+rank attacker coarse |
| 6 | mobility | 512 | 0..383: piece_type*64+sq occupancy; 384..447: 384+stm*32+min(pseudo_legal_count,31) |
| 7 | global | 64 | 0: stm; 2+castle_mask (0..15 -> 2..17); 18+bucket (piece_count-2)/2 clamped 0..15; 34+phase (0..2) |

`piece_type` order is pawn=0 knight=1 bishop=2 rook=3 queen=4 king=5.
`sq = rank*8+file`, file 0=a, rank 0=first rank. Colors: white=0 black=1.

## Extraction order and dedup

1. Iterate squares 0..63 ascending, emitting per-piece entries in the
   fixed order pawn, king, knight, bishop, rook, queen, occupancy(group 6).
2. Threat loop: victim square outer 0..63, attacker square inner 0..63,
   skipping empty and same-color pairs, testing `piece_attacks`.
3. Mobility: pseudo-legal move count of side to move, promotions count
   as 4, en-passant captures count, castling counts iff pieces between
   are empty (legality beyond that is not checked — count is pseudo-legal
   by definition).
4. Global entries.
5. Sort ascending by (group,index), dedup exact duplicates.

Output is a sorted unique list of (group:u8, index:u16). Out-of-vocab
indices must never be produced; loaders must validate group < 8 and
index < vocab[group].

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

## Test coverage required

Golden vectors must include quiet move, capture, promotion,
castling rights change, en passant square set, king move, and a
piece exchange, each with full sorted feature lists.
