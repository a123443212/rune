# .rune-data Format v1

Little-endian. Versioned. Checksummed. No silent misreads.

## Shard file layout

```
offset  field
0       magic            8 bytes  "RUNEDATA"
8       format_version   u32      (=1)
12      schema_version   u32      (=1)
16      feature_version  u16 len + bytes
..      compression      u8       (0=raw, 1=deflate)
..      record_count     u64
..      shard_id         u32
..      shard_count      u32
..      dataset_id       u16 len + bytes
..      uncompressed_len u64      (payload bytes after decompression)
..      header_crc       u32      (crc32 over all header bytes above)
..      payload          (stored bytes: raw, or deflate stream)
..      payload_crc      u32      (crc32 over stored payload bytes)
```

Payload (after decompression) = offset table (`record_count` × u64
byte offsets into the record section) followed by records. Offsets
give O(1) random access without scanning.

## Record layout

Fixed 42-byte prefix, then variable sections:

```
identity        u64   (FNV-1a of canonical FEN; strict or normalized)
phase           u8    (0/1/2, same rule as Python game_phase)
stm             u8
piece_count     u8
pawn_count      u8
queens          u8
imbalance_bucket u8   (|white-black| material / 3, capped at 7)
teacher_wdl     u8
flags           u8    (bit0 has_teacher, bit1 perspective is side-to-move)
teacher_value   f32
teacher_cp      i32
game_hash       u64   (FNV-1a of game id)
ply             u16
source_id       u32
feat_count      u16
fen_len         u16
features        feat_count × (u16 group, u16 index)
fen             fen_len bytes
```

Schema evolution: `format_version`/`schema_version` bump on any
layout change; readers reject unknown versions loudly (verified by
`rejects_bad_magic`/tamper tests). FEN is stored per record so any
future feature version can re-derive features without the source PGN.

## Manifest (`manifest.json` per dataset dir)

`dataset, schema, feature_version, shards, total_records, hash`
(FNV over shard hashes + dataset id + seed), `compression, seed,
shard_files`. Two runs with the same manifest hash used the same
bytes — this is the reproducibility contract training records rely on.

## Python bridge (`tools/data_bridge/reader.py`)

Pure-Python reader (struct + zlib, no new deps): header/CRC
validation, `to_torch_batch` (pre-extracted features → ids/masks,
widths derived from data — fixed guesses truncate group 6, which
needs width 34), `ShardLoader` batch iterator. Verified
bit-identical to `RuneDataset` collate on 512 shared records.
Epoch delivery ≈ Python collate (both Python-loop-bound); the
pyo3 contiguous-buffer path is the documented follow-up, not a claim.
