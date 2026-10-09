import json
import os
import struct

MAGIC = b"RUNEDATA"
FORMAT_VERSION = 1


def read_shard(path):
    with open(path, "rb") as f:
        blob = f.read()
    off = 0
    assert blob[off:off + 8] == MAGIC, "bad magic"
    off += 8
    (fver,) = struct.unpack_from("<I", blob, off)
    off += 4
    assert fver == FORMAT_VERSION, f"unsupported format version {fver}"
    (schema,) = struct.unpack_from("<I", blob, off)
    off += 4
    assert schema in (1, 2), f"unsupported schema {schema}"
    (flen,) = struct.unpack_from("<H", blob, off)
    off += 2
    feature_version = blob[off:off + flen].decode()
    off += flen
    compression = blob[off]
    off += 1
    assert compression in (0, 1), f"unknown compression {compression}"
    (count,) = struct.unpack_from("<Q", blob, off)
    off += 8
    (shard_id,) = struct.unpack_from("<I", blob, off)
    off += 4
    (shard_count,) = struct.unpack_from("<I", blob, off)
    off += 4
    (dlen,) = struct.unpack_from("<H", blob, off)
    off += 2
    dataset_id = blob[off:off + dlen].decode()
    off += dlen
    (raw_len,) = struct.unpack_from("<Q", blob, off)
    off += 8
    header_end = off
    (header_crc,) = struct.unpack_from("<I", blob, off)
    off += 4
    import zlib

    assert zlib.crc32(blob[:header_end]) & 0xFFFFFFFF == header_crc, "header crc mismatch"
    payload = blob[off:-4]
    (payload_crc,) = struct.unpack_from("<I", blob, len(blob) - 4)
    assert zlib.crc32(payload) & 0xFFFFFFFF == payload_crc, "payload crc mismatch"
    if compression == 1:
        import zlib as _z

        raw = _z.decompress(payload)
        assert len(raw) == raw_len, "decompressed length mismatch"
    else:
        raw = payload
    offsets = struct.unpack_from(f"<{count}Q", raw, 0)
    table = 8 * count
    records = []
    for k in range(count):
        p = table + offsets[k]
        (identity,) = struct.unpack_from("<Q", raw, p)
        p += 8
        phase, stm, pc, pw, q, imb, wdl, flags = struct.unpack_from("<8B", raw, p)
        p += 8
        (tvalue,) = struct.unpack_from("<f", raw, p)
        p += 4
        (tcp,) = struct.unpack_from("<i", raw, p)
        p += 4
        (game_hash,) = struct.unpack_from("<Q", raw, p)
        p += 8
        (ply,) = struct.unpack_from("<H", raw, p)
        p += 2
        (source_id,) = struct.unpack_from("<I", raw, p)
        p += 4
        (nfeat,) = struct.unpack_from("<H", raw, p)
        p += 2
        (nfen,) = struct.unpack_from("<H", raw, p)
        p += 2
        if schema >= 2:
            (active_round,) = struct.unpack_from("<I", raw, p)
            p += 4
            (sel_score,) = struct.unpack_from("<f", raw, p)
            p += 4
            (sel_method,) = struct.unpack_from("<B", raw, p)
            p += 1
        else:
            active_round, sel_score, sel_method = 0, 0.0, 0
        feats = []
        for _ in range(nfeat):
            (g, i) = struct.unpack_from("<HH", raw, p)
            p += 4
            feats.append((g, i))
        fen = raw[p:p + nfen].decode()
        records.append({
            "fen": fen, "identity": identity, "phase": phase, "stm": stm,
            "piece_count": pc, "pawn_count": pw, "queens": q,
            "imbalance_bucket": imb, "teacher_wdl": wdl,
            "has_teacher": bool(flags & 1), "perspective_stm": bool(flags & 2),
            "teacher_value": tvalue, "teacher_cp": tcp, "game_hash": game_hash,
            "ply": ply, "source_id": source_id, "features": feats,
            "active_round": active_round, "sel_score": sel_score,
            "sel_method": sel_method,
        })
    header = {"feature_version": feature_version, "compression": compression,
              "record_count": count, "shard_id": shard_id,
              "shard_count": shard_count, "dataset_id": dataset_id,
              "schema": schema}
    return header, records


def load_dataset(path):
    if os.path.isdir(path):
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        recs = []
        header = None
        for name in manifest["shard_files"]:
            h, r = read_shard(os.path.join(path, name))
            header = header or h
            recs.extend(r)
        return manifest, header, recs
    header, recs = read_shard(path)
    return None, header, recs


def to_torch_batch(records, group_widths=None, device="cpu"):
    import torch

    from training.features.python_features import VOCAB_SIZES

    dev = torch.device(device)
    if group_widths is None:
        group_widths = [0] * 9
        for r in records:
            counts = [0] * 9
            for g, _ in r["features"]:
                counts[g] += 1
            for g in range(9):
                group_widths[g] = max(group_widths[g], counts[g])
        group_widths = [max(1, w) for w in group_widths]
    n = len(records)
    ids, masks = [], []
    for g in range(9):
        ids.append(torch.zeros(n, group_widths[g], dtype=torch.long, device=dev))
        masks.append(torch.zeros(n, group_widths[g], dtype=torch.float32, device=dev))
    values, wdls = [], []
    for b, r in enumerate(records):
        per = {}
        for g, i in r["features"]:
            per.setdefault(g, []).append(i % VOCAB_SIZES[g])
        for g in range(9):
            for j, v in enumerate(per.get(g, [])[:group_widths[g]]):
                ids[g][b, j] = v
                masks[g][b, j] = 1.0
        values.append(r["teacher_value"] if r["has_teacher"] else 0.0)
        wdls.append(r["teacher_wdl"])
    return ids, masks, torch.tensor(values, device=dev), torch.tensor(wdls, dtype=torch.long, device=dev)


class ShardLoader:
    def __init__(self, path, batch_size=256, shuffle=False, seed=0, device="cpu"):
        _, _, self.records = load_dataset(path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        self.epoch = 0

    def __iter__(self):
        import random

        idx = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(idx)
        self.epoch += 1
        for s in range(0, len(idx), self.batch_size):
            chunk = [self.records[i] for i in idx[s:s + self.batch_size]]
            yield to_torch_batch(chunk, device=self.device)

    def __len__(self):
        return (len(self.records) + self.batch_size - 1) // self.batch_size
