import json
import os


def shard_key(record):
    game = record.get("game", "go")
    state = record.get("state", "")
    return game + "\x00" + state.split()[0]


def validate_record(game, record):
    try:
        state = record.get("state", "")
        game.legal(state)
        game.context(state)
    except Exception:
        return False
    mv = record.get("move", None)
    if mv is not None and int(mv) not in game.legal(state):
        return False
    return True


def filter_record(game, record, min_move=1, max_move=10000):
    try:
        parts = record.get("state", "").split()
        move_no = int(parts[4]) if len(parts) > 4 else 1
    except Exception:
        return False
    if move_no < min_move or move_no > max_move:
        return False
    return True


def dedup_records(records):
    seen = set()
    out = []
    for r in records:
        k = shard_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def write_shards(records, out_dir, num_shards=2):
    from training.export.export import fnv1a
    os.makedirs(out_dir, exist_ok=True)
    shards = [[] for _ in range(max(1, num_shards))]
    for i, r in enumerate(records):
        shards[i % len(shards)].append(r)
    manifest_shards = []
    for i, part in enumerate(shards):
        path = os.path.join(out_dir, f"shard_{i:04d}.jsonl")
        with open(path, "w") as f:
            for r in part:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        with open(path, "rb") as f:
            h = format(fnv1a(f.read()), "016x")
        manifest_shards.append({"path": os.path.basename(path), "count": len(part), "hash": h})
    total = sum(s["count"] for s in manifest_shards)
    manifest = {
        "game": "go",
        "shards": manifest_shards,
        "positions": total,
        "teacher": records[0].get("teacher", "") if records else "",
        "feature_version": records[0].get("feature_version", "") if records else "",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def shard_pipeline(game, records, out_dir, num_shards=2, min_move=1, max_move=10000):
    valid = [r for r in records if validate_record(game, r)]
    kept = [r for r in valid if filter_record(game, r, min_move, max_move)]
    uniq = dedup_records(kept)
    manifest = write_shards(uniq, out_dir, num_shards)
    manifest["validated"] = len(valid)
    manifest["kept"] = len(kept)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
