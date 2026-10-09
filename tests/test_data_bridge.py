import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

torch.manual_seed(0)

from tools.data_bridge.reader import ShardLoader, load_dataset, read_shard, to_torch_batch

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "tiny.rune-data")


def test_read_shard_header():
    header, recs = read_shard(FIXTURE)
    assert header["record_count"] == len(recs) == 28
    assert header["shard_count"] == 1
    assert header["feature_version"] == "grouped_hkav2_fullthreats_v02"
    assert all(len(r["features"]) > 0 for r in recs)
    assert all(r["perspective_stm"] for r in recs)


def test_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.rune-data"
    bad.write_bytes(b"GARBAGE" + b"\x00" * 64)
    with pytest.raises(AssertionError):
        read_shard(str(bad))


def test_bridge_matches_python_collate():
    from training.datasets import pipeline as P
    from training.datasets.rune_dataset import make_loader
    from training.features.python_features import extract_features

    header, recs = read_shard(FIXTURE)
    by_fen = {}
    for r in recs:
        by_fen.setdefault(r["fen"], []).append(r)
    fens = sorted(by_fen.keys())
    py_pool = [{"fen": f, "value": 0.0, "wdl": 1, "game_id": f"g{i}", "ply": 8}
               for i, f in enumerate(fens)]
    kept, _ = P.clean_pipeline(py_pool)
    rs_recs = [by_fen[r["fen"]][0] for r in kept]
    rb = to_torch_batch(rs_recs)
    pb = next(iter(make_loader(kept, batch_size=4096, shuffle=False, seed=0)[0]))
    assert len(rb[0][0]) == len(pb[0][0])
    for g in range(9):
        assert torch.equal(pb[0][g], rb[0][g]), f"ids group {g}"
        assert torch.equal(pb[1][g], rb[1][g]), f"masks group {g}"


def test_shard_loader_batches():
    n = 0
    for ids, masks, values, wdls in ShardLoader(FIXTURE, batch_size=8):
        n += len(values)
        assert len(ids) == 9 and len(masks) == 9
    assert n == 28
