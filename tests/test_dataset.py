import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.datasets import pipeline as P

BASE = [
    {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "wdl": 1, "game_id": "g1", "ply": 0},
    {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "value": 0.12, "wdl": 1, "game_id": "g1", "ply": 1},
    {"fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "value": 0.3, "wdl": 1, "game_id": "g2", "ply": 20},
    {"fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "value": -0.5, "wdl": 2, "game_id": "g3", "ply": 60},
    {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "value": 0.1, "wdl": 1, "game_id": "g1", "ply": 0},
    {"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "value": 0.0, "wdl": 1, "game_id": "gx", "ply": 0},
    {"fen": "P7/8/8/8/8/8/8/K6k w - - 0 1", "value": 0.0, "wdl": 1, "game_id": "gx", "ply": 0},
]


def test_integrity_and_dedup():
    kept, stats = P.filter_integrity(BASE)
    assert stats["rejected"] == 2
    kept2, s2 = P.deduplicate(kept)
    assert s2["duplicates"] == 1
    assert len(kept2) == 4


def test_quality_and_balance():
    recs = [dict(r, ply=10) for r in BASE[:4]]
    kept, _ = P.quality_filter(recs, min_ply=4)
    assert len(kept) == 4
    kept, info = P.phase_balance(recs * 3, max_ratio=1.0)
    assert info["kept"] <= len(recs * 3)


def test_split_no_game_leak():
    recs = []
    for g in range(20):
        for p in range(5):
            recs.append({"fen": BASE[2]["fen"], "value": 0.0, "wdl": 1, "game_id": f"g{g}", "ply": p})
    splits = P.split_by_game(recs)
    games = {k: {r["game_id"] for r in v} for k, v in splits.items()}
    assert not (games["train"] & games["val"])
    assert not (games["train"] & games["test"])
    assert not (games["val"] & games["test"])
    assert sum(len(v) for v in splits.values()) == len(recs)


def test_sampling_modes():
    recs = []
    for i in range(60):
        r = dict(BASE[i % 4])
        r = dict(r, game_id=f"g{i}")
        r["teacher_value"] = 0.5 if i % 2 == 0 else -0.5
        r["student_value"] = 0.0
        recs.append(r)
    a = P.sample_random(recs, 20, seed=0)
    b = P.sample_stratified(recs, 21, seed=0)
    c = P.sample_disagreement(recs, 20, disagreement_frac=0.5, seed=0)
    assert len(a) == 20 and len(b) == 21 and len(c) == 20


def test_clean_pipeline_end_to_end(tmp_path):
    out = str(tmp_path / "clean.jsonl")
    kept, stats = P.clean_pipeline(BASE)
    P.save_jsonl(out, kept)
    back = P.load_jsonl(out)
    assert len(back) == len(kept)
    assert stats["dedup"]["duplicates"] >= 1
