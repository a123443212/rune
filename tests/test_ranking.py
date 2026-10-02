import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.datasets.rune_dataset import RuneDataset
from training.datasets.siblings import pairs_to_batch
from training.experiments.ranking_ablation import RankingAblation
from training.losses.composite import CompositeLoss

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]


def test_composite_toggle():
    fn = CompositeLoss(value=True, wdl=True, ranking=False)
    v = torch.tensor([0.5, -0.5])
    w = torch.randn(2, 3)
    out = fn(v, w, v, torch.tensor([0, 2]))
    assert out["total"].item() > 0
    assert out["rank"].item() == 0.0
    fn1 = CompositeLoss(value=True, wdl=False, ranking=True, lambda_rank=1.0)
    out1 = fn1(v, w, v, torch.tensor([0, 2]),
               rank=(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0])))
    assert abs(out1["rank"].item() - 0.05) < 1e-6
    assert out1["wdl"].item() if "wdl" in out1 else True


def test_pairs_batch_shapes():
    recs = [{"fen": f, "value": 0.0, "wdl": 1} for f in FENS * 4]
    ds = RuneDataset(recs)
    pairs = [{"parent": FENS[0], "child_a": FENS[0], "child_b": FENS[1],
              "sign": 1.0, "margin": 0.3, "teacher_id": "t", "teacher_a": 0.2,
              "teacher_b": -0.1}]
    ids, masks, a, b, signs, weights = pairs_to_batch(ds, pairs)
    assert a == [0] and b == [1] and signs == [1.0]
    assert weights == [0.6]


def test_ranking_ablation_tiny(tmp_path):
    recs = []
    for i in range(32):
        v = 0.1 * ((i % 5) - 2)
        recs.append({"fen": FENS[i % 2], "value": v, "wdl": 1, "game_id": f"g{i // 4}",
                     "ply": 10})
    pairs = [{"parent": FENS[0], "child_a": FENS[0], "child_b": FENS[1],
              "sign": 1.0 if i % 2 == 0 else -1.0, "margin": 0.3, "teacher_id": "t",
              "teacher_a": 0.2, "teacher_b": -0.1} for i in range(8)]
    cfg = {"arch": "RUNE-MLP", "seed": 0, "lr": 1e-3, "batch_size": 8, "lambda_rank": 0.5}
    res = RankingAblation(cfg).run(recs, recs[:8], pairs, str(tmp_path), budget=128)
    assert "L0" in res and "L1" in res
    assert "sibling_acc" in res["L1"]
