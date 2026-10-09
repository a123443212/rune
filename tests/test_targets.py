import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from training.datasets.targets import (
    INVALID,
    PROVISIONAL,
    UNSTABLE,
    VALID,
    assign_validity,
    build_rank_pairs,
    quality_weight,
    soft_wdl_probs,
    stability_from_levels,
)


def test_validity_states():
    assert assign_validity({"teacher_v": 0.2, "has_teacher": True}, {})[0] == VALID
    assert assign_validity({}, {})[0] == INVALID
    assert assign_validity({"teacher_v": 5.0, "has_teacher": True}, {})[0] == INVALID
    assert assign_validity({"teacher_v": 0.2, "has_teacher": True},
                           {"require_provenance": True}) == (PROVISIONAL, "no_provenance")
    assert assign_validity({"teacher_v": 0.2, "has_teacher": True,
                            "teacher_stability": 0.9},
                           {"unstable_above": 0.3})[0] == UNSTABLE
    assert assign_validity({"teacher_v": 0.2, "has_teacher": True, "teacher_depth": 6},
                           {"min_depth": 10}) == (PROVISIONAL, "shallow")


def test_stability_from_levels():
    levels = {"d6": {"value_stm": 0.1}, "d10": {"value_stm": 0.12}, "d14": {"value_stm": 0.11}}
    mx, info = stability_from_levels(levels)
    assert mx is not None and mx < 0.05
    assert info["wdl_flips"] == 0
    assert stability_from_levels({"d6": {"value_stm": 0.1}})[0] is None


def test_quality_weight_bounded():
    assert quality_weight({}, "uniform") == 1.0
    assert quality_weight({"target_quality": 2.0}, "quality") == 1.0
    assert quality_weight({"target_quality": -1.0}, "quality") == 0.25
    assert quality_weight({"target_quality": 0.0}, "difficulty") == 1.0


def test_soft_wdl_probs():
    assert soft_wdl_probs({"teacher_wdl_probs": [0.1, 0.8, 0.1]}) == [0.1, 0.8, 0.1]
    assert soft_wdl_probs({"teacher_w": 0}) == [1.0, 0.0, 0.0]
    assert soft_wdl_probs({}) == [0.0, 1.0, 0.0]


def test_rank_pairs_ties():
    kids = [("a", 0.9), ("b", 0.88), ("c", 0.1)]
    out = build_rank_pairs(kids, tie_margin=0.05)
    assert ("a", "b") not in out["strict"]
    assert ("a", "c") in out["strict"] or ("a", "c") in out["soft"]
    assert len(out["ties"]) == 1
    assert build_rank_pairs([("a", 0.5)], 0.05) == {"strict": [], "soft": [], "ties": []}


def test_soft_wdl_and_quality_train_step():
    from training.trainer.trainer import Trainer

    cfg = {"arch": "RUNE-03-A", "seed": 0, "lr": 3e-4, "weight_decay": 0.01,
           "batch_size": 4, "lambda_wdl": 0.5, "lambda_rank": 0.0,
           "dense_params": {},
           "distill": {"enabled": True, "task": "value_wdl", "alpha": 0.5,
                       "soft_wdl": True, "quality_weighted": True}}
    tr = Trainer(cfg)
    torch.manual_seed(0)
    ids = [torch.randint(0, 64, (4, 4)) for _ in range(9)]
    masks = [torch.ones(4, 4) for _ in range(9)]
    value = torch.zeros(4)
    wdl = torch.ones(4, dtype=torch.long)
    teach = {"v": torch.zeros(4), "w": torch.ones(4, dtype=torch.long),
             "u": torch.zeros(4),
             "wp": torch.tensor([[0.1, 0.8, 0.1]] * 4),
             "qw": torch.tensor([1.0, 1.0, 0.0, 0.0])}
    losses = tr.train_step((ids, masks, None, value, wdl, teach))
    assert losses["distill_wdl"] > 0
    plain = tr.train_step((ids, masks, None, value, wdl, None))
    assert "dist_total" not in plain
    assert plain["total"] > 0
