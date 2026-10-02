import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.conftest import find_binding

_d = find_binding()
if _d and _d not in sys.path:
    sys.path.insert(0, _d)

import rune_bindings as rb

from training.datasets.composition import composition
from training.datasets.siblings import build_sibling_pairs
from training.samplers.disagreement import disagreement_metrics, sample_mixture, score_records


def scored_pool():
    pool = []
    for i in range(30):
        pool.append({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                     "value": 0.0, "wdl": 1, "game_id": f"g{i}", "ply": 10,
                     "teacher_value": 0.5 if i % 3 == 0 else 0.0, "student_value": 0.0,
                     "phase": i % 3})
    return pool


def test_score_and_metrics():
    scored = score_records(scored_pool())
    assert len(scored) == 30
    m = disagreement_metrics(scored)
    assert m["n"] == 30 and m["max_gap"] == 0.5
    assert "0" in m["mean_gap_by_phase"]


def test_mixture_ratios():
    pool = scored_pool()
    pure = sample_mixture(pool, 12, mode="random", disagreement_ratio=0.0, seed=0)
    assert len(pure) == 12
    mixed = sample_mixture(pool, 12, mode="random", disagreement_ratio=0.5, seed=0)
    assert len(mixed) == 12
    top = [r for r in mixed if r.get("dis_gap") == 0.5]
    assert len(top) >= 6


def test_composition_sums():
    comp = composition(scored_pool()[:10])
    assert abs(sum(comp["phase"].values()) - 1.0) < 1e-9
    assert abs(sum(comp["material"].values()) - 1.0) < 1e-9
    assert abs(sum(comp["tactical_proxy"].values()) - 1.0) < 1e-9


def test_sibling_pairs_provenance():
    teacher = lambda f: 0.3 if "4P3" in f else -0.2
    parents = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
               "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"]
    pairs = build_sibling_pairs(rb, parents, teacher, teacher_id="dummy",
                                max_pairs=10, margin_min=0.1, seed=0)
    assert len(pairs) > 0
    for p in pairs:
        assert p["parent"] and p["child_a"] != p["child_b"]
        assert p["teacher_id"] == "dummy"
        assert abs(p["teacher_a"] - p["teacher_b"]) >= 0.1
