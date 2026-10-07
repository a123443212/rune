import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.engine.cache import EvalCache, cache_key
from training.engine.contract import EvalOutput, make_output, needs_only
from training.engine.lazy import LazyConfig, bounded_refine, should_refine
from training.engine.scale import canonical_value, engine_score, is_extreme, value_from_wdl, value_wdl_consistent, wdl_normalize


def test_contract_minimal():
    o = make_output(0.5, [0.6, 0.3, 0.1])
    assert abs(o.value - 0.5) < 1e-9
    assert abs(sum(o.wdl) - 1.0) < 1e-9
    assert o.to_minimal()[0] == o.value
    assert "trace" not in needs_only(o, ["value", "wdl"])


def test_scale_shared():
    assert canonical_value(2.0) == 1.0
    assert canonical_value(-2.0) == -1.0
    assert engine_score(1.0) == 1000
    assert engine_score(-1.0) == -1000
    assert engine_score(99.0) < 9000
    assert is_extreme(0.99)
    assert not is_extreme(0.1)


def test_wdl_consistency():
    assert value_wdl_consistent(0.5, [0.7, 0.2, 0.1])
    assert not value_wdl_consistent(0.9, [0.1, 0.1, 0.8])
    w = wdl_normalize([2.0, 1.0, 1.0])
    assert abs(sum(w) - 1.0) < 1e-9


def test_stm_cache_key():
    a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    b = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
    assert cache_key(a, "h", "full") != cache_key(b, "h", "full")
    assert cache_key(a, "hA", "full") != cache_key(a, "hB", "full")


def test_cache_correctness():
    c = EvalCache("modelA", "full", 4)
    assert c.get("f1") is None
    c.put("f1", 0.5)
    assert c.get("f1") == 0.5
    d = EvalCache("modelB", "full", 4)
    assert d.get("f1") is None


def test_lazy_modes():
    l0 = LazyConfig(mode="L0")
    l1 = LazyConfig(mode="L1")
    l2 = LazyConfig(mode="L2", margin=0.08)
    assert should_refine(0.1, 0.0, 0.3, 0.0, 0.5, l0) is True
    assert should_refine(0.1, 0.0, 0.3, 0.0, 0.5, l1) is False
    assert should_refine(0.9, 0.0, 0.3, 0.0, 0.5, l1) is True
    assert should_refine(0.9, 0.0, 0.3, 0.0, 0.5, l2) is False
    assert should_refine(0.15, 0.0, 0.3, 0.9, 0.5, l2) is True
    assert should_refine(0.2, 0.0, 0.3, 0.0, 0.5, l2) is False
    assert should_refine(0.3, 0.0, 0.3, 0.0, 0.2, l2) is True
    assert bounded_refine(0, l2) is True
    assert bounded_refine(5, l2) is False


def test_search_mate_and_draw():
    from training.engine.search import AlphaBeta

    class MatedBoard:
        def legal_moves(self):
            return []

        def is_checkmate(self):
            return True

    class StaleBoard:
        def legal_moves(self):
            return []

        def is_checkmate(self):
            return False

    class NoEval:
        def evaluate(self, board):
            raise AssertionError("eval must not run on terminal nodes")

        def order_moves(self, board, moves):
            return moves

    ab = AlphaBeta(NoEval())
    s, m, _ = ab.search(MatedBoard(), 2)
    assert m is None
    assert abs(s - (-10000.0)) < 1e-6
    ab = AlphaBeta(NoEval())
    s, m, _ = ab.search(StaleBoard(), 2)
    assert m is None
    assert abs(s - 0.0) < 1e-9
