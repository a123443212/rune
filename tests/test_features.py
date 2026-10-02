import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.conftest import find_binding

_d = find_binding()
if _d and _d not in sys.path:
    sys.path.insert(0, _d)

import pytest

try:
    import rune_bindings as rb

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

from training.features import python_features as PF

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1",
]


def test_version_matches():
    if not HAS_BINDINGS:
        pytest.skip("bindings not built")
    assert rb.feature_version() == PF.FEATURE_VERSION


def test_python_features_valid():
    for fen in FENS:
        feats = PF.extract_features(fen)
        assert len(feats) > 0
        assert feats == sorted(set(feats))
        for g, idx in feats:
            assert 0 <= g < 8
            assert 0 <= idx < PF.VOCAB_SIZES[g]


def test_python_matches_cpp():
    if not HAS_BINDINGS:
        pytest.skip("bindings not built")
    for fen in FENS:
        py_feats = PF.extract_features(fen)
        cpp_feats = rb.extract_features(fen)
        cpp_tuples = sorted((g, i) for g, i in cpp_feats)
        assert py_feats == cpp_tuples, f"mismatch on {fen}"


def test_mobility_matches_pseudo_count():
    if not HAS_BINDINGS:
        pytest.skip("bindings not built")
    for fen in FENS:
        b = rb.Board(fen)
        n = b.pseudo_count()
        feats = PF.extract_features(fen)
        mob = [i for g, i in feats if g == 6 and 384 <= i < 448]
        assert len(mob) == 1
        assert mob[0] % 32 == min(n, 31)
