import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training import games
from training.features import context as ctx_mod
from training.features import python_features as pf
from training.games.shogi import ShogiGame, parse_sfen

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_registry():
    assert set(games.available()) == {"chess", "shogi"}
    with pytest.raises(ValueError):
        games.get("go")


def test_chess_delegates_to_legacy():
    g = games.get("chess")
    assert g.extract(START_FEN) == pf.extract_features(START_FEN)
    assert g.context(START_FEN) == ctx_mod.context_vector(START_FEN)
    board, _, _, _ = pf.parse_fen(START_FEN)
    assert g.phase(START_FEN) == pf.game_phase(board)
    assert g.normalize(START_FEN) == pf.normalized_key(START_FEN)
    assert g.feature_version == pf.FEATURE_VERSION
    assert g.context_dim == ctx_mod.CONTEXT_DIM


def test_shogi_startpos_extract():
    g = games.get("shogi")
    feats = g.extract(START_SFEN)
    assert feats == sorted(set(feats))
    assert len(feats) > 60
    for gg, ii in feats:
        assert 0 <= gg < g.num_groups
        assert 0 <= ii < g.vocabs[gg]
    assert (7, 0) in feats
    assert (7, 10) in feats
    assert g.phase(START_SFEN) == 0
    assert ShogiGame().extract(START_SFEN) == feats


def test_shogi_hands_and_check():
    g = games.get("shogi")
    mid = "lnsgkgsn1/1r5b1/pppp1pppp/4p4/9/4P4/PPPP1PPPP/1B5R1/LNSGKGSNL b 2P 10"
    feats = g.extract(mid)
    assert (3, 2) in feats
    assert g.phase(mid) == 0
    gives = "4k4/9/9/9/9/9/9/9/4R4 w - 1"
    feats = g.extract(gives)
    assert any(gg == 5 for gg, _ in feats)
    assert (7, 13) in feats


def test_shogi_context_and_normalize():
    g = games.get("shogi")
    ctx = g.context(START_SFEN)
    assert len(ctx) == g.context_dim == 12
    assert all(0.0 <= v <= 1.0 for v in ctx)
    assert ctx[0] == 0.0
    assert g.normalize(START_SFEN) == " ".join(START_SFEN.split()[:3])


def test_shogi_bad_sfen():
    g = games.get("shogi")
    with pytest.raises(ValueError):
        parse_sfen("8/8 w - 1")
    with pytest.raises(ValueError):
        g.extract("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSN b - 1")


def feats_to_tensors(g, feats):
    ids = []
    masks = []
    for gg in range(g.num_groups):
        idx = [i for q, i in feats if q == gg]
        if idx:
            ids.append(torch.tensor([idx]))
            masks.append(torch.ones(1, len(idx)))
        else:
            ids.append(torch.zeros(1, 1, dtype=torch.long))
            masks.append(torch.zeros(1, 1))
    return ids, masks


def test_shared_stack_both_games():
    from training.models.rune_models import build_model
    cases = [("chess", START_FEN), ("shogi", START_SFEN)]
    for game, state in cases:
        g = games.get(game)
        m = build_model("RUNE-MLP", game=game)
        m.eval()
        ids, masks = feats_to_tensors(g, g.extract(state))
        with torch.no_grad():
            v, w = m(ids, masks)
        assert v.shape == (1,)
        assert w.shape == (1, 3)
        assert -1.0 <= float(v[0]) <= 1.0
        spec = m.model_spec()
        assert spec["game"] == game
        assert spec["feature_set"] == g.feature_version


def test_shared_stack_buckets_decode_matches_game_phase():
    from training.models.rune_models import build_model
    cases = [("chess", START_FEN), ("shogi", START_SFEN)]
    for game, state in cases:
        g = games.get(game)
        m = build_model("RUNE-MLP", buckets=3, game=game)
        m.eval()
        ids, masks = feats_to_tensors(g, g.extract(state))
        with torch.no_grad():
            v, w = m(ids, masks)
        assert v.shape == (1,)
        assert m.head.phases_from_ids(ids, masks).item() == g.phase(state)


def test_export_header_carries_game(tmp_path):
    from training.export.export import export_model, read_header
    from training.models.rune_models import build_model
    for game in ("chess", "shogi"):
        m = build_model("RUNE-MLP", game=game)
        p = str(tmp_path / f"{game}.rune")
        header = export_model(m, p, quantization="fp32")
        assert header["game"] == game
        assert read_header(p)["game"] == game


def test_dataset_game_aware():
    from training.datasets.rune_dataset import make_loader
    recs = [
        {"game": "shogi", "state": START_SFEN, "value": 0.1, "wdl": 1},
        {"game": "shogi", "state": START_SFEN, "value": -0.2, "wdl": 0},
    ]
    loader, ds = make_loader(recs, batch_size=2, shuffle=False, seed=0)
    assert ds.game.game_id == "shogi"
    batch = next(iter(loader))
    ids, masks, value, wdl = batch
    assert len(ids) == 9
    assert value.tolist() == pytest.approx([0.1, -0.2])


def test_flex_dataset_game_context():
    from training.datasets.flex_dataset import make_flex_loader
    recs = [{"game": "shogi", "state": START_SFEN, "value": 0.0, "wdl": 1}]
    loader, ds = make_flex_loader(recs, batch_size=1, shuffle=False, seed=0)
    batch = next(iter(loader))
    ids, masks, ctx, value, wdl = batch
    assert ctx.shape == (1, 12)
