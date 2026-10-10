import torch


def test_token_adapter_xiangqi():
    from training.games import get as get_game
    from training.models.rune_models import build_model
    from training.search.token_adapter import build_token_callbacks, uniform_priors
    from training.games.xiangqi_legal import legal_moves
    torch.manual_seed(2)
    g = get_game("xiangqi")
    m = build_model("RUNE-MLP", game="xiangqi")
    m.eval()
    s = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
    eval_fn = build_token_callbacks(g, m, legal_moves)
    v, p = eval_fn(s)
    assert -1.0 <= v <= 1.0
    assert len(p) == 44
    assert abs(sum(p) - 1.0) < 1e-9
    assert abs(sum(uniform_priors([1, 2, 3])) - 1.0) < 1e-9


def test_token_adapter_shogi():
    from training.games import get as get_game
    from training.models.rune_models import build_model
    from training.search.token_adapter import search_move
    from training.games.shogi_legal import apply_move, legal_moves
    torch.manual_seed(2)
    g = get_game("shogi")
    m = build_model("RUNE-MLP", game="shogi")
    m.eval()
    s = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    mv, info = search_move(g, m, s, legal_moves, apply_move, simulations=4, seed=1)
    assert mv in legal_moves(s)
    assert info["root_visits"] > 0


def test_search_teacher_distills():
    from training.games import get as get_game
    from training.models.rune_models import build_model
    from training.engine.search_teacher import distill_states, teacher_cost
    from training.games.xiangqi_legal import apply_move, legal_moves
    torch.manual_seed(2)
    g = get_game("xiangqi")
    m = build_model("RUNE-MLP", game="xiangqi")
    m.eval()
    s = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
    recs = distill_states(g, m, [s], legal_moves, apply_move, simulations=4, seed=1)
    assert len(recs) == 1
    assert recs[0]["move"] in legal_moves(s)
    assert recs[0]["teacher"] == "search_classic_v01"
    c = teacher_cost(recs)
    assert c["positions"] == 1
    assert c["simulations"] == 4


def test_material_evals():
    from training.search.classic_eval import material_eval
    xe = material_eval("xiangqi")
    se = material_eval("shogi")
    s = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
    assert xe(s) == 0.0
    t = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert se(t) == 0.0
