import pytest

START = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"


def test_startpos_counts():
    from training.games.xiangqi_legal import apply_move, legal_moves
    red = legal_moves(START)
    assert len(red) == 44
    assert red == sorted(red)
    for mv in red:
        assert len(mv) == 4
    black = legal_moves(apply_move(START, red[0]))
    assert len(black) == 44


def test_apply_roundtrip_and_stm():
    from training.games.xiangqi import parse_fen
    from training.games.xiangqi_legal import apply_move, decode_move, legal_moves
    moves = legal_moves(START)
    fr, to = decode_move(moves[0])
    assert 0 <= fr < 90
    assert 0 <= to < 90
    nxt = apply_move(START, moves[0])
    board, stm, move_no = parse_fen(nxt)
    assert stm == 1
    assert move_no == 1
    assert board[to] is not None
    assert board[fr] is None
    with pytest.raises(ValueError):
        apply_move(START, "9999")
    with pytest.raises(ValueError):
        apply_move(START, moves[0][2:] + moves[0][:2] if moves[0][2:] != moves[0][:2] else "8080")


def test_exposed_check_filtered():
    from training.games.xiangqi import parse_fen
    from training.games.xiangqi_legal import apply_move, decode_move, king_in_check, legal_moves
    s = "4k4/4a4/9/9/4r4/9/9/9/9/4K4 w - - 0 1"
    moves = legal_moves(s)
    assert len(moves) > 0
    for mv in moves:
        nxt = apply_move(s, mv)
        board, _, _ = parse_fen(nxt)
        assert not king_in_check(board, 0)


def test_flying_general_check():
    from training.games.xiangqi_legal import king_in_check, legal_moves
    from training.games.xiangqi import parse_fen
    fly = "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"
    board, _, _ = parse_fen(fly)
    assert king_in_check(board, 0)
    assert king_in_check(board, 1)
    blocked = "4k4/9/9/9/4p4/9/9/9/9/4K4 w - - 0 1"
    b2, _, _ = parse_fen(blocked)
    assert not king_in_check(b2, 0)


def test_mate_is_loss():
    from training.games.xiangqi_legal import game_result, legal_moves
    mate = "3aka3/2H1P4/9/4R4/9/9/9/9/9/4K4 b - - 0 1"
    assert legal_moves(mate) == []
    assert game_result(mate) == "1-0"
    assert game_result(START) == "*"


def test_horse_leg_and_cannon_screen():
    from training.games.xiangqi_legal import apply_move, legal_moves
    s = "3k5/9/9/9/3p5/3H5/9/9/9/5K3 w - - 0 1"
    moves = legal_moves(s)
    assert "3958" not in moves
    assert "3956" not in moves
    assert "3950" in moves
    for mv in moves:
        nxt = apply_move(s, mv)
        assert legal_moves(nxt) is not None
