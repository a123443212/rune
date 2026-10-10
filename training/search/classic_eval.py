from training.games.xiangqi import crossed as xiangqi_crossed
from training.games.xiangqi import parse_fen as parse_xiangqi


def xiangqi_material(state):
    board, stm, _ = parse_xiangqi(state)
    values = {"p": 60, "h": 270, "r": 600, "c": 300, "a": 120, "e": 120, "k": 0}
    score = 0.0
    for sq, cell in enumerate(board):
        if cell is None:
            continue
        p, color = cell
        v = float(values.get(p, 0))
        if p == "p":
            from training.games.xiangqi import crossed, sq_rank
            if crossed(sq_rank(sq), color):
                v += 60.0
        if color == stm:
            score += v
        else:
            score -= v
    return score


def shogi_material(state):
    from training.games.shogi import parse_sfen
    values = {"p": 100, "l": 300, "n": 300, "s": 500, "g": 600, "b": 800, "r": 900, "k": 0,
              "pp": 500, "pl": 500, "pn": 500, "ps": 550, "hb": 900, "dr": 1000}
    board, stm, hand, _ = parse_sfen(state)
    score = 0.0
    for cell in board:
        if cell is None:
            continue
        p, color = cell
        v = float(values.get(p, 0))
        if color == stm:
            score += v
        else:
            score -= v
    hand_values = {"p": 100, "l": 300, "n": 300, "s": 500, "g": 600, "b": 800, "r": 900}
    for (t, c), n in hand.items():
        v = float(hand_values.get(t, 0)) * n
        if c == stm:
            score += v
        else:
            score -= v
    return score


def material_eval(game_id):
    if game_id == "xiangqi":
        return xiangqi_material
    return shogi_material
