from training.features.python_features import game_phase, make_sq, on_board, parse_fen, piece_attacks, sq_file, sq_rank

CONTEXT_DIM = 17
CONTEXT_NAMES = ["stm", "phase", "us_pawns", "them_pawns", "us_minors", "them_minors",
                 "us_rooks", "them_rooks", "us_queens", "them_queens", "us_total",
                 "them_total", "shield", "castle", "ep", "check", "halfmove"]


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def context_vector(fen):
    board, stm, castle_mask, ep_sq = parse_fen(fen)
    us_pawns = 0
    them_pawns = 0
    us_minors = 0
    them_minors = 0
    us_rooks = 0
    them_rooks = 0
    us_queens = 0
    them_queens = 0
    us_total = 0
    them_total = 0
    for c in board:
        if c is None:
            continue
        if c[1] == ("w" if stm == 0 else "b"):
            us_total += 1
            if c[0] == "p":
                us_pawns += 1
            if c[0] in ("n", "b"):
                us_minors += 1
            if c[0] == "r":
                us_rooks += 1
            if c[0] == "q":
                us_queens += 1
        else:
            them_total += 1
            if c[0] == "p":
                them_pawns += 1
            if c[0] in ("n", "b"):
                them_minors += 1
            if c[0] == "r":
                them_rooks += 1
            if c[0] == "q":
                them_queens += 1
    color = "w" if stm == 0 else "b"
    enemy = "b" if stm == 0 else "w"
    king = next((sq for sq in range(64) if board[sq] == ("k", color)), -1)
    shield = 0
    if king >= 0:
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f, r = sq_file(king) + df, sq_rank(king) + dr
                if not on_board(f, r):
                    continue
                cell = board[make_sq(f, r)]
                if cell is not None and cell[1] == color and cell[0] != "k":
                    shield += 1
    check = 0
    if king >= 0:
        for asq in range(64):
            attacker = board[asq]
            if attacker is None or attacker[1] != enemy:
                continue
            if piece_attacks(board, asq, king):
                check = 1
                break
    parts = fen.split()
    half = 0
    if len(parts) > 4:
        try:
            half = max(0, int(parts[4]))
        except ValueError:
            half = 0
    return [clamp01(float(stm)), clamp01(game_phase(board) / 2.0),
            clamp01(us_pawns / 8.0), clamp01(them_pawns / 8.0),
            clamp01(us_minors / 8.0), clamp01(them_minors / 8.0),
            clamp01(us_rooks / 4.0), clamp01(them_rooks / 4.0),
            clamp01(us_queens / 2.0), clamp01(them_queens / 2.0),
            clamp01(us_total / 16.0), clamp01(them_total / 16.0),
            clamp01(shield / 8.0), clamp01(castle_mask / 15.0),
            clamp01(float(1 if ep_sq >= 0 else 0)), clamp01(float(check)),
            clamp01(half / 100.0)]
