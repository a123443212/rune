from training.features.python_features import game_phase, make_sq, on_board, parse_fen, piece_attacks, sq_file, sq_rank

CONTEXT_DIM = 12
CONTEXT_NAMES = ["stm", "phase", "pawns", "minors", "rooks", "queens", "shield", "total",
                 "castle", "ep", "check", "halfmove"]


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def context_vector(fen):
    board, stm, castle_mask, ep_sq = parse_fen(fen)
    pawns = sum(1 for c in board if c is not None and c[0] == "p")
    minors = sum(1 for c in board if c is not None and c[0] in ("n", "b"))
    rooks = sum(1 for c in board if c is not None and c[0] == "r")
    queens = sum(1 for c in board if c is not None and c[0] == "q")
    total = sum(1 for c in board if c is not None)
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
    return [clamp01(float(stm)), clamp01(game_phase(board) / 2.0), clamp01(pawns / 16.0),
            clamp01(minors / 8.0), clamp01(rooks / 4.0), clamp01(queens / 2.0),
            clamp01(shield / 8.0), clamp01(total / 32.0), clamp01(castle_mask / 15.0),
            clamp01(float(1 if ep_sq >= 0 else 0)), clamp01(float(check)),
            clamp01(half / 100.0)]
