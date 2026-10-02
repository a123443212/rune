from training.features.python_features import game_phase, make_sq, on_board, parse_fen, sq_file, sq_rank

CONTEXT_DIM = 8
CONTEXT_NAMES = ["stm", "phase", "pawns", "minors", "rooks", "queens", "shield", "total"]


def context_vector(fen):
    board, stm, _, _ = parse_fen(fen)
    pawns = sum(1 for c in board if c is not None and c[0] == "p")
    minors = sum(1 for c in board if c is not None and c[0] in ("n", "b"))
    rooks = sum(1 for c in board if c is not None and c[0] == "r")
    queens = sum(1 for c in board if c is not None and c[0] == "q")
    total = sum(1 for c in board if c is not None)
    color = "w" if stm == 0 else "b"
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
    return [float(stm), game_phase(board) / 2.0, pawns / 16.0, minors / 8.0,
            rooks / 4.0, queens / 2.0, shield / 8.0, total / 32.0]
