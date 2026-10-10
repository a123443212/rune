from training.games.xiangqi import crossed, in_palace, make_sq, on_board, sq_file, sq_rank


def pawn_dests(board, sq, color):
    f = sq_file(sq)
    r = sq_rank(sq)
    fwd = 1 if color == 0 else -1
    out = []
    if on_board(f, r + fwd):
        out.append(make_sq(f, r + fwd))
    if crossed(r, color):
        if on_board(f - 1, r):
            out.append(make_sq(f - 1, r))
        if on_board(f + 1, r):
            out.append(make_sq(f + 1, r))
    return out


def horse_dests(board, sq, color):
    f = sq_file(sq)
    r = sq_rank(sq)
    out = []
    for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
        tf = f + df
        tr = r + dr
        if not on_board(tf, tr):
            continue
        if abs(df) == 2:
            leg = make_sq(f + df // 2, r)
        else:
            leg = make_sq(f, r + dr // 2)
        if board[leg] is not None:
            continue
        out.append(make_sq(tf, tr))
    return out


def slide_ray(board, sq, df, dr, capture):
    f = sq_file(sq)
    r = sq_rank(sq)
    out = []
    f += df
    r += dr
    while on_board(f, r):
        t = make_sq(f, r)
        if board[t] is None:
            if not capture:
                out.append(t)
        else:
            if capture:
                out.append(t)
            break
        f += df
        r += dr
    return out


def rook_dests(board, sq, color):
    out = []
    for df, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        out.extend(slide_ray(board, sq, df, dr, False))
        caps = slide_ray(board, sq, df, dr, True)
        for t in caps:
            if board[t] is not None and board[t][1] != color:
                out.append(t)
    return out


def cannon_dests(board, sq, color):
    f = sq_file(sq)
    r = sq_rank(sq)
    out = []
    for df, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cf, cr = f + df, r + dr
        while on_board(cf, cr) and board[make_sq(cf, cr)] is None:
            out.append(make_sq(cf, cr))
            cf += df
            cr += dr
        if not on_board(cf, cr):
            continue
        cf += df
        cr += dr
        while on_board(cf, cr):
            t = make_sq(cf, cr)
            if board[t] is not None:
                if board[t][1] != color:
                    out.append(t)
                break
            cf += df
            cr += dr
    return out


def advisor_dests(board, sq, color):
    f = sq_file(sq)
    r = sq_rank(sq)
    out = []
    for df, dr in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        tf, tr = f + df, r + dr
        if in_palace(tf, tr, color):
            out.append(make_sq(tf, tr))
    return out


def elephant_dests(board, sq, color):
    f = sq_file(sq)
    r = sq_rank(sq)
    out = []
    for df, dr in ((-2, -2), (2, -2), (-2, 2), (2, 2)):
        tf, tr = f + df, r + dr
        if not on_board(tf, tr):
            continue
        if color == 0 and tr > 4:
            continue
        if color == 1 and tr < 5:
            continue
        if board[make_sq(f + df // 2, r + dr // 2)] is not None:
            continue
        out.append(make_sq(tf, tr))
    return out


def king_dests(board, sq, color):
    f = sq_file(sq)
    r = sq_rank(sq)
    out = []
    for df, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        tf, tr = f + df, r + dr
        if in_palace(tf, tr, color):
            out.append(make_sq(tf, tr))
    foe = 1 - color
    for t in range(90):
        if board[t] is not None and board[t] == ("k", foe):
            kf = sq_file(t)
            if kf == f:
                lo = sq if sq < t else t
                hi = t if sq < t else sq
                blocked = False
                for m in range(lo + 9, hi, 9):
                    if board[m] is not None:
                        blocked = True
                        break
                if not blocked:
                    out.append(t)
    return out


def pseudo_dests(board, sq):
    cell = board[sq]
    if cell is None:
        return []
    p, color = cell
    if p == "p":
        return pawn_dests(board, sq, color)
    if p == "h":
        return horse_dests(board, sq, color)
    if p == "r":
        return rook_dests(board, sq, color)
    if p == "c":
        return cannon_dests(board, sq, color)
    if p == "a":
        return advisor_dests(board, sq, color)
    if p == "e":
        return elephant_dests(board, sq, color)
    if p == "k":
        return king_dests(board, sq, color)
    return []
