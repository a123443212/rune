from training.games.go_board import clamp01, liberty_map, parse_grid, parse_stm, valid_size


DEFAULT_KOMI = 7.5


def parse_komi(token):
    try:
        v = float(token)
    except ValueError:
        raise ValueError("bad go komi")
    if v < -50.0 or v > 50.0:
        raise ValueError("bad go komi")
    return v


def parse_sq(token, n):
    if token == "-":
        return None
    try:
        v = int(token)
    except ValueError:
        raise ValueError("bad go ko")
    if v < 0 or v >= n * n:
        raise ValueError("bad go ko")
    return v


def parse_count(token, lo, hi):
    try:
        v = int(token)
    except ValueError:
        raise ValueError("bad go count")
    if v < lo or v > hi:
        raise ValueError("bad go count")
    return v


def decode_state(state, size=9):
    parts = state.split()
    if not parts:
        raise ValueError("bad go state")
    grid = parts[0]
    rows = grid.split("/")
    n = len(rows)
    if n != size:
        if valid_size(n):
            size = n
        else:
            raise ValueError("bad go size")
    if not valid_size(size):
        raise ValueError("bad go size")
    board = parse_grid(grid, size)
    stm = 0
    if len(parts) > 1:
        stm = parse_stm(parts[1])
    ko = None
    komi = DEFAULT_KOMI
    move_no = 1
    pass_no = 0
    if len(parts) > 2:
        ko = parse_sq(parts[2], size)
    if len(parts) > 3:
        komi = parse_komi(parts[3])
    if len(parts) > 4:
        move_no = parse_count(parts[4], 1, 10000)
    if len(parts) > 5:
        pass_no = parse_count(parts[5], 0, 500)
    return {
        "board": board,
        "n": size,
        "stm": stm,
        "ko": ko,
        "komi": komi,
        "move_no": move_no,
        "pass_no": pass_no,
        "grid": grid,
    }


def encode_state(board, n, stm, ko=None, komi=DEFAULT_KOMI, move_no=1, pass_no=0):
    rows = []
    for r in range(n):
        chars = []
        for c in range(n):
            v = board[r * n + c]
            if v == 0:
                chars.append(".")
            elif v == 1:
                chars.append("X")
            else:
                chars.append("O")
        rows.append("".join(chars))
    grid = "/".join(rows)
    if stm == 0:
        s = "b"
    else:
        s = "w"
    if ko is None:
        k = "-"
    else:
        k = str(int(ko))
    return grid + " " + s + " " + k + " " + str(float(komi)) + " " + str(int(move_no)) + " " + str(int(pass_no))


def normalize_state(state, size=9):
    d = decode_state(state, size)
    n = d["n"]
    board = d["board"]
    rows = []
    for r in range(n):
        chars = []
        for c in range(n):
            v = board[r * n + c]
            if v == 0:
                chars.append(".")
            elif v == 1:
                chars.append("X")
            else:
                chars.append("O")
        rows.append("".join(chars))
    grid = "/".join(rows)
    if d["stm"] == 0:
        s = "b"
    else:
        s = "w"
    if d["ko"] is None:
        return grid + " " + s
    return grid + " " + s + " " + str(int(d["ko"]))


def komi_bucket(komi):
    v = (float(komi) + 5.0) / 20.0
    v = clamp01(v)
    b = int(v * 7.99)
    if b < 0:
        return 0
    if b > 7:
        return 7
    return b


def move_bucket(move_no):
    b = (int(move_no) - 1) // 20
    if b < 0:
        return 0
    if b > 5:
        return 5
    return b


def game_phase(move_no, n):
    total = n * n
    if int(move_no) <= total // 3:
        return 0
    if int(move_no) <= (2 * total) // 3:
        return 1
    return 2


def context_v02(state, size=9):
    d = decode_state(state, size)
    board = d["board"]
    n = d["n"]
    stm = d["stm"]
    komi = d["komi"]
    move_no = d["move_no"]
    pass_no = d["pass_no"]
    ko = d["ko"]
    us = 0
    them = 0
    empty = 0
    for v in board:
        if v == 0:
            empty += 1
        elif (v == 1 and stm == 0) or (v == -1 and stm == 1):
            us += 1
        else:
            them += 1
    total = float(n * n)
    libs = liberty_map(board, n)
    lib1 = 0
    lib2 = 0
    lib3 = 0
    for sq in range(n * n):
        v = board[sq]
        if v == 0:
            continue
        mine = (v == 1 and stm == 0) or (v == -1 and stm == 1)
        if not mine:
            continue
        k = libs[sq]
        if k <= 1:
            lib1 += 1
        elif k == 2:
            lib2 += 1
        else:
            lib3 += 1
    if us > 0:
        f1 = clamp01(float(lib1) / float(us))
        f2 = clamp01(float(lib2) / float(us))
        f3 = clamp01(float(lib3) / float(us))
    else:
        f1 = 0.0
        f2 = 0.0
        f3 = 0.0
    if ko is None:
        ko_f = 0.0
    else:
        ko_f = 1.0
    return [
        clamp01(float(stm)),
        clamp01(float(us) / total),
        clamp01(float(them) / total),
        clamp01(float(komi) / 15.0),
        clamp01(float(move_no) / 200.0),
        clamp01(float(pass_no) / 2.0),
        clamp01(ko_f),
        clamp01(f1),
        clamp01(f2),
        clamp01(f3),
        clamp01(float(empty) / total),
        clamp01(float(game_phase(move_no, n)) / 2.0),
    ]
