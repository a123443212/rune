VALID_SIZES = (9, 13, 19)


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def valid_size(n):
    return n == 9 or n == 13 or n == 19


def parse_stm(token):
    if token == "w" or token == "O":
        return 1
    return 0


def stm_to_stone(stm):
    if stm == 0:
        return 1
    return -1


def stone_to_stm(stone):
    if stone == 1:
        return 0
    return 1


def parse_grid(grid, n):
    rows = grid.split("/")
    if len(rows) != n:
        raise ValueError("bad go grid")
    board = [0] * (n * n)
    for r in range(n):
        row = rows[r]
        if len(row) != n:
            raise ValueError("bad go row")
        for c in range(n):
            ch = row[c]
            if ch == ".":
                board[r * n + c] = 0
            elif ch == "X":
                board[r * n + c] = 1
            elif ch == "O":
                board[r * n + c] = -1
            else:
                raise ValueError("bad go stone")
    return board


def grid_rows(board, n):
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
    return rows


def on_board(r, c, n):
    return 0 <= r < n and 0 <= c < n


def neighbors_of(sq, n):
    r = sq // n
    c = sq % n
    out = []
    if r > 0:
        out.append((r - 1) * n + c)
    if r + 1 < n:
        out.append((r + 1) * n + c)
    if c > 0:
        out.append(r * n + (c - 1))
    if c + 1 < n:
        out.append(r * n + (c + 1))
    return out


def group_and_liberties(board, start, n):
    target = board[start]
    if target == 0:
        return [], set()
    seen = set()
    stones = []
    libs = set()
    stack = [start]
    seen.add(start)
    while stack:
        cur = stack.pop()
        stones.append(cur)
        for nb in neighbors_of(cur, n):
            v = board[nb]
            if v == 0:
                libs.add(nb)
            elif v == target and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return stones, libs


def liberties_of(board, r, c, n):
    v = board[r * n + c]
    if v == 0:
        return 0
    _, libs = group_and_liberties(board, r * n + c, n)
    return len(libs)


def liberty_map(board, n):
    out = [0] * (n * n)
    done = [False] * (n * n)
    for sq in range(n * n):
        if board[sq] == 0 or done[sq]:
            continue
        stones, libs = group_and_liberties(board, sq, n)
        k = len(libs)
        for s in stones:
            out[s] = k
            done[s] = True
    return out
