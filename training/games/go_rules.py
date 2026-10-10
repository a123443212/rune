from training.games.go_board import group_and_liberties, neighbors_of


def opponent_stone(stone):
    if stone == 1:
        return -1
    return 1


def play_stone(board, n, sq, stone, ko_sq=None):
    if board[sq] != 0:
        return None
    if ko_sq is not None and sq == ko_sq:
        return None
    nb = list(board)
    nb[sq] = stone
    opp = opponent_stone(stone)
    captured = []
    seen = set()
    for adj in neighbors_of(sq, n):
        if nb[adj] != opp or adj in seen:
            continue
        stones, libs = group_and_liberties(nb, adj, n)
        for s in stones:
            seen.add(s)
        if len(libs) == 0:
            captured.extend(stones)
    for s in captured:
        nb[s] = 0
    _, libs = group_and_liberties(nb, sq, n)
    if len(libs) == 0:
        return None
    new_ko = None
    if len(captured) == 1:
        stones, libs = group_and_liberties(nb, sq, n)
        if len(stones) == 1 and len(libs) == 1:
            new_ko = captured[0]
    return nb, new_ko, captured


def legal_moves(board, n, stm, ko_sq=None):
    if stm == 0:
        stone = 1
    else:
        stone = -1
    out = []
    for sq in range(n * n):
        if board[sq] != 0:
            continue
        if ko_sq is not None and sq == ko_sq:
            continue
        res = play_stone(board, n, sq, stone, ko_sq)
        if res is not None:
            out.append(sq)
    out.append(-1)
    return out


def apply_move(board, n, stm, move, ko_sq=None):
    if move == -1:
        return list(board), None, []
    if stm == 0:
        stone = 1
    else:
        stone = -1
    res = play_stone(board, n, move, stone, ko_sq)
    if res is None:
        raise ValueError("illegal go move")
    nb, new_ko, captured = res
    return nb, new_ko, captured


def territory_owner(board, n):
    owner = [0] * (n * n)
    done = [False] * (n * n)
    for sq in range(n * n):
        if board[sq] != 0 or done[sq]:
            continue
        region = []
        border = set()
        stack = [sq]
        done[sq] = True
        while stack:
            cur = stack.pop()
            region.append(cur)
            r = cur // n
            c = cur % n
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if nr < 0 or nc < 0 or nr >= n or nc >= n:
                    continue
                nsq = nr * n + nc
                v = board[nsq]
                if v == 0 and not done[nsq]:
                    done[nsq] = True
                    stack.append(nsq)
                elif v != 0:
                    border.add(v)
        fill = 0
        if len(border) == 1:
            fill = next(iter(border))
        for s in region:
            owner[s] = fill
    return owner


def area_score(board, n, komi=7.5):
    black = 0
    white = 0
    for v in board:
        if v == 1:
            black += 1
        elif v == -1:
            white += 1
    owner = territory_owner(board, n)
    for sq in range(n * n):
        if board[sq] != 0:
            continue
        if owner[sq] == 1:
            black += 1
        elif owner[sq] == -1:
            white += 1
    return float(black) - float(white) - float(komi)
