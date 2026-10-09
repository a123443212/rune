VOCAB_SIZES = [256, 256, 256, 128, 128, 512, 512, 64, 4560]
NUM_GROUPS = 9
TOKENS = 8
TOKEN_DIM = 32
FEATURE_VERSION = "grouped_hkav2_fullthreats_v02"

GROUP_NAMES = [
    "pawn_structure",
    "king_zone",
    "minor_pieces",
    "rooks",
    "queens",
    "threats",
    "mobility",
    "global",
    "pawn_pairs",
]

PAWN_PAIR_VOCAB = 4560


def pawn_pair_index(ida, idb):
    if ida == idb:
        return None
    lo, hi = (ida, idb) if ida < idb else (idb, ida)
    idx = hi * (hi - 1) // 2 + lo
    if idx < 0 or idx >= PAWN_PAIR_VOCAB:
        return None
    return idx

TYPE_INDEX = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def sq_file(sq):
    return sq & 7


def sq_rank(sq):
    return sq >> 3


def make_sq(f, r):
    return r * 8 + f


def on_board(f, r):
    return 0 <= f < 8 and 0 <= r < 8


def parse_fen(fen):
    parts = fen.split()
    placement, side, castle, ep = parts[0], parts[1], parts[2], parts[3]
    board = [None] * 64
    rank = 7
    fil = 0
    for c in placement:
        if c == "/":
            rank -= 1
            fil = 0
        elif c.isdigit():
            fil += int(c)
        else:
            color = "w" if c.isupper() else "b"
            board[make_sq(fil, rank)] = (c.lower(), color)
            fil += 1
    stm = 0 if side == "w" else 1
    mask = 0
    if castle != "-":
        if "K" in castle:
            mask |= 1
        if "Q" in castle:
            mask |= 2
        if "k" in castle:
            mask |= 4
        if "q" in castle:
            mask |= 8
    ep_sq = -1
    if ep != "-":
        ep_sq = make_sq(ord(ep[0]) - ord("a"), ord(ep[1]) - ord("1"))
    return board, stm, mask, ep_sq


def sliding_attacks(board, frm, target):
    p, _ = board[frm]
    df = sq_file(target) - sq_file(frm)
    dr = sq_rank(target) - sq_rank(frm)
    adf, adr = abs(df), abs(dr)
    diag = p in ("b", "q")
    straight = p in ("r", "q")
    if not diag and not straight:
        return False
    is_diag = adf == adr
    is_straight = df == 0 or dr == 0
    if is_diag and not diag:
        return False
    if is_straight and not straight:
        return False
    if not is_diag and not is_straight:
        return False
    sf = 0 if df == 0 else (1 if df > 0 else -1)
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    f, r = sq_file(frm) + sf, sq_rank(frm) + sr
    while (f, r) != (sq_file(target), sq_rank(target)):
        if board[make_sq(f, r)] is not None:
            return False
        f += sf
        r += sr
    return True


def piece_attacks(board, frm, target):
    if board[frm] is None:
        return False
    p, color = board[frm]
    df = sq_file(target) - sq_file(frm)
    dr = sq_rank(target) - sq_rank(frm)
    adf, adr = abs(df), abs(dr)
    if p == "p":
        d = 1 if color == "w" else -1
        return adr == 1 and adf == 1 and dr == d
    if p == "n":
        return (adf == 1 and adr == 2) or (adf == 2 and adr == 1)
    if p == "k":
        return adf <= 1 and adr <= 1
    return sliding_attacks(board, frm, target)


def pseudo_move_count(board, stm, castle_mask, ep_sq):
    color = "w" if stm == 0 else "b"
    count = 0
    for sq in range(64):
        cell = board[sq]
        if cell is None or cell[1] != color:
            continue
        p, _ = cell
        f, r = sq_file(sq), sq_rank(sq)
        if p == "p":
            d = 1 if color == "w" else -1
            promo_rank = 7 if color == "w" else 0
            if on_board(f, r + d) and board[make_sq(f, r + d)] is None:
                to = make_sq(f, r + d)
                count += 4 if sq_rank(to) == promo_rank else 1
                start = 1 if color == "w" else 6
                if r == start and board[make_sq(f, r + 2 * d)] is None:
                    count += 1
            for df in (-1, 1):
                if not on_board(f + df, r + d):
                    continue
                to = make_sq(f + df, r + d)
                if board[to] is not None and board[to][1] != color:
                    count += 4 if sq_rank(to) == promo_rank else 1
                if to == ep_sq:
                    count += 1
        elif p == "n":
            for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
                if not on_board(f + df, r + dr):
                    continue
                to = make_sq(f + df, r + dr)
                if board[to] is None or board[to][1] != color:
                    count += 1
        elif p == "k":
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    if not on_board(f + df, r + dr):
                        continue
                    to = make_sq(f + df, r + dr)
                    if board[to] is None or board[to][1] != color:
                        count += 1
            if color == "w" and sq == make_sq(4, 0):
                if (castle_mask & 1) and board[make_sq(5, 0)] is None and board[make_sq(6, 0)] is None:
                    count += 1
                if (castle_mask & 2) and board[make_sq(3, 0)] is None and board[make_sq(2, 0)] is None and board[make_sq(1, 0)] is None:
                    count += 1
            if color == "b" and sq == make_sq(4, 7):
                if (castle_mask & 4) and board[make_sq(5, 7)] is None and board[make_sq(6, 7)] is None:
                    count += 1
                if (castle_mask & 8) and board[make_sq(3, 7)] is None and board[make_sq(2, 7)] is None and board[make_sq(1, 7)] is None:
                    count += 1
        else:
            diag = p in ("b", "q")
            straight = p in ("r", "q")
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    is_diag = df != 0 and dr != 0
                    if is_diag and not diag:
                        continue
                    if not is_diag and not straight:
                        continue
                    ff, rr = f + df, r + dr
                    while on_board(ff, rr):
                        to = make_sq(ff, rr)
                        if board[to] is None:
                            count += 1
                        else:
                            if board[to][1] != color:
                                count += 1
                            break
                        ff += df
                        rr += dr
    return count


def game_phase(board):
    n = 0
    for cell in board:
        if cell is None:
            continue
        if cell[0] in ("p", "k"):
            continue
        n += 1
    if n >= 12:
        return 0
    if n >= 6:
        return 1
    return 2


def extract_features(fen):
    board, stm, castle_mask, ep_sq = parse_fen(fen)
    us = "w" if stm == 0 else "b"
    kus = -1
    for sq in range(64):
        if board[sq] == ("k", us):
            kus = sq
            break
    fold = kus >= 0 and sq_file(kus) < 4

    def rel(sq):
        if not fold:
            return sq
        return sq_rank(sq) * 8 + (7 - sq_file(sq))

    def rel_color(color):
        return 0 if color == us else 1

    parts = fen.split()
    half = 0
    if len(parts) > 4:
        try:
            half = max(0, int(parts[4]))
        except ValueError:
            half = 0
    feats = []
    for sq in range(64):
        cell = board[sq]
        if cell is None:
            continue
        p, color = cell
        ci = rel_color(color)
        s = rel(sq)
        if p == "p":
            feats.append((0, ci * 64 + s))
            feats.append((0, 128 + sq_file(s) * 4 + sq_rank(s) // 2))
        if p == "k":
            feats.append((1, ci * 64 + s))
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    f, r = sq_file(sq) + df, sq_rank(sq) + dr
                    if not on_board(f, r):
                        continue
                    nsq = rel(make_sq(f, r))
                    feats.append((1, 128 + ci * 32 + nsq // 2))
        if p == "n":
            feats.append((2, ci * 64 + s))
        if p == "b":
            feats.append((2, 128 + ci * 64 + s))
        if p == "r":
            feats.append((3, ci * 64 + s))
        if p == "q":
            feats.append((4, ci * 64 + s))
        feats.append((6, TYPE_INDEX[p] * 64 + s))
    for vsq in range(64):
        victim = board[vsq]
        if victim is None:
            continue
        for asq in range(64):
            attacker = board[asq]
            if attacker is None or attacker[1] == victim[1]:
                continue
            if not piece_attacks(board, asq, vsq):
                continue
            vt = TYPE_INDEX[victim[0]]
            feats.append((5, vt * 64 + rel(vsq)))
            coarse = 384 + sq_file(rel(asq)) * 8 + sq_rank(asq)
            if coarse < 512:
                feats.append((5, coarse))
    mob = pseudo_move_count(board, stm, castle_mask, ep_sq)
    mob = min(mob, 31)
    feats.append((6, 384 + stm * 32 + mob))
    feats.append((7, stm))
    feats.append((7, 2 + castle_mask))
    total = sum(1 for c in board if c is not None)
    bucket = (total - 2) // 2
    bucket = max(0, min(15, bucket))
    feats.append((7, 18 + bucket))
    feats.append((7, 34 + game_phase(board)))
    if ep_sq >= 0:
        feats.append((7, 37 + sq_file(rel(ep_sq))))
    feats.append((7, 45 + min(half // 20, 4)))
    pawns = []
    for sq in range(64):
        cell = board[sq]
        if cell is None or cell[0] != "p":
            continue
        if sq < 8 or sq > 55:
            continue
        pawns.append(rel_color(cell[1]) * 48 + (rel(sq) - 8))
    for i in range(len(pawns)):
        for j in range(i + 1, len(pawns)):
            idx = pawn_pair_index(pawns[i], pawns[j])
            if idx is not None:
                feats.append((8, idx))
    return sorted(set(feats))


def normalized_key(fen):
    parts = fen.split()
    return " ".join(parts[:4])
