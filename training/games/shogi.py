import torch

from training.games.base import GameSpec

TYPES = ["p", "l", "n", "s", "g", "b", "r", "k",
         "pp", "pl", "pn", "ps", "hb", "dr"]
PROMO_OF = {"pp": "p", "pl": "l", "pn": "n", "ps": "s", "hb": "b", "dr": "r"}
HAND_TYPES = ["p", "l", "n", "s", "g", "b", "r"]
HAND_MAX = [18, 4, 4, 4, 4, 2, 2]
HAND_BASE = [0, 19, 24, 29, 34, 39, 42]

FEATURE_VERSION = "shogi_raw_v01"
VOCABS = [648, 648, 972, 45, 45, 1134, 1134, 64, 1]
CONTEXT_DIM = 12
CONTEXT_NAMES = ["stm", "phase", "us_hand", "them_hand", "us_promo",
                 "them_promo", "us_board", "them_board", "check",
                 "king_adv", "mat_lead", "move_no"]


def sq_file(sq):
    return sq % 9


def sq_rank(sq):
    return sq // 9


def on_board(f, r):
    return 0 <= f < 9 and 0 <= r < 9


def parse_sfen(sfen):
    parts = sfen.split()
    placement = parts[0]
    side = parts[1] if len(parts) > 1 else "b"
    hands = parts[2] if len(parts) > 2 else "-"
    move_no = 1
    if len(parts) > 3:
        try:
            move_no = max(1, int(parts[3]))
        except ValueError:
            move_no = 1
    board = [None] * 81
    ranks = placement.split("/")
    if len(ranks) != 9:
        raise ValueError(f"bad sfen board {sfen}")
    for ri, rank in enumerate(ranks):
        f = 0
        i = 0
        while i < len(rank):
            c = rank[i]
            if c.isdigit():
                f += int(c)
                i += 1
            else:
                promo = False
                if c == "+":
                    promo = True
                    i += 1
                    c = rank[i]
                color = 0 if c.isupper() else 1
                pt = c.lower()
                if promo:
                    pt = {"p": "pp", "l": "pl", "n": "pn", "s": "ps",
                          "b": "hb", "r": "dr"}.get(pt)
                    if pt is None:
                        raise ValueError(f"bad sfen promo {sfen}")
                if f >= 9:
                    raise ValueError(f"bad sfen rank {sfen}")
                board[ri * 9 + f] = (pt, color)
                f += 1
                i += 1
        if f != 9:
            raise ValueError(f"bad sfen rank width {sfen}")
    hand = {}
    if hands != "-":
        num = ""
        for c in hands:
            if c.isdigit():
                num += c
            else:
                n = int(num) if num else 1
                num = ""
                color = 0 if c.isupper() else 1
                hand[(c.lower(), color)] = hand.get((c.lower(), color), 0) + n
    stm = 0 if side == "b" else 1
    return board, stm, hand, move_no


def step_moves(pt, color):
    f = -1 if color == 0 else 1
    if pt == "p":
        return [(0, f)]
    if pt == "n":
        return [(-1, 2 * f), (1, 2 * f)]
    if pt == "s":
        return [(0, f), (-1, f), (1, f), (-1, -f), (1, -f)]
    if pt in ("g", "pp", "pl", "pn", "ps"):
        return [(0, f), (-1, f), (1, f), (-1, 0), (1, 0), (0, -f)]
    if pt == "k":
        return [(-1, -1), (0, -1), (1, -1), (-1, 0),
                (1, 0), (-1, 1), (0, 1), (1, 1)]
    if pt == "hb":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if pt == "dr":
        return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    return []


def slide_dirs(pt):
    if pt in ("l",):
        return None
    if pt == "b":
        return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    if pt == "r":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if pt == "hb":
        return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    if pt == "dr":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return []


def piece_attacks(board, frm, target):
    cell = board[frm]
    if cell is None:
        return False
    pt, color = cell
    ff, rf = sq_file(frm), sq_rank(frm)
    tf, rt = sq_file(target), sq_rank(target)
    if pt == "l":
        f = -1 if color == 0 else 1
        if tf != ff:
            return False
        step = 1 if rt > rf else -1
        if step != f:
            return False
        r = rf + step
        while r != rt:
            if board[r * 9 + tf] is not None:
                return False
            r += step
        return True
    for df, dr in step_moves(pt, color):
        if ff + df == tf and rf + dr == rt:
            return True
    for df, dr in slide_dirs(pt):
        f, r = ff + df, rf + dr
        while on_board(f, r):
            if f == tf and r == rt:
                return True
            if board[r * 9 + f] is not None:
                break
            f += df
            r += dr
    return False


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def game_phase_of(hand_total, promo_count):
    n = hand_total + promo_count
    if n <= 3:
        return 0
    if n <= 9:
        return 1
    return 2


class ShogiGame(GameSpec):
    game_id = "shogi"
    num_groups = 9
    vocabs = tuple(VOCABS)
    tokens = 8
    token_dim = 32
    context_dim = CONTEXT_DIM
    context_names = tuple(CONTEXT_NAMES)
    feature_version = FEATURE_VERSION

    def extract(self, state):
        board, stm, hand, move_no = parse_sfen(state)
        feats = []
        promo = 0
        for sq in range(81):
            cell = board[sq]
            if cell is None:
                continue
            pt, color = cell
            ci = 0 if color == stm else 1
            ti = TYPES.index(pt)
            if pt in ("p", "l", "n", "s"):
                feats.append((0, (ci * 4 + ti) * 81 + sq))
            elif pt in ("g", "b", "r", "k"):
                feats.append((1, (ci * 4 + (ti - 4)) * 81 + sq))
            else:
                promo += 1
                feats.append((2, (ci * 6 + (ti - 8)) * 81 + sq))
            if pt in ("p", "l", "n", "s", "g", "b", "r"):
                feats.append((6, (ci * 7 + TYPES.index(pt)) * 81 + sq))
        for ti, t in enumerate(HAND_TYPES):
            cu = hand.get((t, stm), 0)
            ct = hand.get((t, 1 - stm), 0)
            feats.append((3, HAND_BASE[ti] + min(cu, HAND_MAX[ti])))
            feats.append((4, HAND_BASE[ti] + min(ct, HAND_MAX[ti])))
        kus = -1
        for sq in range(81):
            if board[sq] == ("k", stm):
                kus = sq
                break
        check = 0
        if kus >= 0:
            for asq in range(81):
                attacker = board[asq]
                if attacker is None or attacker[1] == stm:
                    continue
                if piece_attacks(board, asq, kus):
                    check = 1
                    feats.append((5, TYPES.index(attacker[0]) * 81 + asq))
        hand_total = sum(hand.values())
        phase = game_phase_of(hand_total, promo)
        feats.append((7, stm))
        feats.append((7, 2 + min(hand_total // 4, 7)))
        feats.append((7, 10 + phase))
        if check:
            feats.append((7, 13))
        feats.append((7, 14 + min(move_no // 20, 5)))
        return sorted(set(feats))

    def context(self, state):
        board, stm, hand, move_no = parse_sfen(state)
        us_hand = sum(v for (t, c), v in hand.items() if c == stm)
        them_hand = sum(v for (t, c), v in hand.items() if c != stm)
        us_promo = 0
        them_promo = 0
        us_board = 0
        them_board = 0
        kus = -1
        for sq in range(81):
            cell = board[sq]
            if cell is None:
                continue
            pt, color = cell
            if color == stm:
                us_board += 1
                if pt == "k":
                    kus = sq
                if pt in PROMO_OF:
                    us_promo += 1
            else:
                them_board += 1
                if pt in PROMO_OF:
                    them_promo += 1
        check = 0
        if kus >= 0:
            for asq in range(81):
                attacker = board[asq]
                if attacker is None or attacker[1] == stm:
                    continue
                if piece_attacks(board, asq, kus):
                    check = 1
                    break
        if kus >= 0:
            kr = sq_rank(kus)
            adv = (8 - kr) / 8.0 if stm == 0 else kr / 8.0
        else:
            adv = 0.0
        lead = (us_board + us_hand - them_board - them_hand + 20) / 40.0
        phase = game_phase_of(us_hand + them_hand, us_promo + them_promo)
        return [clamp01(float(stm)), clamp01(phase / 2.0),
                clamp01(us_hand / 20.0), clamp01(them_hand / 20.0),
                clamp01(us_promo / 8.0), clamp01(them_promo / 8.0),
                clamp01(us_board / 20.0), clamp01(them_board / 20.0),
                clamp01(float(check)), clamp01(adv),
                clamp01(lead), clamp01(move_no / 200.0)]

    def phase(self, state):
        board, _, hand, _ = parse_sfen(state)
        promo = sum(1 for c in board if c is not None and c[0] in PROMO_OF)
        return game_phase_of(sum(hand.values()), promo)

    def normalize(self, state):
        parts = state.split()
        return " ".join(parts[:3])

    def phase_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 10) & (g7 <= 12)
        cand = torch.where(valid, g7, torch.full_like(g7, 10))
        return (cand.amax(dim=1) - 10).clamp(0, 2).long()
