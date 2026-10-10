import torch

from training.games.base import GameSpec
from training.games.shogi import HAND_TYPES, ShogiGame, parse_sfen, piece_attacks, sq_file, sq_rank
from training.games.shogi_legal import king_in_check
from training.games.shogi_moves import drop_squares

FEATURE_VERSION_V02 = "shogi_sem_v02"


def checking_drops(board, hand, stm):
    found = []
    for t in HAND_TYPES:
        n = hand.get((t, stm), 0)
        if n <= 0:
            continue
        for sq in drop_squares(board, n, t, stm):
            nb = list(board)
            nb[sq] = (t, stm)
            foe = 1 - stm
            kus = -1
            for s in range(81):
                if nb[s] == ("k", foe):
                    kus = s
                    break
            if kus < 0:
                continue
            if piece_attacks(nb, sq, kus):
                found.append(sq)
    return sorted(set(found))


def ring_pressure(board, stm):
    kus = -1
    for sq in range(81):
        if board[sq] == ("k", stm):
            kus = sq
            break
    if kus < 0:
        return 0
    kf = sq_file(kus)
    kr = sq_rank(kus)
    n = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            tf = kf + df
            tr = kr + dr
            if tf < 0 or tf > 8 or tr < 0 or tr > 8:
                continue
            t = tr * 9 + tf
            hit = False
            for asq in range(81):
                attacker = board[asq]
                if attacker is None or attacker[1] == stm:
                    continue
                if piece_attacks(board, asq, t):
                    hit = True
                    break
            if hit:
                n += 1
    return n


def drop_mobility(board, hand, stm):
    union = set()
    for t in HAND_TYPES:
        n = hand.get((t, stm), 0)
        if n <= 0:
            continue
        for sq in drop_squares(board, n, t, stm):
            union.add(sq)
    return len(union)


class ShogiGameV02(GameSpec):
    game_id = "shogi"
    num_groups = 9
    vocabs = (648, 648, 972, 45, 45, 1134, 1134, 64, 1)
    tokens = 8
    token_dim = 32
    context_dim = 12
    context_names = ("stm", "phase", "us_hand", "them_hand", "us_promo", "them_promo",
                     "us_board", "them_board", "check", "king_adv", "mat_lead", "move_no")
    feature_version = FEATURE_VERSION_V02

    def __init__(self):
        self.v01 = ShogiGame()

    def extract(self, state):
        feats = self.v01.extract(state)
        board, stm, hand, _ = parse_sfen(state)
        drops = checking_drops(board, hand, stm)
        ring = ring_pressure(board, stm)
        mob = drop_mobility(board, hand, stm)
        out = list(feats)
        out.append((7, 20 + min(len(drops), 9)))
        out.append((7, 30 + min(ring, 9)))
        out.append((7, 40 + min(mob // 9, 9)))
        if drops:
            out.append((8, 0))
        return sorted(set(out))

    def context(self, state):
        return self.v01.context(state)

    def phase(self, state):
        return self.v01.phase(state)

    def normalize(self, state):
        return self.v01.normalize(state)

    def phase_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 10) & (g7 <= 12)
        cand = torch.where(valid, g7, torch.full_like(g7, 10))
        return (cand.amax(dim=1) - 10).clamp(0, 2).long()
