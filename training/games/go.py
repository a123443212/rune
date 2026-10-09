import torch

from training.games.base import GameSpec

FEATURE_VERSION = "go_planes_v01"
CONTEXT_DIM = 12
CONTEXT_NAMES = ["stm", "us_stones", "them_stones", "komi", "move_no", "pass_no", "ko", "lib1", "lib2", "lib3", "empty", "phase"]


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def parse_state(state, size=9):
    parts = state.split()
    grid = parts[0]
    rows = grid.split("/")
    stm = 0
    if len(parts) > 1 and parts[1] in ("w", "O"):
        stm = 1
    return rows, stm


class GoGame(GameSpec):
    game_id = "go"
    num_groups = 9
    vocabs = (361, 361, 361, 64, 64, 361, 361, 64, 18)
    tokens = 8
    token_dim = 32
    context_dim = CONTEXT_DIM
    context_names = tuple(CONTEXT_NAMES)
    feature_version = FEATURE_VERSION

    def __init__(self, size=9):
        self.size = size

    def extract(self, state):
        rows, stm = parse_state(state, self.size)
        feats = []
        n = self.size
        for r in range(min(n, len(rows))):
            row = rows[r]
            for c in range(min(n, len(row))):
                ch = row[c]
                if ch == ".":
                    continue
                mine = (ch == "X" and stm == 0) or (ch == "O" and stm == 1)
                ci = 0 if mine else 1
                sq = r * n + c
                feats.append((0, (ci * 181 + sq) % 361))
                feats.append((6, sq % 361))
        feats.append((7, stm))
        feats.append((7, 2))
        return sorted(set(feats))

    def planes(self, state):
        rows, stm = parse_state(state, self.size)
        n = self.size
        import numpy as np
        out = np.zeros((1, 1, n, n), dtype=np.float32)
        for r in range(min(n, len(rows))):
            row = rows[r]
            for c in range(min(n, len(row))):
                ch = row[c]
                if ch == ".":
                    out[0, 0, r, c] = 0.0
                elif (ch == "X" and stm == 0) or (ch == "O" and stm == 1):
                    out[0, 0, r, c] = 1.0
                else:
                    out[0, 0, r, c] = -1.0
        return out

    def context(self, state):
        rows, stm = parse_state(state, self.size)
        n = self.size
        us = 0
        them = 0
        empty = 0
        for r in range(min(n, len(rows))):
            for ch in rows[r][:n]:
                if ch == ".":
                    empty += 1
                elif (ch == "X" and stm == 0) or (ch == "O" and stm == 1):
                    us += 1
                elif ch in ("X", "O"):
                    them += 1
        total = float(n * n)
        return [clamp01(float(stm)), clamp01(us / total), clamp01(them / total), 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, clamp01(empty / total), 0.0]

    def phase(self, state):
        return 0

    def normalize(self, state):
        return state.split()[0]

    def phase_from_ids(self, group_ids, group_mask):
        return torch.zeros(group_ids[0].size(0), dtype=torch.long)
