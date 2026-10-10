import torch

from training.games.base import GameSpec
from training.games.go_planes import PLANES_V02, planes_v02, token_features_v02
from training.games.go_rules import apply_move, area_score, legal_moves
from training.games.go_state import context_v02, decode_state, encode_state, game_phase, normalize_state


FEATURE_VERSION_V02 = "go_planes_v02"
CONTEXT_DIM_V02 = 12
CONTEXT_NAMES_V02 = ("stm", "us_stones", "them_stones", "komi", "move_no", "pass_no", "ko", "lib1", "lib2", "lib3", "empty", "phase")


class GoGameV02(GameSpec):
    game_id = "go"
    num_groups = 9
    vocabs = (361, 361, 361, 64, 64, 361, 361, 64, 18)
    tokens = 8
    token_dim = 32
    context_dim = CONTEXT_DIM_V02
    context_names = CONTEXT_NAMES_V02
    feature_version = FEATURE_VERSION_V02

    def __init__(self, size=9):
        self.size = size

    def extract(self, state):
        return token_features_v02(state, self.size)

    def planes(self, state):
        return planes_v02(state, self.size)

    def context(self, state):
        return context_v02(state, self.size)

    def phase(self, state):
        d = decode_state(state, self.size)
        return game_phase(d["move_no"], d["n"])

    def normalize(self, state):
        return normalize_state(state, self.size)

    def phase_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 20) & (g7 <= 22)
        cand = torch.where(valid, g7, torch.full_like(g7, 20))
        return (cand.amax(dim=1) - 20).clamp(0, 2).long()

    def legal(self, state):
        d = decode_state(state, self.size)
        return legal_moves(d["board"], d["n"], d["stm"], d["ko"])

    def apply(self, state, move):
        d = decode_state(state, self.size)
        nb, new_ko, _ = apply_move(d["board"], d["n"], d["stm"], int(move), d["ko"])
        if d["stm"] == 0:
            nstm = 1
        else:
            nstm = 0
        nmove = int(d["move_no"]) + 1
        if int(move) == -1:
            npass = int(d["pass_no"]) + 1
        else:
            npass = 0
        return encode_state(nb, d["n"], nstm, new_ko, d["komi"], nmove, npass)

    def score(self, state):
        d = decode_state(state, self.size)
        return area_score(d["board"], d["n"], d["komi"])
