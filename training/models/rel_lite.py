# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

import torch.nn as nn

from training.models.relational import RelationalHead, RelationalMixer, FlexEmbedder

LITE_TOKENS = 6
LITE_DIM = 24


class RuneRelLite(nn.Module):
    def __init__(self, gate="clip", alpha=1.0, pair=False, game="chess", vocab_sizes=None, ctx_dim=None):
        super().__init__()
        if gate not in ("clip", "hard_sigmoid", "screlu"):
            raise ValueError(f"unknown gate {gate}")
        from training.games import get as get_game
        self.game = get_game(game)
        self.tokens = LITE_TOKENS
        self.dim = LITE_DIM
        self.gate = gate
        self.alpha = alpha
        self.pair = pair
        vocabs = list(vocab_sizes) if vocab_sizes is not None else list(self.game.vocabs)
        cd = ctx_dim if ctx_dim is not None else self.game.context_dim
        self.ctx_dim = cd
        self.embedder = FlexEmbedder(LITE_TOKENS, LITE_DIM, vocabs)
        self.mixer = RelationalMixer(LITE_TOKENS, LITE_DIM, gate, alpha, False, ctx_dim=cd)
        self.head = RelationalHead(LITE_TOKENS, LITE_DIM, pair=pair)

    def forward(self, group_ids, group_mask, ctx):
        x = self.embedder(group_ids, group_mask)
        x = self.mixer(x, ctx)
        return self.head(x.reshape(x.size(0), -1))

    def arch_tensors(self):
        d = {
            "wq": self.mixer.wq.weight.detach(),
            "bq": self.mixer.wq.bias.detach(),
            "wk": self.mixer.wk.weight.detach(),
            "bk": self.mixer.wk.bias.detach(),
            "wvv": self.mixer.wv.weight.detach(),
            "bvv": self.mixer.wv.bias.detach(),
            "gabS": self.mixer.gab_static.detach(),
            "w1": self.head.fc1.weight.detach(),
            "b1": self.head.fc1.bias.detach(),
            "w2": self.head.fc2.weight.detach(),
            "b2": self.head.fc2.bias.detach(),
            "wvo": self.head.fcv.weight.detach(),
            "bvo": self.head.fcv.bias.detach(),
            "wwdl": self.head.fcwdl.weight.detach(),
            "bwdl": self.head.fcwdl.bias.detach(),
        }
        return d

    def export_order(self):
        return ["wq", "bq", "wk", "bk", "wvv", "bvv", "gabS",
                "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(len(self.embedder.tables))}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_bytes(self):
        return self.parameter_count() * 4

    def model_spec(self, quantization="fp32"):
        return {
            "game": self.game.game_id,
            "arch": "RUNE-REL-LITE",
            "arch_version": "0.2.1" if self.pair else "0.1.0",
            "feature_set": self.game.feature_version,
            "tokens": self.tokens,
            "token_dim": self.dim,
            "attention": "gated_relational",
            "geometric_bias": "static",
            "head": "value_wdl_pair" if self.pair else "value_wdl",
            "head_pair": self.pair,
            "quantization": quantization,
            "gate": self.gate,
            "alpha": self.alpha,
            "context_dim": self.ctx_dim,
        }


def build_rel_lite(gate="clip", alpha=1.0, pair=False, game="chess", vocab_sizes=None, ctx_dim=None):
    return RuneRelLite(gate=gate, alpha=alpha, pair=pair, game=game, vocab_sizes=vocab_sizes, ctx_dim=ctx_dim)
