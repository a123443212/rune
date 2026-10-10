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

from training.models.activations import apply_gate, clip01
from training.models.attention import RuneAttention
from training.models.heads import BucketedHead, GroupedEmbedder, SfnnHead, ValueWdlHead

ARCH_IDS = ["RUNE-SFNN", "RUNE-MLP", "RUNE-ATTN", "RUNE-ATTN-GAB", "RUNE-ATTN-MH4", "RUNE-MLP-S", "RUNE-SFNN-C", "RUNE-ATTN-DUAL"]

EXPORT_ORDER = {
    "RUNE-SFNN": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-MLP": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-MLP-S": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-SFNN-C": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-ATTN-DUAL": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "wq2", "bq2", "wk2", "bk2", "wvv2", "bvv2", "gab2", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
    "RUNE-ATTN": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
    "RUNE-ATTN-GAB": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
}


class RuneFullModel(nn.Module):
    def __init__(self, arch_id, buckets=1, gate="clip", pair=False, game="chess"):
        super().__init__()
        if arch_id not in ARCH_IDS:
            raise ValueError(f"unknown arch {arch_id}")
        if buckets not in (1, 3):
            raise ValueError(f"head buckets must be 1 or 3, got {buckets}")
        from training.games import get as get_game
        self.game = get_game(game)
        self.tokens_n = self.game.tokens
        self.dim_n = self.game.token_dim
        self.input_dim = self.tokens_n * self.dim_n
        self.arch_id = arch_id
        self.buckets = buckets
        self.gate = gate
        self.pair = pair
        self.embedder = GroupedEmbedder(self.game.vocabs, self.dim_n)
        phase_fn = self.game.phase_from_ids
        self.attn2 = None
        if arch_id in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
            self.attn = RuneAttention(use_gab=(arch_id == "RUNE-ATTN-GAB"), gate=gate,
                                      tokens=self.tokens_n, token_dim=self.dim_n)
            self.head = BucketedHead(ValueWdlHead, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else ValueWdlHead(pair=pair, input_dim=self.input_dim)
        elif arch_id == "RUNE-ATTN-DUAL":
            self.attn = RuneAttention(use_gab=False, gate=gate,
                                      tokens=self.tokens_n, token_dim=self.dim_n)
            from training.models.dual_attention import RuneDualAttention
            self.attn2 = RuneAttention(use_gab=False, gate=gate,
                                       tokens=self.tokens_n, token_dim=self.dim_n)
            self.head = BucketedHead(ValueWdlHead, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else ValueWdlHead(pair=pair, input_dim=self.input_dim)
        elif arch_id == "RUNE-ATTN-MH4":
            from training.models.multi_head import RuneMultiHeadMixer
            self.attn = RuneMultiHeadMixer(gate=gate)
            self.head = BucketedHead(ValueWdlHead, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else ValueWdlHead(pair=pair, input_dim=self.input_dim)
        elif arch_id == "RUNE-MLP":
            self.attn = None
            self.head = BucketedHead(ValueWdlHead, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else ValueWdlHead(pair=pair, input_dim=self.input_dim)
        elif arch_id == "RUNE-MLP-S":
            from training.models.mlp_small import H1 as S_H1
            from training.models.mlp_small import H2 as S_H2
            self.attn = None
            mk = lambda pair=pair, input_dim=None: ValueWdlHead(S_H1, S_H2, pair, input_dim or self.input_dim)
            self.head = BucketedHead(mk, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else ValueWdlHead(S_H1, S_H2, pair, self.input_dim)
        elif arch_id == "RUNE-SFNN-C":
            from training.models.sfnn_compact import H1 as C_H1
            from training.models.sfnn_compact import H2 as C_H2
            self.attn = None
            mkc = lambda pair=pair, input_dim=None: ValueWdlHead(C_H1, C_H2, pair, input_dim or self.input_dim)
            self.head = BucketedHead(mkc, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else ValueWdlHead(C_H1, C_H2, pair, self.input_dim)
        else:
            self.attn = None
            self.head = BucketedHead(SfnnHead, pair=pair, input_dim=self.input_dim,
                                     phase_fn=phase_fn) if buckets == 3 else SfnnHead(pair=pair, input_dim=self.input_dim)

    def forward(self, group_ids, group_mask):
        x = self.embedder(group_ids, group_mask)
        if self.attn is not None:
            x = self.attn(x)
        if self.attn2 is not None:
            x = self.attn2(x)
        flat = x.reshape(x.size(0), -1)
        if self.buckets == 3:
            phase = self.head.phases_from_ids(group_ids, group_mask)
            return self.head(flat, phase)
        return self.head(flat)

    def arch_tensors(self):
        if self.arch_id == "RUNE-ATTN-DUAL":
            d = {
                "wq": self.attn.wq.weight.detach(),
                "bq": self.attn.wq.bias.detach(),
                "wk": self.attn.wk.weight.detach(),
                "bk": self.attn.wk.bias.detach(),
                "wvv": self.attn.wv.weight.detach(),
                "bvv": self.attn.wv.bias.detach(),
                "gab": self.attn.gab.detach(),
                "wq2": self.attn2.wq.weight.detach(),
                "bq2": self.attn2.wq.bias.detach(),
                "wk2": self.attn2.wk.weight.detach(),
                "bk2": self.attn2.wk.bias.detach(),
                "wvv2": self.attn2.wv.weight.detach(),
                "bvv2": self.attn2.wv.bias.detach(),
                "gab2": self.attn2.gab.detach(),
            }
            heads = self.head.heads if self.buckets == 3 else [self.head]
            for b, h in enumerate(heads):
                suffix = f"_b{b}" if self.buckets == 3 else ""
                d["w1" + suffix] = h.fc1.weight.detach()
                d["b1" + suffix] = h.fc1.bias.detach()
                d["w2" + suffix] = h.fc2.weight.detach()
                d["b2" + suffix] = h.fc2.bias.detach()
                d["wvo" + suffix] = h.fcv.weight.detach()
                d["bvo" + suffix] = h.fcv.bias.detach()
                d["wwdl" + suffix] = h.fcwdl.weight.detach()
                d["bwdl" + suffix] = h.fcwdl.bias.detach()
            return d
        if self.arch_id == "RUNE-ATTN-MH4":
            d = self.attn.arch_tensors()
            heads = self.head.heads if self.buckets == 3 else [self.head]
            for b, h in enumerate(heads):
                suffix = f"_b{b}" if self.buckets == 3 else ""
                d["w1" + suffix] = h.fc1.weight.detach()
                d["b1" + suffix] = h.fc1.bias.detach()
                d["w2" + suffix] = h.fc2.weight.detach()
                d["b2" + suffix] = h.fc2.bias.detach()
                d["wvo" + suffix] = h.fcv.weight.detach()
                d["bvo" + suffix] = h.fcv.bias.detach()
                d["wwdl" + suffix] = h.fcwdl.weight.detach()
                d["bwdl" + suffix] = h.fcwdl.bias.detach()
            return d
        if self.arch_id in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
            d = {
                "wq": self.attn.wq.weight.detach(),
                "bq": self.attn.wq.bias.detach(),
                "wk": self.attn.wk.weight.detach(),
                "bk": self.attn.wk.bias.detach(),
                "wvv": self.attn.wv.weight.detach(),
                "bvv": self.attn.wv.bias.detach(),
                "gab": self.attn.gab.detach(),
            }
            heads = self.head.heads if self.buckets == 3 else [self.head]
            for b, h in enumerate(heads):
                suffix = f"_b{b}" if self.buckets == 3 else ""
                d["w1" + suffix] = h.fc1.weight.detach()
                d["b1" + suffix] = h.fc1.bias.detach()
                d["w2" + suffix] = h.fc2.weight.detach()
                d["b2" + suffix] = h.fc2.bias.detach()
                d["wvo" + suffix] = h.fcv.weight.detach()
                d["bvo" + suffix] = h.fcv.bias.detach()
                d["wwdl" + suffix] = h.fcwdl.weight.detach()
                d["bwdl" + suffix] = h.fcwdl.bias.detach()
            return d
        d = {}
        heads = self.head.heads if self.buckets == 3 else [self.head]
        for b, h in enumerate(heads):
            suffix = f"_b{b}" if self.buckets == 3 else ""
            d["w1" + suffix] = h.fc1.weight.detach()
            d["b1" + suffix] = h.fc1.bias.detach()
            d["w2" + suffix] = h.fc2.weight.detach()
            d["b2" + suffix] = h.fc2.bias.detach()
            d["wv" + suffix] = h.fcv.weight.detach()
            d["bv" + suffix] = h.fcv.bias.detach()
            d["wwdl" + suffix] = h.fcwdl.weight.detach()
            d["bwdl" + suffix] = h.fcwdl.bias.detach()
        return d

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(self.game.num_groups)}

    def export_order(self):
        if self.arch_id == "RUNE-ATTN-MH4":
            from training.models.multi_head import RuneMultiHeadMixer
            order = RuneMultiHeadMixer.export_order() + [
                "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]
        else:
            order = list(EXPORT_ORDER[self.arch_id])
        if self.buckets == 3:
            head_keys = ("w1", "b1", "w2", "b2", "wvo", "bvo", "wv", "bv", "wwdl", "bwdl")
            base = [k for k in order if k not in head_keys]
            out = list(base)
            for b in range(3):
                out += [f"{k}_b{b}" for k in order if k in head_keys]
            return out
        return order

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_bytes(self):
        return self.parameter_count() * 4

    def model_spec(self, quantization="fp32"):
        attention = "none"
        gab = "none"
        if self.arch_id in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
            attention = "gated_linear"
            gab = "learned" if self.arch_id == "RUNE-ATTN-GAB" else "none"
        if self.arch_id == "RUNE-ATTN-DUAL":
            attention = "gated_linear_x2"
            gab = "none"
        if self.arch_id == "RUNE-ATTN-MH4":
            attention = "multi_head"
            gab = "per_head"
        return {
            "game": self.game.game_id,
            "arch": self.arch_id,
            "arch_version": "0.2.0" if self.pair else "0.1.0",
            "feature_set": self.game.feature_version,
            "tokens": self.tokens_n,
            "token_dim": self.dim_n,
            "attention": attention,
            "geometric_bias": gab,
            "head": "value_wdl_pair" if self.pair else "value_wdl",
            "head_buckets": self.buckets,
            "head_pair": self.pair,
            "gate": self.gate if self.attn is not None else "clip",
            "quantization": quantization,
        }


def build_model(arch_id, gate="clip", pair=False, game="chess", buckets=1):
    return RuneFullModel(arch_id, buckets=buckets, gate=gate, pair=pair, game=game)
