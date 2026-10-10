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
from training.models.soft_attention import RuneSoftAttention
from training.models.swiglu_head import SwiGluWdlHead

ARCH_IDS = ["RUNE-SFNN", "RUNE-MLP", "RUNE-ATTN-GAB", "RUNE-ATTN-SOFT"]

FROZEN_IDS = ("RUNE-ATTN", "RUNE-ATTN-DUAL", "RUNE-ATTN-MH4", "RUNE-REL-LITE", "RUNE-MLP-S", "RUNE-SFNN-C")

EXPORT_ORDER = {
    "RUNE-SFNN": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-MLP": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-ATTN-GAB": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
    "RUNE-ATTN-SOFT": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
}

HEAD_KEYS = ("w1", "b1", "wgate", "bgate", "wup", "bup", "w2", "b2", "wvo", "bvo", "wv", "bv", "wwdl", "bwdl")


class RuneFullModel(nn.Module):
    def __init__(self, arch_id, buckets=1, gate="clip", pair=False, game="chess",
                 head="value_wdl", head_h1=None, head_h2=None):
        super().__init__()
        if arch_id not in ARCH_IDS:
            raise ValueError(f"unknown arch {arch_id}")
        if buckets not in (1, 3):
            raise ValueError(f"head buckets must be 1 or 3, got {buckets}")
        if head not in ("value_wdl", "value_swiglu"):
            raise ValueError(f"unknown head {head}")
        if head == "value_swiglu" and pair:
            raise ValueError("swiglu head does not support pair")
        from training.games import get as get_game
        self.game = get_game(game)
        self.tokens_n = self.game.tokens
        self.dim_n = self.game.token_dim
        self.input_dim = self.tokens_n * self.dim_n
        self.arch_id = arch_id
        self.buckets = buckets
        self.gate = "softmax" if arch_id == "RUNE-ATTN-SOFT" else gate
        self.pair = pair
        self.head_kind = head
        if head_h1 is None:
            head_h1 = 256 if arch_id == "RUNE-SFNN" else 128
        if head_h2 is None:
            head_h2 = 32
        self.head_h1 = head_h1
        self.head_h2 = head_h2
        self.embedder = GroupedEmbedder(self.game.vocabs, self.dim_n)
        phase_fn = self.game.phase_from_ids
        if arch_id == "RUNE-ATTN-GAB":
            self.attn = RuneAttention(use_gab=True, gate=gate,
                                      tokens=self.tokens_n, token_dim=self.dim_n)
        elif arch_id == "RUNE-ATTN-SOFT":
            self.attn = RuneSoftAttention(use_gab=True,
                                          tokens=self.tokens_n, token_dim=self.dim_n)
        else:
            self.attn = None
        self.head = self._build_head(phase_fn)

    def _make_single_head(self, input_dim):
        if self.head_kind == "value_swiglu":
            return SwiGluWdlHead(self.head_h1, self.head_h2, input_dim)
        if self.arch_id == "RUNE-SFNN":
            return SfnnHead(self.head_h1, self.head_h2, self.pair, input_dim)
        return ValueWdlHead(self.head_h1, self.head_h2, self.pair, input_dim)

    def _build_head(self, phase_fn):
        if self.buckets == 3:
            mk = lambda pair=self.pair, input_dim=None: self._make_single_head(
                input_dim if input_dim is not None else self.input_dim)
            return BucketedHead(mk, pair=self.pair, input_dim=self.input_dim, phase_fn=phase_fn)
        return self._make_single_head(self.input_dim)

    def forward(self, group_ids, group_mask):
        x = self.embedder(group_ids, group_mask)
        if self.attn is not None:
            x = self.attn(x)
        flat = x.reshape(x.size(0), -1)
        if self.buckets == 3:
            phase = self.head.phases_from_ids(group_ids, group_mask)
            return self.head(flat, phase)
        return self.head(flat)

    def _head_tensors(self):
        d = {}
        heads = self.head.heads if self.buckets == 3 else [self.head]
        for b, h in enumerate(heads):
            suffix = f"_b{b}" if self.buckets == 3 else ""
            if isinstance(h, SwiGluWdlHead):
                d["wgate" + suffix] = h.fc_gate.weight.detach()
                d["bgate" + suffix] = h.fc_gate.bias.detach()
                d["wup" + suffix] = h.fc_up.weight.detach()
                d["bup" + suffix] = h.fc_up.bias.detach()
                d["w2" + suffix] = h.fc2.weight.detach()
                d["b2" + suffix] = h.fc2.bias.detach()
                vkey = "wvo" if self.attn is not None else "wv"
                bkey = "bvo" if self.attn is not None else "bv"
                d[vkey + suffix] = h.fcv.weight.detach()
                d[bkey + suffix] = h.fcv.bias.detach()
                d["wwdl" + suffix] = h.fcwdl.weight.detach()
                d["bwdl" + suffix] = h.fcwdl.bias.detach()
            else:
                d["w1" + suffix] = h.fc1.weight.detach()
                d["b1" + suffix] = h.fc1.bias.detach()
                d["w2" + suffix] = h.fc2.weight.detach()
                d["b2" + suffix] = h.fc2.bias.detach()
                vkey = "wvo" if self.attn is not None else "wv"
                bkey = "bvo" if self.attn is not None else "bv"
                d[vkey + suffix] = h.fcv.weight.detach()
                d[bkey + suffix] = h.fcv.bias.detach()
                d["wwdl" + suffix] = h.fcwdl.weight.detach()
                d["bwdl" + suffix] = h.fcwdl.bias.detach()
        return d

    def arch_tensors(self):
        if self.attn is not None:
            d = {
                "wq": self.attn.wq.weight.detach(),
                "bq": self.attn.wq.bias.detach(),
                "wk": self.attn.wk.weight.detach(),
                "bk": self.attn.wk.bias.detach(),
                "wvv": self.attn.wv.weight.detach(),
                "bvv": self.attn.wv.bias.detach(),
                "gab": self.attn.gab.detach(),
            }
            d.update(self._head_tensors())
            return d
        return self._head_tensors()

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(self.game.num_groups)}

    def export_order(self):
        order = list(EXPORT_ORDER[self.arch_id])
        if self.head_kind == "value_swiglu":
            order = ["wgate", "bgate", "wup", "bup"] + [k for k in order if k not in ("w1", "b1")]
        if self.buckets == 3:
            base = [k for k in order if k not in HEAD_KEYS]
            out = list(base)
            for b in range(3):
                out += [f"{k}_b{b}" for k in order if k in HEAD_KEYS]
            return out
        return order

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_bytes(self):
        return self.parameter_count() * 4

    def model_spec(self, quantization="fp32"):
        attention = "none"
        gab = "none"
        if self.arch_id == "RUNE-ATTN-GAB":
            attention = "gated_linear"
            gab = "learned"
        if self.arch_id == "RUNE-ATTN-SOFT":
            attention = "softmax_scaled"
            gab = "learned"
        if self.head_kind == "value_swiglu":
            head_name = "value_swiglu"
            version = "0.3.0"
        else:
            head_name = "value_wdl_pair" if self.pair else "value_wdl"
            version = "0.2.0" if self.pair else "0.1.0"
        return {
            "game": self.game.game_id,
            "arch": self.arch_id,
            "arch_version": version,
            "feature_set": self.game.feature_version,
            "tokens": self.tokens_n,
            "token_dim": self.dim_n,
            "attention": attention,
            "geometric_bias": gab,
            "head": head_name,
            "head_buckets": self.buckets,
            "head_pair": self.pair,
            "gate": self.gate,
            "quantization": quantization,
        }


def build_model(arch_id, gate="clip", pair=False, game="chess", buckets=1,
                head="value_wdl", head_h1=None, head_h2=None):
    return RuneFullModel(arch_id, buckets=buckets, gate=gate, pair=pair, game=game,
                         head=head, head_h1=head_h1, head_h2=head_h2)
