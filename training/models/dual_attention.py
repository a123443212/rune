import torch.nn as nn

from training.models.attention import RuneAttention
from training.models.heads import BucketedHead, GroupedEmbedder, ValueWdlHead


class RuneDualAttention(nn.Module):
    def __init__(self, buckets=1, gate="clip", pair=False, game="chess"):
        super().__init__()
        if gate not in ("clip", "hard_sigmoid", "screlu"):
            raise ValueError(f"unknown gate {gate}")
        if buckets not in (1, 3):
            raise ValueError(f"head buckets must be 1 or 3, got {buckets}")
        from training.games import get as get_game
        self.game = get_game(game)
        self.tokens_n = self.game.tokens
        self.dim_n = self.game.token_dim
        self.input_dim = self.tokens_n * self.dim_n
        self.arch_id = "RUNE-ATTN-DUAL"
        self.buckets = buckets
        self.gate = gate
        self.pair = pair
        self.embedder = GroupedEmbedder(self.game.vocabs, self.dim_n)
        phase_fn = self.game.phase_from_ids
        self.attn1 = RuneAttention(use_gab=False, gate=gate, tokens=self.tokens_n, token_dim=self.dim_n)
        self.attn2 = RuneAttention(use_gab=False, gate=gate, tokens=self.tokens_n, token_dim=self.dim_n)
        if buckets == 3:
            self.head = BucketedHead(ValueWdlHead, pair=pair, input_dim=self.input_dim, phase_fn=phase_fn)
        else:
            self.head = ValueWdlHead(pair=pair, input_dim=self.input_dim)

    def forward(self, group_ids, group_mask):
        x = self.embedder(group_ids, group_mask)
        x = self.attn1(x)
        x = self.attn2(x)
        flat = x.reshape(x.size(0), -1)
        if self.buckets == 3:
            phase = self.head.phases_from_ids(group_ids, group_mask)
            return self.head(flat, phase)
        return self.head(flat)

    def arch_tensors(self):
        d = {
            "wq": self.attn1.wq.weight.detach(),
            "bq": self.attn1.wq.bias.detach(),
            "wk": self.attn1.wk.weight.detach(),
            "bk": self.attn1.wk.bias.detach(),
            "wvv": self.attn1.wv.weight.detach(),
            "bvv": self.attn1.wv.bias.detach(),
            "gab": self.attn1.gab.detach(),
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

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(self.game.num_groups)}

    def export_order(self):
        order = ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab",
                 "wq2", "bq2", "wk2", "bk2", "wvv2", "bvv2", "gab2",
                 "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]
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
        return {
            "game": self.game.game_id,
            "arch": self.arch_id,
            "arch_version": "0.2.0" if self.pair else "0.1.0",
            "feature_set": self.game.feature_version,
            "tokens": self.tokens_n,
            "token_dim": self.dim_n,
            "attention": "gated_linear_x2",
            "geometric_bias": "none",
            "head": "value_wdl_pair" if self.pair else "value_wdl",
            "head_buckets": self.buckets,
            "head_pair": self.pair,
            "gate": self.gate,
            "quantization": quantization,
        }


def build_dual_attention(buckets=1, gate="clip", pair=False, game="chess"):
    return RuneDualAttention(buckets=buckets, gate=gate, pair=pair, game=game)
