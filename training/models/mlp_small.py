import torch.nn as nn

from training.models.heads import BucketedHead, GroupedEmbedder, ValueWdlHead

H1 = 64
H2 = 16


class RuneMlpSmall(nn.Module):
    def __init__(self, buckets=1, pair=False, game="chess"):
        super().__init__()
        if buckets not in (1, 3):
            raise ValueError(f"head buckets must be 1 or 3, got {buckets}")
        from training.games import get as get_game
        self.game = get_game(game)
        self.tokens_n = self.game.tokens
        self.dim_n = self.game.token_dim
        self.input_dim = self.tokens_n * self.dim_n
        self.arch_id = "RUNE-MLP-S"
        self.buckets = buckets
        self.pair = pair
        self.embedder = GroupedEmbedder(self.game.vocabs, self.dim_n)
        phase_fn = self.game.phase_from_ids
        if buckets == 3:
            self.head = BucketedHead(
                lambda pair=pair, input_dim=None: ValueWdlHead(H1, H2, pair, input_dim or self.input_dim),
                pair=pair, input_dim=self.input_dim, phase_fn=phase_fn)
        else:
            self.head = ValueWdlHead(H1, H2, pair, self.input_dim)

    def forward(self, group_ids, group_mask):
        x = self.embedder(group_ids, group_mask)
        flat = x.reshape(x.size(0), -1)
        if self.buckets == 3:
            phase = self.head.phases_from_ids(group_ids, group_mask)
            return self.head(flat, phase)
        return self.head(flat)

    def arch_tensors(self):
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
        order = ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"]
        if self.buckets == 3:
            out = []
            for b in range(3):
                out += [f"{k}_b{b}" for k in order]
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
            "attention": "none",
            "geometric_bias": "none",
            "head": "value_wdl_pair" if self.pair else "value_wdl",
            "head_buckets": self.buckets,
            "head_pair": self.pair,
            "head_h1": H1,
            "head_h2": H2,
            "gate": "clip",
            "quantization": quantization,
        }


def build_mlp_small(buckets=1, pair=False, game="chess"):
    return RuneMlpSmall(buckets=buckets, pair=pair, game=game)
