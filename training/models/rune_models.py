import torch
import torch.nn as nn
import torch.nn.functional as F

from training.features.python_features import NUM_GROUPS, TOKENS, TOKEN_DIM, VOCAB_SIZES

ARCH_IDS = ["RUNE-SFNN", "RUNE-MLP", "RUNE-ATTN", "RUNE-ATTN-GAB"]

EXPORT_ORDER = {
    "RUNE-SFNN": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-MLP": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-ATTN": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
    "RUNE-ATTN-GAB": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
}


def clip01(x):
    return torch.clamp(x, 0.0, 1.0)


class GroupedEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(v, TOKEN_DIM) for v in VOCAB_SIZES])
        for emb in self.tables:
            nn.init.uniform_(emb.weight, -0.01, 0.01)

    def forward(self, group_ids, group_mask):
        toks = []
        for g in range(8):
            e = self.tables[g](group_ids[g])
            e = (e * group_mask[g].unsqueeze(-1)).sum(dim=1)
            toks.append(e)
        e8 = self.tables[8](group_ids[8])
        e8 = (e8 * group_mask[8].unsqueeze(-1)).sum(dim=1)
        toks[0] = toks[0] + e8
        x = torch.stack(toks, dim=1)
        return clip01(x)


class ValueWdlHead(nn.Module):
    def __init__(self, hidden1=128, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(TOKENS * TOKEN_DIM, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fcv = nn.Linear(hidden2, 1)
        self.fcwdl = nn.Linear(hidden2, 3)

    def forward(self, flat):
        h1 = clip01(self.fc1(flat))
        h2 = clip01(self.fc2(h1))
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        wdl = self.fcwdl(h2)
        return value, wdl


class SfnnHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(TOKENS * TOKEN_DIM, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fcv = nn.Linear(32, 1)
        self.fcwdl = nn.Linear(32, 3)

    def forward(self, flat):
        h1 = clip01(self.fc1(flat))
        h2 = clip01(self.fc2(h1))
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        wdl = self.fcwdl(h2)
        return value, wdl


class RuneAttention(nn.Module):
    def __init__(self, use_gab):
        super().__init__()
        self.use_gab = use_gab
        self.wq = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.wk = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.wv = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.gab = nn.Parameter(torch.zeros(TOKENS, TOKENS))

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        s = q @ k.transpose(-1, -2)
        if self.use_gab:
            s = s + self.gab
        a = torch.clamp(s, 0.0, 1.0)
        y = a @ v
        return x + y


class RuneFullModel(nn.Module):
    def __init__(self, arch_id):
        super().__init__()
        if arch_id not in ARCH_IDS:
            raise ValueError(f"unknown arch {arch_id}")
        self.arch_id = arch_id
        self.embedder = GroupedEmbedder()
        if arch_id in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
            self.attn = RuneAttention(use_gab=(arch_id == "RUNE-ATTN-GAB"))
            self.head = ValueWdlHead()
        elif arch_id == "RUNE-MLP":
            self.attn = None
            self.head = ValueWdlHead()
        else:
            self.attn = None
            self.head = SfnnHead()

    def forward(self, group_ids, group_mask):
        x = self.embedder(group_ids, group_mask)
        if self.attn is not None:
            x = self.attn(x)
        flat = x.reshape(x.size(0), -1)
        return self.head(flat)

    def arch_tensors(self):
        if self.arch_id in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
            return {
                "wq": self.attn.wq.weight.detach(),
                "bq": self.attn.wq.bias.detach(),
                "wk": self.attn.wk.weight.detach(),
                "bk": self.attn.wk.bias.detach(),
                "wvv": self.attn.wv.weight.detach(),
                "bvv": self.attn.wv.bias.detach(),
                "gab": self.attn.gab.detach(),
                "w1": self.head.fc1.weight.detach(),
                "b1": self.head.fc1.bias.detach(),
                "w2": self.head.fc2.weight.detach(),
                "b2": self.head.fc2.bias.detach(),
                "wvo": self.head.fcv.weight.detach(),
                "bvo": self.head.fcv.bias.detach(),
                "wwdl": self.head.fcwdl.weight.detach(),
                "bwdl": self.head.fcwdl.bias.detach(),
            }
        return {
            "w1": self.head.fc1.weight.detach(),
            "b1": self.head.fc1.bias.detach(),
            "w2": self.head.fc2.weight.detach(),
            "b2": self.head.fc2.bias.detach(),
            "wv": self.head.fcv.weight.detach(),
            "bv": self.head.fcv.bias.detach(),
            "wwdl": self.head.fcwdl.weight.detach(),
            "bwdl": self.head.fcwdl.bias.detach(),
        }

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(NUM_GROUPS)}

    def export_order(self):
        return list(EXPORT_ORDER[self.arch_id])

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
        return {
            "arch": self.arch_id,
            "arch_version": "0.1.0",
            "feature_set": "grouped_hkav2_fullthreats_v02",
            "tokens": TOKENS,
            "token_dim": TOKEN_DIM,
            "attention": attention,
            "geometric_bias": gab,
            "head": "value_wdl",
            "quantization": quantization,
        }


def build_model(arch_id):
    return RuneFullModel(arch_id)
