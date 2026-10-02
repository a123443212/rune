import torch
import torch.nn as nn

from training.features.context import CONTEXT_DIM
from training.features.python_features import NUM_GROUPS, VOCAB_SIZES
from training.models.token_layout import check_layout

GATE_FNS = ("clip", "hard_sigmoid")
DYN_CAP = 0.25


def apply_gate(name, s):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * s + 0.5, 0.0, 1.0)
    return torch.clamp(s, 0.0, 1.0)


class FlexEmbedder(nn.Module):
    def __init__(self, tokens=8, dim=32):
        super().__init__()
        self.layout = check_layout(tokens, dim)
        self.tokens = tokens
        self.dim = dim
        self.tables = nn.ModuleList([nn.Embedding(v, dim) for v in VOCAB_SIZES])
        for emb in self.tables:
            nn.init.uniform_(emb.weight, -0.01, 0.01)

    def forward(self, group_ids, group_mask):
        toks = []
        for sources in self.layout:
            acc = 0.0
            for g, lo, hi in sources:
                ids = group_ids[g].clamp(0, VOCAB_SIZES[g] - 1)
                sel = ((group_ids[g] >= lo) & (group_ids[g] < hi)).float() * group_mask[g]
                acc = acc + (self.tables[g](ids) * sel.unsqueeze(-1)).sum(dim=1)
            toks.append(acc)
        return torch.clamp(torch.stack(toks, dim=1), 0.0, 1.0)


class RelationalMixer(nn.Module):
    def __init__(self, tokens=8, dim=32, gate="clip", alpha=1.0, dynamic_bias=False):
        super().__init__()
        if gate not in GATE_FNS:
            raise ValueError(f"unknown gate {gate}")
        self.tokens = tokens
        self.dim = dim
        self.gate = gate
        self.alpha = alpha
        self.dynamic_bias = dynamic_bias
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.gab_static = nn.Parameter(torch.zeros(tokens, tokens))
        if dynamic_bias:
            self.dyn_u = nn.Parameter(torch.zeros(tokens, CONTEXT_DIM))
            self.dyn_w = nn.Parameter(torch.zeros(tokens, CONTEXT_DIM))
        else:
            self.dyn_u = None
            self.dyn_w = None

    def forward(self, x, ctx):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        s = q @ k.transpose(-1, -2) + self.gab_static
        if self.dynamic_bias:
            u = ctx @ self.dyn_u.t()
            w = ctx @ self.dyn_w.t()
            delta = torch.clamp(u.unsqueeze(-1) * w.unsqueeze(-2), -DYN_CAP, DYN_CAP)
            s = s + delta
        g = apply_gate(self.gate, s)
        return x + self.alpha * (g @ v)


class RelationalHead(nn.Module):
    def __init__(self, tokens=8, dim=32):
        super().__init__()
        self.fc1 = nn.Linear(tokens * dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fcv = nn.Linear(32, 1)
        self.fcwdl = nn.Linear(32, 3)

    def forward(self, flat):
        h1 = torch.clamp(self.fc1(flat), 0.0, 1.0)
        h2 = torch.clamp(self.fc2(h1), 0.0, 1.0)
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        return value, self.fcwdl(h2)


class RuneRelational(nn.Module):
    def __init__(self, tokens=8, dim=32, gate="clip", alpha=1.0, dynamic_bias=False):
        super().__init__()
        self.tokens = tokens
        self.dim = dim
        self.gate = gate
        self.alpha = alpha
        self.dynamic_bias = dynamic_bias
        self.embedder = FlexEmbedder(tokens, dim)
        self.mixer = RelationalMixer(tokens, dim, gate, alpha, dynamic_bias)
        self.head = RelationalHead(tokens, dim)

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
        }
        if self.dynamic_bias:
            d["dynU"] = self.mixer.dyn_u.detach()
            d["dynW"] = self.mixer.dyn_w.detach()
        d.update({
            "w1": self.head.fc1.weight.detach(),
            "b1": self.head.fc1.bias.detach(),
            "w2": self.head.fc2.weight.detach(),
            "b2": self.head.fc2.bias.detach(),
            "wvo": self.head.fcv.weight.detach(),
            "bvo": self.head.fcv.bias.detach(),
            "wwdl": self.head.fcwdl.weight.detach(),
            "bwdl": self.head.fcwdl.bias.detach(),
        })
        return d

    def export_order(self):
        order = ["wq", "bq", "wk", "bk", "wvv", "bvv", "gabS"]
        if self.dynamic_bias:
            order += ["dynU", "dynW"]
        return order + ["w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(NUM_GROUPS)}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_bytes(self):
        return self.parameter_count() * 4

    def model_spec(self, quantization="fp32"):
        return {
            "arch": "RUNE-REL-02",
            "arch_version": "0.2.0",
            "feature_set": "grouped_hkav2_fullthreats_v01",
            "tokens": self.tokens,
            "token_dim": self.dim,
            "attention": "gated_relational",
            "geometric_bias": "dynamic" if self.dynamic_bias else "static",
            "head": "value_wdl",
            "quantization": quantization,
            "gate": self.gate,
            "alpha": self.alpha,
            "context_dim": CONTEXT_DIM,
        }


def build_rel_model(tokens=8, dim=32, gate="clip", alpha=1.0, dynamic_bias=False):
    return RuneRelational(tokens, dim, gate, alpha, dynamic_bias)
