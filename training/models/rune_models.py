import torch
import torch.nn as nn
import torch.nn.functional as F

from training.features.python_features import NUM_GROUPS, TOKENS, TOKEN_DIM, VOCAB_SIZES

ARCH_IDS = ["RUNE-SFNN", "RUNE-MLP", "RUNE-ATTN", "RUNE-ATTN-GAB", "RUNE-ATTN-MH4"]

EXPORT_ORDER = {
    "RUNE-SFNN": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-MLP": ["w1", "b1", "w2", "b2", "wv", "bv", "wwdl", "bwdl"],
    "RUNE-ATTN": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
    "RUNE-ATTN-GAB": ["wq", "bq", "wk", "bk", "wvv", "bvv", "gab", "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"],
}


def clip01(x):
    return torch.clamp(x, 0.0, 1.0)


def apply_gate(name, s):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * s + 0.5, 0.0, 1.0)
    if name == "screlu":
        c = torch.clamp(s, 0.0, 1.0)
        return c * c
    return torch.clamp(s, 0.0, 1.0)


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


class BucketedHead(nn.Module):
    def __init__(self, head_fn=ValueWdlHead):
        super().__init__()
        self.heads = nn.ModuleList([head_fn(), head_fn(), head_fn()])

    def phases_from_ids(self, group_ids, group_mask):
        g7 = group_ids[7].clamp(0, 63)
        valid = (group_mask[7] > 0.5) & (g7 >= 34) & (g7 <= 36)
        cand = torch.where(valid, g7, torch.full_like(g7, 34))
        return (cand.amax(dim=1) - 34).clamp(0, 2).long()

    def forward(self, flat, phase):
        value = torch.empty(flat.size(0), dtype=flat.dtype, device=flat.device)
        wdl = torch.empty(flat.size(0), 3, dtype=flat.dtype, device=flat.device)
        for b in range(3):
            rows = (phase == b).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            v, w = self.heads[b](flat[rows])
            value[rows] = v
            wdl[rows] = w
        return value, wdl


class RuneAttention(nn.Module):
    def __init__(self, use_gab, gate="clip"):
        super().__init__()
        if gate not in ("clip", "hard_sigmoid", "screlu"):
            raise ValueError(f"unknown gate {gate}")
        self.use_gab = use_gab
        self.gate = gate
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
        a = apply_gate(self.gate, s)
        y = a @ v
        return x + y


class RuneFullModel(nn.Module):
    def __init__(self, arch_id, buckets=1, gate="clip"):
        super().__init__()
        if arch_id not in ARCH_IDS:
            raise ValueError(f"unknown arch {arch_id}")
        if buckets not in (1, 3):
            raise ValueError(f"head buckets must be 1 or 3, got {buckets}")
        self.arch_id = arch_id
        self.buckets = buckets
        self.gate = gate
        self.embedder = GroupedEmbedder()
        if arch_id in ("RUNE-ATTN", "RUNE-ATTN-GAB"):
            self.attn = RuneAttention(use_gab=(arch_id == "RUNE-ATTN-GAB"), gate=gate)
            self.head = BucketedHead(ValueWdlHead) if buckets == 3 else ValueWdlHead()
        elif arch_id == "RUNE-ATTN-MH4":
            from training.models.multi_head import RuneMultiHeadMixer
            self.attn = RuneMultiHeadMixer(gate=gate)
            self.head = BucketedHead(ValueWdlHead) if buckets == 3 else ValueWdlHead()
        elif arch_id == "RUNE-MLP":
            self.attn = None
            self.head = BucketedHead(ValueWdlHead) if buckets == 3 else ValueWdlHead()
        else:
            self.attn = None
            self.head = BucketedHead(SfnnHead) if buckets == 3 else SfnnHead()

    def forward(self, group_ids, group_mask):
        x = self.embedder(group_ids, group_mask)
        if self.attn is not None:
            x = self.attn(x)
        flat = x.reshape(x.size(0), -1)
        if self.buckets == 3:
            phase = self.head.phases_from_ids(group_ids, group_mask)
            return self.head(flat, phase)
        return self.head(flat)

    def arch_tensors(self):
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
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(NUM_GROUPS)}

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
        if self.arch_id == "RUNE-ATTN-MH4":
            attention = "multi_head"
            gab = "per_head"
        return {
            "arch": self.arch_id,
            "arch_version": "0.1.0",
            "feature_set": "grouped_hkav2_fullthreats_v02",
            "tokens": TOKENS,
            "token_dim": TOKEN_DIM,
            "attention": attention,
            "geometric_bias": gab,
            "head": "value_wdl",
            "head_buckets": self.buckets,
            "gate": self.gate if self.attn is not None else "clip",
            "quantization": quantization,
        }


def build_model(arch_id, gate="clip"):
    return RuneFullModel(arch_id, gate=gate)
