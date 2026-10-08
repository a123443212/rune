import torch
import torch.nn as nn

VARIANTS = ("A", "B", "C", "D")
POOL_MODES = ("none", "per_token", "shared")
SHARED_WIDTH = 32


def check_dims(dims):
    if len(dims) != 8:
        raise ValueError("token dims must have 8 entries")
    for d in dims:
        if d < 8 or d > 64:
            raise ValueError("token dim out of range")
    return list(dims)


class VarEmbedder(nn.Module):
    def __init__(self, group_widths):
        super().__init__()
        from training.features.python_features import NUM_GROUPS, VOCAB_SIZES

        self.group_widths = list(group_widths)
        self.tables = nn.ModuleList(
            [nn.Embedding(VOCAB_SIZES[g], group_widths[g]) for g in range(NUM_GROUPS)]
        )
        for emb in self.tables:
            nn.init.uniform_(emb.weight, -0.01, 0.01)

    def forward(self, group_ids, group_mask):
        from training.features.python_features import VOCAB_SIZES

        toks = []
        for g in range(8):
            ids = group_ids[g].clamp(0, VOCAB_SIZES[g] - 1)
            e = self.tables[g](ids) * group_mask[g].unsqueeze(-1)
            toks.append(e.sum(dim=1))
        return torch.clamp(torch.cat(toks, dim=1), 0.0, 1.0)


class TokenPool(nn.Module):
    def __init__(self, token_dims, mode="none", pool_clip=True, shared_width=SHARED_WIDTH):
        super().__init__()
        if mode not in POOL_MODES:
            raise ValueError(f"unknown pooling {mode}")
        self.token_dims = check_dims(token_dims)
        self.mode = mode
        self.pool_clip = pool_clip
        self.shared_width = shared_width
        if mode == "per_token":
            self.pw = nn.ParameterList([nn.Parameter(torch.eye(d)) for d in self.token_dims])
            self.pb = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in self.token_dims])
        elif mode == "shared":
            for d in self.token_dims:
                if d > shared_width:
                    raise ValueError("token dim exceeds shared width")
            self.shared_s = nn.Parameter(torch.eye(shared_width))
            self.tok_s = nn.ParameterList(
                [nn.Parameter(torch.ones(d)) for d in self.token_dims]
            )
            self.tok_b = nn.ParameterList(
                [nn.Parameter(torch.zeros(d)) for d in self.token_dims]
            )

    def forward(self, segs):
        outs = []
        for t in range(8):
            x = segs[t]
            if self.mode == "none":
                y = x
            elif self.mode == "per_token":
                y = x @ self.pw[t].t() + self.pb[t]
            else:
                y = (x @ self.shared_s.t())[:, : self.token_dims[t]] * self.tok_s[t]
                y = y + self.tok_b[t]
            if self.pool_clip:
                y = torch.clamp(y, 0.0, 1.0)
            outs.append(y)
        return outs

    def split(self, flat, widths):
        segs = []
        off = 0
        for w in widths:
            segs.append(flat[:, off:off + w])
            off += w
        return segs


class ChannelGate(nn.Module):
    def __init__(self, token_dims, enabled=False):
        super().__init__()
        self.token_dims = check_dims(token_dims)
        self.enabled = enabled
        total = sum(self.token_dims)
        if enabled:
            self.ga = nn.Parameter(torch.zeros(total))
            self.gb = nn.Parameter(torch.ones(total))
        else:
            self.register_parameter("ga", None)
            self.register_parameter("gb", None)

    def forward(self, flat):
        if not self.enabled:
            return flat
        return flat * torch.clamp(self.ga * flat + self.gb, 0.0, 1.0)


class DenseHead(nn.Module):
    def __init__(self, total_dim, h1=128, h2=32):
        super().__init__()
        self.h1 = h1
        self.h2 = h2
        self.fc1 = nn.Linear(total_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fcv = nn.Linear(h2, 1)
        self.fcwdl = nn.Linear(h2, 3)

    def forward(self, flat):
        h1 = torch.clamp(self.fc1(flat), 0.0, 1.0)
        h2 = torch.clamp(self.fc2(h1), 0.0, 1.0)
        value = torch.tanh(self.fcv(h2)).squeeze(-1)
        return value, self.fcwdl(h2)


class DenseModel(nn.Module):
    def __init__(self, variant="A", token_dims=None, pooling="none", pool_clip=True,
                 gate_on=False, shared_width=SHARED_WIDTH, head_h1=128, head_h2=32):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.token_dims = check_dims(token_dims or [32] * 8)
        self.pooling = pooling
        self.pool_clip = pool_clip
        self.gate_on = gate_on
        self.shared_width = shared_width
        self.head_h1 = head_h1
        self.head_h2 = head_h2
        if pooling == "shared":
            group_widths = [shared_width] * 8
        else:
            group_widths = list(self.token_dims)
        self.group_widths = group_widths
        self.embedder = VarEmbedder(group_widths)
        self.pool = TokenPool(self.token_dims, pooling, pool_clip, shared_width)
        self.gate = ChannelGate(self.token_dims, gate_on)
        self.head = DenseHead(sum(self.token_dims), head_h1, head_h2)

    def forward(self, group_ids, group_mask):
        acc = self.embedder(group_ids, group_mask)
        segs = self.pool.split(acc, self.group_widths)
        outs = self.pool(segs)
        flat = torch.cat(outs, dim=1)
        flat = self.gate(flat)
        return self.head(flat)

    def arch_tensors(self):
        d = {}
        if self.pooling == "per_token":
            for t in range(8):
                d[f"pool_w{t}"] = self.pool.pw[t].detach()
                d[f"pool_b{t}"] = self.pool.pb[t].detach()
        elif self.pooling == "shared":
            d["pool_S"] = self.pool.shared_s.detach()
            for t in range(8):
                d[f"pool_s{t}"] = self.pool.tok_s[t].detach()
                d[f"pool_b{t}"] = self.pool.tok_b[t].detach()
        if self.gate_on:
            d["gate_a"] = self.gate.ga.detach()
            d["gate_b"] = self.gate.gb.detach()
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
        order = []
        if self.pooling == "per_token":
            for t in range(8):
                order += [f"pool_w{t}", f"pool_b{t}"]
        elif self.pooling == "shared":
            order += ["pool_S"]
            for t in range(8):
                order += [f"pool_s{t}", f"pool_b{t}"]
        if self.gate_on:
            order += ["gate_a", "gate_b"]
        return order + ["w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]

    def embedding_tensors(self):
        return {f"emb{g}": self.embedder.tables[g].weight.detach() for g in range(8)}

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_bytes(self):
        return self.parameter_count() * 4

    def model_spec(self, quantization="fp32"):
        return {
            "arch": f"RUNE-03-{self.variant}",
            "arch_version": "0.3.0",
            "feature_set": "grouped_hkav2_fullthreats_v01",
            "tokens": 8,
            "token_dim": sum(self.token_dims),
            "token_dims": list(self.token_dims),
            "attention": "none",
            "geometric_bias": "none",
            "head": "value_wdl",
            "quantization": quantization,
            "variant": self.variant,
            "pooling": self.pooling,
            "pool_clip": self.pool_clip,
            "gate_on": self.gate_on,
            "shared_width": self.shared_width,
            "head_h1": self.head_h1,
            "head_h2": self.head_h2,
        }


def dense_presets():
    return {
        "A": {"variant": "A", "token_dims": [32] * 8, "pooling": "none", "gate_on": False},
        "B": {"variant": "B", "token_dims": [32] * 8, "pooling": "per_token", "gate_on": False},
        "C": {"variant": "C", "token_dims": [32] * 8, "pooling": "per_token", "gate_on": True},
        "D": {"variant": "D", "token_dims": [32] * 8, "pooling": "shared", "gate_on": False},
    }


ALLOCATION_H1 = [28, 32, 32, 28, 20, 48, 40, 28]


BUDGET_ALLOCS = {
    "uniform_128": [16] * 8,
    "uniform_192": [24] * 8,
    "uniform_256": [32] * 8,
    "uniform_320": [40] * 8,
    "h1_256": list(ALLOCATION_H1),
}


def allocation_for(name):
    if name not in BUDGET_ALLOCS:
        raise ValueError(f"unknown allocation {name}")
    return list(BUDGET_ALLOCS[name])


def build_dense_model(variant="A", token_dims=None, pooling=None, pool_clip=True,
                      gate_on=None, shared_width=SHARED_WIDTH, head_h1=None,
                      head_h2=None):
    base = dict(dense_presets()[variant])
    if token_dims is not None:
        base["token_dims"] = token_dims
    if pooling is not None:
        base["pooling"] = pooling
    if gate_on is not None:
        base["gate_on"] = gate_on
    if head_h1 is not None:
        base["head_h1"] = head_h1
    if head_h2 is not None:
        base["head_h2"] = head_h2
    return DenseModel(shared_width=shared_width, **base)
