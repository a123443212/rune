import torch
import torch.nn as nn

from training.features.python_features import TOKENS, TOKEN_DIM

NUM_HEADS = 4
HEAD_DIM = 8

assert NUM_HEADS * HEAD_DIM == TOKEN_DIM


def clip01(x):
    return torch.clamp(x, 0.0, 1.0)


def apply_gate(name, s):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * s + 0.5, 0.0, 1.0)
    if name == "screlu":
        c = torch.clamp(s, 0.0, 1.0)
        return c * c
    return torch.clamp(s, 0.0, 1.0)


class RuneMultiHeadMixer(nn.Module):
    def __init__(self, num_heads=NUM_HEADS, head_dim=HEAD_DIM, gate="clip"):
        super().__init__()
        if num_heads * head_dim != TOKEN_DIM:
            raise ValueError("heads * head_dim must equal token dim")
        if gate not in ("clip", "hard_sigmoid", "screlu"):
            raise ValueError(f"unknown gate {gate}")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate = gate
        self.q = nn.ModuleList([nn.Linear(TOKEN_DIM, head_dim) for _ in range(num_heads)])
        self.k = nn.ModuleList([nn.Linear(TOKEN_DIM, head_dim) for _ in range(num_heads)])
        self.v = nn.ModuleList([nn.Linear(TOKEN_DIM, head_dim) for _ in range(num_heads)])
        self.gab = nn.ParameterList([nn.Parameter(torch.zeros(TOKENS, TOKENS))
                                     for _ in range(num_heads)])
        self.wo = nn.Linear(TOKEN_DIM, TOKEN_DIM)

    def forward(self, x):
        parts = []
        for h in range(self.num_heads):
            q = self.q[h](x)
            k = self.k[h](x)
            v = self.v[h](x)
            s = q @ k.transpose(-1, -2) + self.gab[h]
            a = apply_gate(self.gate, s)
            parts.append(a @ v)
        y = torch.cat(parts, dim=-1)
        return x + self.wo(y)

    def arch_tensors(self):
        d = {}
        for h in range(self.num_heads):
            s = f"_h{h}"
            d["wq" + s] = self.q[h].weight.detach()
            d["bq" + s] = self.q[h].bias.detach()
            d["wk" + s] = self.k[h].weight.detach()
            d["bk" + s] = self.k[h].bias.detach()
            d["wv" + s] = self.v[h].weight.detach()
            d["bv" + s] = self.v[h].bias.detach()
            d["gab" + s] = self.gab[h].detach()
        d["wo"] = self.wo.weight.detach()
        d["bwo"] = self.wo.bias.detach()
        return d

    @staticmethod
    def export_order():
        order = []
        for h in range(NUM_HEADS):
            s = f"_h{h}"
            order += ["wq" + s, "bq" + s, "wk" + s, "bk" + s, "wv" + s, "bv" + s, "gab" + s]
        return order + ["wo", "bwo"]
