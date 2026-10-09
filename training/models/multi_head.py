import torch
import torch.nn as nn

from training.features.python_features import TOKENS, TOKEN_DIM

NUM_HEADS = 4
HEAD_DIM = 8

assert NUM_HEADS * HEAD_DIM == TOKEN_DIM


def clip01(x):
    return torch.clamp(x, 0.0, 1.0)


class RuneMultiHeadMixer(nn.Module):
    def __init__(self, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super().__init__()
        if num_heads * head_dim != TOKEN_DIM:
            raise ValueError("heads * head_dim must equal token dim")
        self.num_heads = num_heads
        self.head_dim = head_dim
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
            a = clip01(s)
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
