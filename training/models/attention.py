import torch
import torch.nn as nn

from training.features.python_features import TOKEN_DIM, TOKENS
from training.models.activations import apply_gate


class RuneAttention(nn.Module):
    def __init__(self, use_gab, gate="clip", tokens=None, token_dim=None):
        super().__init__()
        if gate not in ("clip", "hard_sigmoid", "screlu"):
            raise ValueError(f"unknown gate {gate}")
        t = tokens if tokens is not None else TOKENS
        d = token_dim if token_dim is not None else TOKEN_DIM
        self.use_gab = use_gab
        self.gate = gate
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.gab = nn.Parameter(torch.zeros(t, t))

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scores = q @ k.transpose(-1, -2)
        if self.use_gab:
            scores = scores + self.gab
        weights = apply_gate(self.gate, scores)
        return x + weights @ v