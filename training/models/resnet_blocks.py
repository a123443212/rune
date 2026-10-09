import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = self.c2(h)
        return F.relu(x + h)


class ValueHead(nn.Module):
    def __init__(self, channels, h2=256):
        super().__init__()
        self.fc1 = nn.Linear(channels, h2)
        self.fcv = nn.Linear(h2, 1)
        self.fcwdl = nn.Linear(h2, 3)

    def forward(self, pooled):
        h = torch.clamp(self.fc1(pooled), 0.0, 1.0)
        value = torch.tanh(self.fcv(h)).squeeze(-1)
        wdl = self.fcwdl(h)
        return value, wdl


class PolicyHead(nn.Module):
    def __init__(self, channels, board, policy_size):
        super().__init__()
        self.fc = nn.Linear(channels * board * board, policy_size)

    def forward(self, flat):
        logits = self.fc(flat)
        return logits, F.softmax(logits, dim=-1)
