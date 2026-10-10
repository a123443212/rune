import torch


def clip01(x):
    return torch.clamp(x, 0.0, 1.0)


def apply_gate(name, scores):
    if name == "hard_sigmoid":
        return torch.clamp(0.2 * scores + 0.5, 0.0, 1.0)
    if name == "screlu":
        clipped = torch.clamp(scores, 0.0, 1.0)
        return clipped * clipped
    return clip01(scores)