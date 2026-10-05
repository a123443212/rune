import torch


def quantize_delta_int8(delta, scale):
    q = torch.round(delta / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def apply_quant_delta(state_int32, qdelta):
    return state_int32 + qdelta.to(torch.int32)


def dequant_tokens(acc_int32, scales):
    out = acc_int32.float() * scales.unsqueeze(-1)
    return torch.clamp(out, 0.0, 1.0)
