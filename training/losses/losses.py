import torch


def ranking_accuracy(value_pred_a, value_pred_b, sign):
    pred_sign = torch.sign(value_pred_a - value_pred_b)
    return ((pred_sign == sign.float()).float()).mean()


def wdl_accuracy(wdl_logits, wdl_tgt):
    return (wdl_logits.argmax(dim=-1) == wdl_tgt).float().mean()
