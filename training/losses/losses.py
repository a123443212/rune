# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

import torch


def ranking_accuracy(value_pred_a, value_pred_b, sign):
    pred_sign = torch.sign(value_pred_a - value_pred_b)
    return ((pred_sign == sign.float()).float()).mean()


def wdl_accuracy(wdl_logits, wdl_tgt):
    return (wdl_logits.argmax(dim=-1) == wdl_tgt).float().mean()
