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

from training.features.python_features import VOCAB_SIZES

TOKEN_COUNTS = (6, 8, 10)
TOKEN_DIMS = (24, 32, 40)


def _full(group, vocabs):
    return [(group, 0, vocabs[group])]


def _split(group, lo, hi, vocabs):
    return [(group, min(lo, vocabs[group]), min(hi, vocabs[group]))]


def layout_for(tokens, vocabs=None):
    vs = list(vocabs) if vocabs is not None else list(VOCAB_SIZES)
    if tokens == 8:
        return [_full(0, vs) + _full(8, vs)] + [_full(g, vs) for g in range(1, 8)]
    if tokens == 6:
        return [_full(0, vs) + _full(8, vs), _full(1, vs), _full(2, vs), _full(3, vs) + _full(4, vs),
                _full(5, vs) + _full(6, vs), _full(7, vs)]
    if tokens == 10:
        return [_full(0, vs) + _full(8, vs), _full(1, vs), _split(2, 0, 128, vs), _split(2, 128, 256, vs),
                _full(3, vs), _full(4, vs), _split(5, 0, 384, vs), _split(5, 384, 512, vs),
                _full(6, vs), _full(7, vs)]
    raise ValueError(f"unsupported token count {tokens}")


def check_layout(tokens, dim, vocabs=None):
    if tokens not in TOKEN_COUNTS:
        raise ValueError(f"unsupported token count {tokens}")
    if dim not in TOKEN_DIMS:
        raise ValueError(f"unsupported token dim {dim}")
    return layout_for(tokens, vocabs)
