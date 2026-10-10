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

from training.datasets.rune_dataset import RuneDataset, record_state


class FlexDataset(RuneDataset):
    def __init__(self, records, max_per_group=None, game=None):
        super().__init__(records, max_per_group=max_per_group, game=game)
        self.contexts = [self.game.context(record_state(r)) for r in records]

    def collate(self, idxs):
        out = super().collate(idxs)
        if len(out) == 6:
            ids, masks, _, value, wdl, teach = out
        else:
            ids, masks, value, wdl = out
            teach = None
        ctx = torch.tensor([self.contexts[i] for i in idxs], dtype=torch.float32)
        if teach is None:
            return ids, masks, ctx, value, wdl
        return ids, masks, ctx, value, wdl, teach


def make_flex_loader(records, batch_size=256, shuffle=True, seed=0, max_per_group=None, game=None):
    ds = FlexDataset(records, max_per_group=max_per_group, game=game)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ds.collate,
        generator=g if shuffle else None,
    )
    return loader, ds
