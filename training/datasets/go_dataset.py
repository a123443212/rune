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

import random

import torch
from torch.utils.data import Dataset


def record_state(r):
    return r.get("state")


class GoPlanesDataset(Dataset):
    def __init__(self, records, game):
        self.records = list(records)
        self.game = game
        self.n = int(game.size)
        self.policy_size = self.n * self.n + 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return i

    def planes_of(self, i):
        import numpy as np
        p = self.game.planes(record_state(self.records[i]))
        return np.asarray(p, dtype="float32")

    def collate(self, idxs):
        import numpy as np
        b = len(idxs)
        n = self.n
        c = 8
        planes = torch.zeros(b, c, n, n)
        value = torch.zeros(b)
        wdl = torch.zeros(b, 3)
        policy = torch.zeros(b, self.policy_size)
        for bi, i in enumerate(idxs):
            r = self.records[i]
            planes[bi] = torch.tensor(self.planes_of(i))
            value[bi] = float(r.get("value", 0.0))
            w = r.get("wdl", [0.0, 1.0, 0.0])
            wdl[bi, 0] = float(w[0])
            wdl[bi, 1] = float(w[1])
            wdl[bi, 2] = float(w[2])
            mv = r.get("move", None)
            if mv is None:
                legal = self.game.legal(record_state(r))
                k = float(len(legal))
                for m in legal:
                    if int(m) == -1:
                        policy[bi, n * n] = 1.0 / k
                    else:
                        policy[bi, int(m)] = 1.0 / k
            else:
                if int(mv) == -1:
                    policy[bi, n * n] = 1.0
                else:
                    policy[bi, int(mv)] = 1.0
        return planes, value, wdl, policy


def make_go_loader(records, game, batch_size=8, shuffle=False, seed=0):
    from torch.utils.data import DataLoader
    ds = GoPlanesDataset(records, game)
    rng = random.Random(seed)
    if shuffle:
        order = list(range(len(ds)))
        rng.shuffle(order)
        ds.records = [ds.records[i] for i in order]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=ds.collate)
    return loader, ds
