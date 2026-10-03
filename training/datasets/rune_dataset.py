import torch
from torch.utils.data import Dataset

from training.features.python_features import NUM_GROUPS, VOCAB_SIZES, extract_features


class RuneDataset(Dataset):
    def __init__(self, records, max_per_group=None):
        self.records = records
        self.feats = [extract_features(r["fen"]) for r in records]
        if max_per_group is None:
            max_per_group = [0] * NUM_GROUPS
            for fl in self.feats:
                counts = [0] * NUM_GROUPS
                for g, _ in fl:
                    counts[g] += 1
                for g in range(NUM_GROUPS):
                    max_per_group[g] = max(max_per_group[g], counts[g])
            max_per_group = [max(1, m) for m in max_per_group]
        self.max_per_group = max_per_group

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return i

    def collate(self, idxs):
        b = len(idxs)
        ids = []
        masks = []
        for g in range(NUM_GROUPS):
            m = self.max_per_group[g]
            gid = torch.zeros(b, m, dtype=torch.long)
            gm = torch.zeros(b, m, dtype=torch.float32)
            for bi, i in enumerate(idxs):
                vals = [idx for gg, idx in self.feats[i] if gg == g][:m]
                for j, v in enumerate(vals):
                    gid[bi, j] = v % VOCAB_SIZES[g]
                    gm[bi, j] = 1.0
            ids.append(gid)
            masks.append(gm)
        value = torch.tensor([self.records[i]["value"] for i in idxs], dtype=torch.float32)
        wdl = torch.tensor([self.records[i]["wdl"] for i in idxs], dtype=torch.long)
        if "teacher_v" in self.records[0]:
            tv = torch.tensor([r.get("teacher_v", 0.0) for r in
                               [self.records[i] for i in idxs]], dtype=torch.float32)
            tw = torch.tensor([r.get("teacher_w", 1) for r in
                               [self.records[i] for i in idxs]], dtype=torch.long)
            tu = torch.tensor([r.get("teacher_u", 0.5) for r in
                               [self.records[i] for i in idxs]], dtype=torch.float32)
            return ids, masks, None, value, wdl, {"v": tv, "w": tw, "u": tu}
        return ids, masks, value, wdl


def make_loader(records, batch_size=256, shuffle=True, seed=0, max_per_group=None):
    ds = RuneDataset(records, max_per_group=max_per_group)
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
