import torch

from training.datasets.rune_dataset import RuneDataset
from training.features.context import context_vector


class FlexDataset(RuneDataset):
    def __init__(self, records, max_per_group=None):
        super().__init__(records, max_per_group=max_per_group)
        self.contexts = [context_vector(r["fen"]) for r in records]

    def collate(self, idxs):
        ids, masks, value, wdl = super().collate(idxs)
        ctx = torch.tensor([self.contexts[i] for i in idxs], dtype=torch.float32)
        return ids, masks, ctx, value, wdl


def make_flex_loader(records, batch_size=256, shuffle=True, seed=0, max_per_group=None):
    ds = FlexDataset(records, max_per_group=max_per_group)
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
