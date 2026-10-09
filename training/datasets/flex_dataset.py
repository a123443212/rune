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
