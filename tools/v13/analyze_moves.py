import sys

import torch

sys.path.insert(0, ".")

from tests.test_v13_incremental import MOVE_PAIRS
from training.features.python_features import extract_features
from training.rune_v13 import build_dense_graph, detect_changed_groups, group_deltas_to_tokens
from training.features.python_features import NUM_GROUPS


def main():
    torch.manual_seed(3)
    tables = [torch.randn(512, 32) * 0.01 for _ in range(NUM_GROUPS)]
    g = build_dense_graph(8)
    print("move,changed_groups,changed_tokens,affected_cells,total_cells")
    for name, (fa, fb) in MOVE_PAIRS.items():
        a = extract_features(fa)
        b = extract_features(fb)
        changed, _, _ = detect_changed_groups(a, b)
        r = group_deltas_to_tokens(tables, a, b, 32)
        ct = r["changed_tokens"]
        cells = len(g.affected_edges(ct))
        print(f"{name},{len(changed)},{len(ct)},{cells},64")


if __name__ == "__main__":
    main()
