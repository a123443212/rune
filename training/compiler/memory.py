ALIGN = 32


def align_up(n, align=ALIGN):
    return ((n + align - 1) // align) * align


def tensor_elems(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def plan_memory(ir):
    model = ir.get("model", {})
    tokens = int(model.get("tokens", 8))
    dim = int(model.get("token_dim", 32))
    h1 = int(model.get("head_h1", 128))
    h2 = int(model.get("head_h2", 32))
    live = [
        ("tokens_buf", tokens * dim),
        ("q_buf", tokens * dim),
        ("k_buf", tokens * dim),
        ("v_buf", tokens * dim),
        ("scores_buf", tokens * tokens),
        ("gate_buf", tokens * tokens),
        ("mixed_raw_buf", tokens * dim),
        ("mixed_buf", tokens * dim),
        ("h1_buf", h1),
        ("h2_buf", h2),
        ("flat_buf", tokens * dim),
        ("tmp_row", max(dim, h1, h2)),
    ]
    offsets = {}
    reuse_groups = [
        ["q_buf", "h1_buf"],
        ["k_buf", "h2_buf"],
        ["scores_buf", "gate_buf"],
    ]
    off = 0
    placements = {}
    for name, elems in live:
        placed = None
        for grp in reuse_groups:
            if name in grp:
                for other in grp:
                    if other in placements and other != name:
                        cand = placements[other]
                        if cand["elems"] >= elems:
                            placed = cand["offset"]
                            break
                if placed is not None:
                    break
        if placed is None:
            start = align_up(off)
            placements[name] = {"offset": start, "elems": elems, "bytes": elems * 4}
            offsets[name] = start
            off = start + elems * 4
        else:
            placements[name] = {"offset": placed, "elems": elems, "bytes": elems * 4, "shared": True}
            offsets[name] = placed
    total = align_up(off)
    plan = {
        "arena_bytes": total,
        "alignment": ALIGN,
        "buffers": placements,
        "strategy": "reuse q/h1, k/h2, scores/gate; single bump arena, no per-eval alloc",
        "in_place": ["gate_in_scores", "mixed_raw_into_mixed_when_alpha_1"],
    }
    return plan
