import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from training.models.adaptive import (
    ARCH_ID,
    ARCH_VERSION,
    build_adaptive_model,
    route_mask,
    wdl_entropy,
)


def tiny_batch(n=4, dim=32):
    ids = [torch.randint(0, 64, (n, 4)) for _ in range(9)]
    masks = [torch.ones(n, 4) for _ in range(9)]
    return ids, masks


def test_forward_shapes():
    m = build_adaptive_model(dim=16)
    m.eval()
    ids, masks = tiny_batch()
    with torch.no_grad():
        cv, cw, rv, rw, diff = m(ids, masks)
    assert cv.shape == (4,) and rv.shape == (4,)
    assert cw.shape == (4, 3) and rw.shape == (4, 3)
    assert diff.shape == (4,)


def test_infer_modes_match_paths():
    torch.manual_seed(0)
    m = build_adaptive_model(dim=16)
    m.eval()
    ids, masks = tiny_batch()
    with torch.no_grad():
        cv, cw, rv, rw, diff = m(ids, masks)
        ov, ow, om = m.infer(ids, masks, mode="cheap")
        av, aw, am = m.infer(ids, masks, mode="always")
    assert torch.equal(ov, cv) and torch.equal(ow, cw)
    assert not om.any()
    assert torch.equal(av, rv) and torch.equal(aw, rw)
    assert am.all()


def test_adaptive_extremes_and_determinism():
    m = build_adaptive_model(dim=16)
    m.eval()
    ids, masks = tiny_batch()
    with torch.no_grad():
        _, _, mask_all = m.infer(ids, masks, mode="adaptive", threshold=-1e9)
        _, _, mask_none = m.infer(ids, masks, mode="adaptive", threshold=1e9)
        _, _, mask_a = m.infer(ids, masks, mode="adaptive", threshold=0.0)
        _, _, mask_b = m.infer(ids, masks, mode="adaptive", threshold=0.0)
        full_v, full_w, _ = m.infer(ids, masks, mode="always")
        mix_v, mix_w, _ = m.infer(ids, masks, mode="adaptive", threshold=-1e9)
    assert mask_all.all() and not mask_none.any()
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(mix_v, full_v) and torch.equal(mix_w, full_w)


def test_hysteresis_logic():
    d = torch.tensor([0.2, 0.5, 0.8])
    prev = torch.tensor([True, True, False])
    got = route_mask(d, 0.6, 0.4, prev)
    assert got.tolist() == [False, True, True]


def test_wdl_entropy_uniform():
    e = wdl_entropy(torch.zeros(2, 3))
    assert abs(e.mean().item() - math.log(3)) < 1e-5


def test_spec_and_budgets():
    m = build_adaptive_model(dim=16, pruned_pairs=[(0, 1)])
    s = m.model_spec()
    assert s["arch"] == ARCH_ID and s["arch_version"] == ARCH_VERSION
    assert s["t_low"] is None and s["threshold"] == 0.5
    assert [0, 1] in s["pruned_pairs"]
    assert m.cheap_parameter_count() < m.parameter_count()
    assert m.model_size_bytes() == m.parameter_count() * 4
    assert m.refine.prune_mask[0, 1].item() == 0.0
    assert m.refine.prune_mask[1, 0].item() == 1.0


def test_export_header_roundtrip(tmp_path):
    import numpy as np

    from training.export.export import export_model, fnv1a, load_exported_arrays, read_header

    torch.manual_seed(0)
    m = build_adaptive_model(dim=16, threshold=0.25, pruned_pairs=[(0, 4)])
    m.eval()
    path = str(tmp_path / "a04.rune")
    header = export_model(m, path, quantization="fp32")
    assert header["arch"] == "RUNE-04"
    assert header["arch_version"] == "0.4.0"
    assert header["threshold"] == 0.25
    assert header["t_low"] is None
    assert header["token_dims"] == [16] * 8
    assert len(header["checksum"]) == 16
    with open(path, "rb") as f:
        import struct

        assert f.read(4) == b"RUNE"
        (n,) = struct.unpack("<I", f.read(4))
        f.read(n)
        payload = f.read()
    assert format(fnv1a(payload), "016x") == header["checksum"]
    assert read_header(path)["arch"] == "RUNE-04"
    _, arrays = load_exported_arrays(path)
    assert [t for t in m.export_order() if not t.startswith("emb")] == \
        [k for k in arrays if not k.startswith("emb")]
