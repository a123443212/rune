import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.conftest import find_binding

_d = find_binding()
if _d and _d not in sys.path:
    sys.path.insert(0, _d)

import pytest

try:
    import rune_bindings as rb

    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

from training.export.export import export_model, fnv1a, load_exported_arrays
from training.features.python_features import extract_features
from training.models.dense import build_dense_model

needs_bindings = pytest.mark.skipif(not HAS_BINDINGS, reason="bindings not built")

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]

DIMS = [24, 40, 32, 28, 20, 44, 36, 12]
DIMS_D = [24, 28, 32, 20, 16, 30, 26, 12]

VARIANTS = [
    ("A", DIMS, "none", False),
    ("B", DIMS, "per_token", False),
    ("C", DIMS, "per_token", True),
    ("D", DIMS_D, "shared", False),
]


def copy_to_cpp(tm, cm):
    emb = tm.embedding_tensors()
    for g in range(8):
        cm.set_embedding(g, [float(x) for x in emb[f"emb{g}"].numpy().reshape(-1)])
    arch = tm.arch_tensors()
    names = tm.export_order()
    flat = []
    for n in names:
        flat.extend([float(x) for x in np.asarray(arch[n]).reshape(-1)])
    cm.set_arch_tensors(names, flat)


def batch_for(tm, fen):
    from training.features.python_features import VOCAB_SIZES

    feats = extract_features(fen)
    ids, masks = [], []
    for g in range(8):
        idx = [i for gg, i in feats if gg == g]
        ids.append(torch.tensor([idx if idx else [0]]))
        masks.append(torch.tensor([[1.0] * len(idx) if idx else [0.0]]))
    return ids, masks


def test_fnv1a_deterministic():
    assert fnv1a(b"RUNE") == fnv1a(bytearray(b"RUNE"))
    assert fnv1a(b"") != fnv1a(b"\x00")
    assert len(format(fnv1a(b"abc"), "016x")) == 16


@needs_bindings
def test_dense_parity_all_variants():
    torch.manual_seed(0)
    for variant, dims, pooling, gate_on in VARIANTS:
        tm = build_dense_model(variant=variant, token_dims=dims, pooling=pooling,
                               gate_on=gate_on)
        for p in tm.parameters():
            torch.nn.init.uniform_(p, -0.05, 0.05)
        tm.eval()
        cm = rb.DenseModel(variant, dims, pooling, True, gate_on)
        copy_to_cpp(tm, cm)
        for fen in FENS:
            cv, cw = cm.eval_fen(fen)
            ids, masks = batch_for(tm, fen)
            with torch.no_grad():
                pv, pw = tm(ids, masks)
            assert abs(cv - pv.item()) < 1e-4, (variant, fen)
            assert np.allclose(np.array(cw), pw[0].numpy(), atol=1e-4)


@needs_bindings
def test_dense_tokens_parity():
    torch.manual_seed(1)
    tm = build_dense_model(variant="C", token_dims=DIMS, pooling="per_token", gate_on=True)
    tm.eval()
    cm = rb.DenseModel("C", DIMS, "per_token", True, True)
    copy_to_cpp(tm, cm)
    for fen in FENS:
        cpp_tok = np.array(cm.tokens_for_fen(fen))
        ids, masks = batch_for(tm, fen)
        with torch.no_grad():
            acc = tm.embedder(ids, masks)
            segs = tm.pool.split(acc, tm.group_widths)
            outs = tm.pool(segs)
            flat = torch.cat(outs, dim=1)
            py_tok = tm.gate(flat)[0].numpy()
        assert np.allclose(cpp_tok, py_tok, atol=1e-5), fen


def test_dense_export_checksum(tmp_path):
    torch.manual_seed(2)
    model = build_dense_model(variant="B", token_dims=DIMS, pooling="per_token")
    p = str(tmp_path / "d.rune")
    header = export_model(model, p, quantization="fp32")
    assert header["arch"] == "RUNE-03-B"
    assert header["token_dims"] == DIMS
    assert header["pooling"] == "per_token"
    assert len(header["checksum"]) == 16
    with open(p, "rb") as f:
        import struct

        assert f.read(4) == b"RUNE"
        (n,) = struct.unpack("<I", f.read(4))
        f.read(n)
        payload = f.read()
    assert format(fnv1a(payload), "016x") == header["checksum"]
    _, arrays = load_exported_arrays(p)
    assert "pool_w0" in arrays and "gate_a" not in arrays


@needs_bindings
def test_cpp_accepts_python_export_and_rejects_tamper(tmp_path):
    torch.manual_seed(3)
    model = build_dense_model(variant="C", token_dims=DIMS, pooling="per_token",
                              gate_on=True)
    p = str(tmp_path / "x.rune")
    export_model(model, p, quantization="int8")
    ok, msg = rb.verify_rune_file(p)
    assert ok, msg
    with open(p, "rb") as f:
        blob = bytearray(f.read())
    blob[-1] ^= 0xFF
    q = str(tmp_path / "x_bad.rune")
    with open(q, "wb") as f:
        f.write(blob)
    ok, msg = rb.verify_rune_file(q)
    assert not ok and "checksum" in msg
    h = str(tmp_path / "bogus.rune")
    with open(h, "wb") as f:
        import struct

        hb = b'{"format":1,"arch":"NOPE","arch_version":"9.9.9"}'
        f.write(b"RUNE" + struct.pack("<I", len(hb)) + hb)
    ok, msg = rb.verify_rune_file(h)
    assert not ok


def test_variant_specs():
    for v, dims, pooling, gate_on in VARIANTS:
        m = build_dense_model(variant=v, token_dims=dims, pooling=pooling, gate_on=gate_on)
        s = m.model_spec()
        assert s["arch"] == f"RUNE-03-{v}" and s["arch_version"] == "0.3.0"
        assert s["token_dims"] == dims and s["pooling"] == pooling
        assert s["gate_on"] == gate_on
