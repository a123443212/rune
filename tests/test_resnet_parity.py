import numpy as np


def test_resnet_forward_shapes():
    from training.models.resnet import build_resnet
    m = build_resnet(board=9, channels=8, blocks=2)
    planes = np.zeros((1, 1, 9, 9), dtype=np.float32)
    planes[0, 0, 4, 4] = 1.0
    import torch
    with torch.no_grad():
        value, wdl, logits, probs = m(torch.tensor(planes))
    assert value.shape == (1,)
    assert wdl.shape == (1, 3)
    assert logits.shape == (1, 82)
    assert probs.shape == (1, 82)
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_resnet_export_roundtrip(tmp_path):
    from training.export.export import load_exported_arrays, verify_file_hashes
    from training.models.resnet import build_resnet
    import torch
    m = build_resnet(board=9, channels=8, blocks=1)
    p = str(tmp_path / "resnet.rune")
    from training.export.export import export_model
    h = export_model(m, p, quantization="fp32")
    assert h["architecture_id"] == "RUNE-RESNET-01"
    assert h["game"] == "go"
    verify_file_hashes(p)
    header, arrays = load_exported_arrays(p)
    assert "stem_w" in arrays
    assert "wpol" in arrays


def test_resnet_reference_matches_torch():
    import torch
    from training.compiler.reference_resnet import forward_resnet
    from training.models.resnet import build_resnet
    m = build_resnet(board=9, channels=8, blocks=1)
    m.eval()
    planes = np.zeros((1, 1, 9, 9), dtype=np.float32)
    planes[0, 0, 0, 0] = 1.0
    planes[0, 0, 8, 8] = -1.0
    with torch.no_grad():
        tv, twdl, tlog, tprobs = m(torch.tensor(planes))
    arrays = {k: v.numpy() for k, v in m.arch_tensors().items()}
    r = forward_resnet(arrays, 9, 8, 1, 82, planes)
    assert abs(r["value"] - float(tv[0])) < 1e-4
    assert np.abs(r["wdl"] - twdl.numpy()[0]).max() < 1e-4
    assert np.abs(r["policy"] - tprobs.numpy()[0]).max() < 1e-4
