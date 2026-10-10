import json
import os


def golden_path():
    return os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "resnet", "eval_v02.json")


def fixture_path():
    return os.path.join(os.path.dirname(__file__), "..", "spec", "test-vectors", "models", "resnet9-8plane-fp32.rune")


def test_golden_file_shape():
    with open(golden_path()) as f:
        d = json.load(f)
    assert d["arch"] == "RUNE-RESNET-01"
    assert d["feature_version"] == "go_planes_v02"
    assert d["in_planes"] == 8
    assert len(d["vectors"]) == 4


def test_export_carries_in_planes(tmp_path):
    from training.export.export import export_model, load_exported_arrays
    from training.models.resnet import build_resnet
    import torch
    torch.manual_seed(7)
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    p = str(tmp_path / "r8.rune")
    h = export_model(m, p, quantization="fp32")
    assert h["in_planes"] == 8
    assert h["feature_set"] == "go_planes_v02"
    header, arrays = load_exported_arrays(p)
    assert header["in_planes"] == 8
    assert list(arrays["stem_w"].shape) == [8, 8, 3, 3]


def test_export_v01_default_in_planes(tmp_path):
    from training.export.export import export_model
    from training.models.resnet import build_resnet
    import torch
    torch.manual_seed(7)
    m = build_resnet(board=9, channels=8, blocks=1)
    p = str(tmp_path / "r1.rune")
    h = export_model(m, p, quantization="fp32")
    assert h["in_planes"] == 1
    assert h["feature_set"] == "go_planes_v01"


def test_ir_carries_in_planes():
    from training.compiler.ir_resnet import build_resnet as build_ir
    from training.models.resnet import build_resnet
    import torch
    torch.manual_seed(7)
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    spec = m.model_spec()
    metas = [{"name": k, "shape": list(v.shape), "dtype": "float32"} for k, v in m.arch_tensors().items()]
    ir = build_ir(spec, metas, {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1})
    assert ir["model"]["in_planes"] == 8
    planes = [t for t in ir["tensors"] if t["name"] == "planes"][0]
    assert planes["shape"] == [8 * 81]
    stem = [o for o in ir["ops"] if o["id"] == "op01"][0]
    assert stem["attrs"]["in_planes"] == 8


def test_golden_values_match_torch():
    import torch
    from training.games.go_v02 import GoGameV02
    from training.models.resnet import build_resnet
    from training.export.export import load_exported_arrays
    with open(golden_path()) as f:
        d = json.load(f)
    header, arrays = load_exported_arrays(fixture_path())
    import torch as t
    t.manual_seed(23)
    m = build_resnet(board=9, channels=8, blocks=2, in_planes=8, feature_set="go_planes_v02")
    state_dict = m.state_dict()
    import numpy as np
    name_map = {"stem_w": "stem.weight", "stem_b": "stem.bias", "vh1": "value.fc1.weight", "bh1": "value.fc1.bias", "wv": "value.fcv.weight", "bv": "value.fcv.bias", "wwdl": "value.fcwdl.weight", "bwdl": "value.fcwdl.bias", "wpol": "policy.fc.weight", "bpol": "policy.fc.bias"}
    for i in range(2):
        name_map[f"b{i}_w1"] = f"tower.{i}.c1.weight"
        name_map[f"b{i}_b1"] = f"tower.{i}.c1.bias"
        name_map[f"b{i}_w2"] = f"tower.{i}.c2.weight"
        name_map[f"b{i}_b2"] = f"tower.{i}.c2.bias"
    for k, v in arrays.items():
        if k in name_map:
            target = state_dict[name_map[k]]
            state_dict[name_map[k]] = torch.tensor(v).reshape(target.shape)
    m.load_state_dict(state_dict)
    m.eval()
    g = GoGameV02(size=9)
    for vec in d["vectors"]:
        planes = g.planes(vec["state"])
        assert list(planes.shape) == vec["planes_shape"]
        assert vec["legal_count"] == len(g.legal(vec["state"]))
        with torch.no_grad():
            v, w, _, _ = m(torch.tensor(planes))
        assert abs(float(v[0]) - vec["value"]) < 1e-5
