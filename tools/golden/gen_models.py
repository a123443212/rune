import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from training.models.rune_models import build_model
from training.export.export import export_model, read_header
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "spec", "test-vectors", "models")
def build():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(7)
    m = build_model("RUNE-MLP")
    export_model(m, os.path.join(OUT, "tiny-mlp-fp32.rune"), quantization="fp32")
    torch.manual_seed(11)
    m2 = build_model("RUNE-ATTN-GAB")
    export_model(m2, os.path.join(OUT, "small-gab-fp32.rune"), quantization="fp32")
    export_model(m2, os.path.join(OUT, "small-gab-int8.rune"), quantization="int8")
    export_model(m2, os.path.join(OUT, "small-gab-int16.rune"), quantization="int16")
    try:
        from training.models.relational import RuneRelational
        torch.manual_seed(13)
        mr = RuneRelational(tokens=8, dim=32)
        export_model(mr, os.path.join(OUT, "rel-08x32-fp32.rune"), quantization="fp32")
    except Exception as e:
        print("relational skip " + str(e))
    try:
        from training.models.dense import DenseModel
        import torch as _t
        _t.manual_seed(17)
        md = DenseModel(variant="B", token_dims=[16, 16, 16, 16, 16, 16, 16, 16])
        export_model(md, os.path.join(OUT, "dense-b-fp32.rune"), quantization="fp32")
        export_model(md, os.path.join(OUT, "dense-b-int8.rune"), quantization="int8")
        export_model(md, os.path.join(OUT, "dense-b-int16.rune"), quantization="int16")
    except Exception as e:
        print("dense skip " + str(e))
    try:
        from training.models.adaptive import AdaptiveModel
        torch.manual_seed(19)
        ma = AdaptiveModel(dim=16)
        export_model(ma, os.path.join(OUT, "adaptive-fp32.rune"), quantization="fp32")
    except Exception as e:
        print("adaptive skip " + str(e))
    for name in sorted(os.listdir(OUT)):
        h = read_header(os.path.join(OUT, name))
        arch = h.get("architecture_id", h.get("arch"))
        fmt = h.get("format")
        hh = h.get("model_hash", h.get("checksum", ""))
        print(name + " arch=" + str(arch) + " fmt=" + str(fmt) + " hash=" + str(hh)[:8])
if __name__ == "__main__":
    build()
