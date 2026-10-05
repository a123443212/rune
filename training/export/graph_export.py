import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.compiler.graph import export_canonical_graph
from training.compiler.ir import build_ir, verify_ir
from training.compiler.memory import plan_memory
from training.compiler.plan import select_kernels


def export_graph_for_model(model, target, path):
    spec = model.model_spec(quantization="fp32")
    order = ["emb" + str(g) for g in range(8)] + model.export_order()
    arch_tensors = model.arch_tensors()
    metas = []
    for name in order:
        if name.startswith("emb"):
            t = model.embedding_tensors()[name]
            metas.append({"name": name, "shape": list(t.shape), "dtype": "float32"})
        else:
            import numpy as np
            arr = arch_tensors[name]
            try:
                sh = list(arr.shape)
            except Exception:
                sh = [int(arr.numel())]
            metas.append({"name": name, "shape": sh, "dtype": "float32"})
    norm = {"arch": spec.get("arch", ""), "arch_version": spec.get("arch_version", "0.2.0"), "tokens": spec.get("tokens", 8), "token_dim": spec.get("token_dim", 32), "quantization": "fp32", "gate": spec.get("gate", "clip"), "alpha": spec.get("alpha", 1.0), "head_h1": spec.get("head_h1", 128), "head_h2": spec.get("head_h2", 32)}
    ir = build_ir(norm, metas, target)
    errs = verify_ir(ir)
    if errs:
        raise ValueError("; ".join(errs))
    plan, fusion = select_kernels(ir)
    ir["kernel_plan"] = plan
    ir["fusion"] = fusion
    ir["memory"] = plan_memory(ir)
    export_canonical_graph(ir, path)
    return ir
