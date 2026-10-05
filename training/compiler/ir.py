IR_VERSION = "1.0"
SPEC_VERSION = "RUNE-10"
COMPILER_VERSION = "0.11.0"

CANONICAL_OPS = [
    "FeatureUpdate",
    "AccumulatorUpdate",
    "Tokenize",
    "Q",
    "K",
    "V",
    "Score",
    "Bias",
    "Gate",
    "Mix",
    "Residual",
    "HeadH1",
    "HeadH2",
    "Value",
    "WDL",
]

ADAPTIVE_OPS = [
    "Route",
    "Cheap",
    "Refine",
]

VALID_ISA = ["portable", "avx2", "avx512"]
VALID_DTYPE = ["fp32", "int8", "int16"]
VALID_LAYOUT = ["row-major", "col-major", "packed", "interleaved"]


def ir_empty():
    return {
        "ir_version": IR_VERSION,
        "spec_version": SPEC_VERSION,
        "model": {},
        "tensors": [],
        "ops": [],
        "memory": {},
        "fusion": [],
        "kernel_plan": [],
        "target": {},
        "hashes": {},
    }


def _tensor_entry(name, shape, dtype, layout, constant, lifetime):
    return {
        "name": name,
        "shape": list(shape),
        "dtype": dtype,
        "layout": layout,
        "constant": bool(constant),
        "lifetime": list(lifetime),
    }


def _op_entry(op_id, kind, inputs, outputs, attrs):
    return {
        "id": op_id,
        "kind": kind,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "attrs": dict(attrs),
    }


def build_ir(model_spec, tensor_metas, target):
    ir = ir_empty()
    arch = model_spec.get("arch", "")
    tokens = int(model_spec.get("tokens", 8))
    dim = int(model_spec.get("token_dim", 32))
    quant = model_spec.get("quantization", "fp32")
    gate = model_spec.get("gate", "clip")
    alpha = float(model_spec.get("alpha", 1.0))
    ir["model"] = {
        "architecture": arch,
        "architecture_version": model_spec.get("arch_version", "0.2.0"),
        "tokens": tokens,
        "token_dim": dim,
        "dtype": "fp32" if quant == "fp32" else quant,
        "quantization": quant,
        "gate": gate,
        "alpha": alpha,
        "head_h1": int(model_spec.get("head_h1", 128)),
        "head_h2": int(model_spec.get("head_h2", 32)),
        "threshold": float(model_spec.get("threshold", 0.5)),
        "t_high": float(model_spec.get("t_high", model_spec.get("threshold", 0.5))),
        "t_low": model_spec.get("t_low", None),
        "has_t_low": bool("t_low" in model_spec and model_spec["t_low"] is not None),
    }
    if arch in ("RUNE-04", "RUNE-05"):
        ir["model"]["adaptive"] = True
    else:
        ir["model"]["adaptive"] = False
    lifetimes = {
        "FeatureUpdate": [0, 1],
        "AccumulatorUpdate": [0, 2],
        "Tokenize": [1, 3],
        "Q": [3, 6],
        "K": [3, 6],
        "V": [3, 7],
        "Score": [6, 7],
        "Bias": [6, 7],
        "Gate": [6, 8],
        "Mix": [7, 8],
        "Residual": [3, 9],
        "HeadH1": [9, 10],
        "HeadH2": [10, 11],
        "Value": [11, 12],
        "WDL": [11, 12],
    }
    tensors = []
    tensors.append(_tensor_entry("features", [tokens], "index", "packed", False, lifetimes["FeatureUpdate"]))
    tensors.append(_tensor_entry("accumulator", [tokens, dim], "acc", "row-major", False, lifetimes["AccumulatorUpdate"]))
    tensors.append(_tensor_entry("tokens", [tokens, dim], "fp32", "row-major", False, lifetimes["Tokenize"]))
    for nm in ("wq", "bq", "wk", "bk", "wvv", "bvv", "wv", "bv", "gabS", "gab"):
        for tm in tensor_metas:
            if tm.get("name") == nm:
                tensors.append(_tensor_entry(nm, tm.get("shape", []), tm.get("dtype", "float32"), "row-major", True, [0, 12]))
    for nm in ("w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"):
        for tm in tensor_metas:
            if tm.get("name") == nm:
                tensors.append(_tensor_entry(nm, tm.get("shape", []), tm.get("dtype", "float32"), "row-major", True, [0, 12]))
    for g in range(8):
        nm = "emb%d" % g
        for tm in tensor_metas:
            if tm.get("name") == nm:
                tensors.append(_tensor_entry(nm, tm.get("shape", []), tm.get("dtype", "float32"), "row-major", True, [0, 12]))
    ir["tensors"] = tensors
    ops = []
    ops.append(_op_entry("op00", "FeatureUpdate", [], ["features"], {"tokens": tokens}))
    ops.append(_op_entry("op01", "AccumulatorUpdate", ["features"], ["accumulator"], {"tokens": tokens, "dim": dim, "quant": quant}))
    ops.append(_op_entry("op02", "Tokenize", ["accumulator"], ["tokens"], {"tokens": tokens, "dim": dim, "gate": "clip"}))
    ops.append(_op_entry("op03", "Q", ["tokens", "wq", "bq"], ["Q"], {"tokens": tokens, "dim": dim}))
    ops.append(_op_entry("op04", "K", ["tokens", "wk", "bk"], ["K"], {"tokens": tokens, "dim": dim}))
    ops.append(_op_entry("op05", "V", ["tokens", "wvv", "bvv"], ["V"], {"tokens": tokens, "dim": dim}))
    ops.append(_op_entry("op06", "Score", ["Q", "K"], ["scores"], {"tokens": tokens, "dim": dim}))
    ops.append(_op_entry("op07", "Bias", ["scores", "gabS"], ["biased"], {"tokens": tokens, "clamp": [-0.25, 0.25]}))
    ops.append(_op_entry("op08", "Gate", ["biased"], ["gate"], {"tokens": tokens, "gate": gate}))
    ops.append(_op_entry("op09", "Mix", ["gate", "V"], ["mixed_raw"], {"tokens": tokens, "dim": dim}))
    ops.append(_op_entry("op10", "Residual", ["tokens", "mixed_raw"], ["mixed"], {"tokens": tokens, "dim": dim, "alpha": alpha}))
    ops.append(_op_entry("op11", "HeadH1", ["mixed"], ["h1"], {"input": tokens * dim, "output": ir["model"]["head_h1"]}))
    ops.append(_op_entry("op12", "HeadH2", ["h1"], ["h2"], {"input": ir["model"]["head_h1"], "output": ir["model"]["head_h2"]}))
    ops.append(_op_entry("op13", "Value", ["h2", "wvo", "bvo"], ["value"], {"input": ir["model"]["head_h2"]}))
    ops.append(_op_entry("op14", "WDL", ["h2", "wwdl", "bwdl"], ["wdl"], {"input": ir["model"]["head_h2"], "output": 3}))
    if ir["model"]["adaptive"]:
        ops.append(_op_entry("op15", "Route", ["h1"], ["route"], {"threshold": ir["model"]["threshold"], "t_high": ir["model"]["t_high"]}))
    ir["ops"] = ops
    ir["target"] = {
        "cpu": target.get("cpu", "generic-x86-64"),
        "isa": target.get("isa", "portable"),
        "vector_width": target.get("vector_width", 1),
        "dtype": ir["model"]["dtype"],
        "quantization": quant,
    }
    return ir


def verify_ir(ir):
    errors = []
    if ir.get("ir_version") != IR_VERSION:
        errors.append("bad ir_version %s" % str(ir.get("ir_version")))
    kinds = [o.get("kind") for o in ir.get("ops", [])]
    for need in ["FeatureUpdate", "AccumulatorUpdate", "Tokenize", "Q", "K", "V", "Score", "Gate", "Mix", "Residual", "HeadH1", "HeadH2", "Value", "WDL"]:
        if need not in kinds:
            errors.append("missing op %s" % need)
    tgt = ir.get("target", {})
    if tgt.get("isa") not in VALID_ISA:
        errors.append("unsupported isa %s" % str(tgt.get("isa")))
    model = ir.get("model", {})
    tokens = int(model.get("tokens", 0))
    dim = int(model.get("token_dim", 0))
    if tokens <= 0 or tokens > 16:
        errors.append("unsupported tokens %s" % str(tokens))
    if dim <= 0 or dim > 128:
        errors.append("unsupported dim %s" % str(dim))
    if model.get("quantization") not in ("fp32", "int8", "int16"):
        errors.append("unsupported quantization %s" % str(model.get("quantization")))
    for t in ir.get("tensors", []):
        for d in t.get("shape", []):
            if not isinstance(d, int) or d < 0 or d > 1000000:
                errors.append("bad shape dim in %s" % t.get("name"))
        if t.get("layout") not in VALID_LAYOUT and t.get("layout") not in ("packed", "row-major", "col-major", "interleaved", "index", "acc", "fp32"):
            errors.append("bad layout %s" % str(t.get("layout")))
    return errors
