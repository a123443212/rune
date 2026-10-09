def _tensor_entry(name, shape, dtype, layout, constant, lifetime):
    return {"name": name, "shape": list(shape), "dtype": dtype, "layout": layout, "constant": bool(constant), "lifetime": list(lifetime)}


def _op_entry(op_id, kind, inputs, outputs, attrs):
    return {"id": op_id, "kind": kind, "inputs": list(inputs), "outputs": list(outputs), "attrs": dict(attrs)}


def build_resnet(model_spec, tensor_metas, target):
    from training.compiler.ir_registry import IR_VERSION, SPEC_VERSION
    board = int(model_spec.get("board_size", model_spec.get("tokens", 9)))
    channels = int(model_spec.get("channels", model_spec.get("token_dim", 32)))
    blocks = int(model_spec.get("num_blocks", 6))
    policy = int(model_spec.get("policy_size", board * board + 1))
    quant = model_spec.get("quantization", "fp32")
    arch = model_spec.get("arch", "RUNE-RESNET-01")
    ir = {"ir_version": IR_VERSION, "spec_version": SPEC_VERSION, "model": {}, "tensors": [], "ops": [], "memory": {}, "fusion": [], "kernel_plan": [], "target": {}, "hashes": {}}
    ir["model"] = {"architecture": arch, "architecture_version": model_spec.get("arch_version", "0.1.0"), "tokens": board, "token_dim": channels, "dtype": "fp32" if quant == "fp32" else quant, "quantization": quant, "gate": "relu", "alpha": 1.0, "head_h1": channels * board * board, "head_h2": int(model_spec.get("head_h2", 256)), "threshold": 0.5, "t_high": 0.5, "t_low": None, "has_t_low": False, "adaptive": False, "board_size": board, "channels": channels, "num_blocks": blocks, "policy_size": policy}
    tensors = [_tensor_entry("planes", [board * board], "index", "packed", False, [0, 1]), _tensor_entry("stem", [board, board], "fp32", "row-major", False, [1, 3])]
    for tm in tensor_metas:
        tensors.append(_tensor_entry(tm.get("name", ""), tm.get("shape", []), tm.get("dtype", "float32"), "row-major", True, [0, 12]))
    ir["tensors"] = tensors
    ops = [_op_entry("op00", "FeaturePlanes", [], ["planes"], {"board": board}), _op_entry("op01", "StemConv", ["planes", "stem_w", "stem_b"], ["x0"], {"board": board, "channels": channels}), _op_entry("op02", "Relu", ["x0"], ["x1"], {})]
    cur = "x1"
    nid = 3
    for b in range(blocks):
        c1 = f"b{b}_c1"
        r1 = f"b{b}_r1"
        c2 = f"b{b}_c2"
        ad = f"b{b}_ad"
        nx = f"x{b + 2}"
        ops.append(_op_entry(f"op{nid:02d}c1", "Conv2D", [cur], [c1], {"board": board, "channels": channels, "block": b}))
        nid += 1
        ops.append(_op_entry(f"op{nid:02d}r1", "Relu", [c1], [r1], {}))
        nid += 1
        ops.append(_op_entry(f"op{nid:02d}c2", "Conv2D", [r1], [c2], {"board": board, "channels": channels, "block": b}))
        nid += 1
        ops.append(_op_entry(f"op{nid:02d}ad", "ResidualAdd", [cur, c2], [ad], {}))
        nid += 1
        ops.append(_op_entry(f"op{nid:02d}rl", "Relu", [ad], [nx], {}))
        nid += 1
        cur = nx
    ops.append(_op_entry("op90", "GlobalPool", [cur], ["pooled"], {"board": board, "channels": channels}))
    ops.append(_op_entry("op91", "Flatten", [cur], ["flat"], {"board": board, "channels": channels}))
    ops.append(_op_entry("op92", "Value", ["pooled", "wv", "bv"], ["value"], {"input": channels}))
    ops.append(_op_entry("op93", "WDL", ["pooled", "wwdl", "bwdl"], ["wdl"], {"input": channels, "output": 3}))
    ops.append(_op_entry("op94", "PolicyLogits", ["flat", "wpol", "bpol"], ["plogits"], {"input": channels * board * board, "output": policy}))
    ops.append(_op_entry("op95", "Policy", ["plogits"], ["policy"], {"output": policy}))
    ir["ops"] = ops
    ir["target"] = {"cpu": target.get("cpu", "generic-x86-64"), "isa": target.get("isa", "portable"), "vector_width": target.get("vector_width", 1), "dtype": ir["model"]["dtype"], "quantization": quant}
    return ir
