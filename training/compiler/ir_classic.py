# RUNE — Relational Unified Neural Evaluator
# Copyright (C) 2026 a123443212
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# This project is dual-licensed under the MIT License and the
# Apache License, Version 2.0. You may choose either license
# when using, copying, modifying, or distributing this software.
#
# MIT License: https://opensource.org/license/mit
# Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, either express or implied.

def _tensor_entry(name, shape, dtype, layout, constant, lifetime):
    return {"name": name, "shape": list(shape), "dtype": dtype, "layout": layout, "constant": bool(constant), "lifetime": list(lifetime)}


def _op_entry(op_id, kind, inputs, outputs, attrs):
    return {"id": op_id, "kind": kind, "inputs": list(inputs), "outputs": list(outputs), "attrs": dict(attrs)}


def build_classic(model_spec, tensor_metas, target):
    from training.compiler.ir_registry import COMPILER_VERSION, IR_VERSION, SPEC_VERSION
    ir = {"ir_version": IR_VERSION, "spec_version": SPEC_VERSION, "model": {}, "tensors": [], "ops": [], "memory": {}, "fusion": [], "kernel_plan": [], "target": {}, "hashes": {}}
    arch = model_spec.get("arch", "")
    tokens = int(model_spec.get("tokens", 8))
    dim = int(model_spec.get("token_dim", 32))
    quant = model_spec.get("quantization", "fp32")
    gate = model_spec.get("gate", "clip")
    alpha = float(model_spec.get("alpha", 1.0))
    ir["model"] = {"architecture": arch, "architecture_version": model_spec.get("arch_version", "0.2.0"), "tokens": tokens, "token_dim": dim, "dtype": "fp32" if quant == "fp32" else quant, "quantization": quant, "gate": gate, "alpha": alpha, "head_h1": int(model_spec.get("head_h1", 128)), "head_h2": int(model_spec.get("head_h2", 32)), "threshold": float(model_spec.get("threshold", 0.5)), "t_high": float(model_spec.get("t_high", model_spec.get("threshold", 0.5))), "t_low": model_spec.get("t_low", None), "has_t_low": bool("t_low" in model_spec and model_spec["t_low"] is not None)}
    ir["model"]["adaptive"] = arch in ("RUNE-04", "RUNE-05")
    ir["model"]["board_size"] = 0
    ir["model"]["channels"] = 0
    ir["model"]["num_blocks"] = 0
    ir["model"]["policy_size"] = 0
    tensors = []
    tensors.append(_tensor_entry("features", [tokens], "index", "packed", False, [0, 1]))
    tensors.append(_tensor_entry("accumulator", [tokens, dim], "acc", "row-major", False, [0, 2]))
    tensors.append(_tensor_entry("tokens", [tokens, dim], "fp32", "row-major", False, [1, 3]))
    for nm in ("wq", "bq", "wk", "bk", "wvv", "bvv", "wv", "bv", "gabS", "gab"):
        for tm in tensor_metas:
            if tm.get("name") == nm:
                tensors.append(_tensor_entry(nm, tm.get("shape", []), tm.get("dtype", "float32"), "row-major", True, [0, 12]))
    for nm in ("w1", "b1", "wgate", "bgate", "wup", "bup", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"):
        for tm in tensor_metas:
            if tm.get("name") == nm:
                tensors.append(_tensor_entry(nm, tm.get("shape", []), tm.get("dtype", "float32"), "row-major", True, [0, 12]))
    for g in range(9):
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
    if arch == "RUNE-ATTN-SOFT":
        ops.append(_op_entry("op08", "Softmax", ["biased"], ["gate"], {"tokens": tokens}))
    else:
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
    ir["target"] = {"cpu": target.get("cpu", "generic-x86-64"), "isa": target.get("isa", "portable"), "vector_width": target.get("vector_width", 1), "dtype": ir["model"]["dtype"], "quantization": quant}
    return ir
