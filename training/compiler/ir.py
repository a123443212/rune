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

from training.compiler.ir_classic import build_classic
from training.compiler.ir_registry import COMPILER_VERSION, IR_VERSION, SPEC_VERSION, VALID_ISA, arch_family
from training.compiler.ir_resnet import build_resnet
from training.compiler.ir_verify import verify_arch, verify_shape, verify_target

CANONICAL_OPS = ["FeatureUpdate", "AccumulatorUpdate", "Tokenize", "Q", "K", "V", "Score", "Bias", "Gate", "Mix", "Residual", "HeadH1", "HeadH2", "Value", "WDL"]
ADAPTIVE_OPS = ["Route", "Cheap", "Refine"]
VALID_DTYPE = ["fp32", "int8", "int16"]
VALID_LAYOUT = ["row-major", "col-major", "packed", "interleaved"]


def ir_empty():
    return {"ir_version": IR_VERSION, "spec_version": SPEC_VERSION, "model": {}, "tensors": [], "ops": [], "memory": {}, "fusion": [], "kernel_plan": [], "target": {}, "hashes": {}}


def build_ir(model_spec, tensor_metas, target):
    arch = model_spec.get("arch", "")
    if arch_family(arch) == "resnet":
        return build_resnet(model_spec, tensor_metas, target)
    return build_classic(model_spec, tensor_metas, target)


def verify_ir(ir):
    errors = []
    if ir.get("ir_version") not in ("1.0", "1.1"):
        errors.append("bad ir_version %s" % str(ir.get("ir_version")))
    if ir.get("spec_version", "RUNE-10") not in ("RUNE-10", "RUNE-11"):
        errors.append("bad spec_version %s" % str(ir.get("spec_version")))
    errors += verify_arch(ir)
    errors += verify_target(ir)
    errors += verify_shape(ir)
    for t in ir.get("tensors", []):
        for d in t.get("shape", []):
            if not isinstance(d, int) or d < 0 or d > 1000000:
                errors.append("bad shape dim in %s" % t.get("name"))
        if t.get("layout") not in VALID_LAYOUT and t.get("layout") not in ("packed", "row-major", "col-major", "interleaved", "index", "acc", "fp32"):
            errors.append("bad layout %s" % str(t.get("layout")))
    return errors
