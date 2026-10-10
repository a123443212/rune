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

def verify_arch(ir):
    from training.compiler.ir_registry import arch_family, required_ops
    errors = []
    arch = ir.get("model", {}).get("architecture", "")
    fam = arch_family(arch)
    if fam == "unknown":
        return [f"unsupported architecture {arch}"]
    kinds = [o.get("kind") for o in ir.get("ops", [])]
    for need in required_ops(arch):
        if need not in kinds:
            errors.append(f"missing op {need} for {arch}")
    return errors


def verify_shape(ir):
    errors = []
    model = ir.get("model", {})
    arch = model.get("architecture", "")
    from training.compiler.ir_registry import arch_family
    fam = arch_family(arch)
    if fam == "resnet":
        board = int(model.get("board_size", model.get("tokens", 0)))
        ch = int(model.get("channels", model.get("token_dim", 0)))
        if board <= 0 or board > 19:
            errors.append(f"unsupported board_size {board}")
        if ch <= 0 or ch > 256:
            errors.append(f"unsupported channels {ch}")
        if int(model.get("num_blocks", 0)) > 64:
            errors.append("unsupported num_blocks")
        if int(model.get("policy_size", 0)) > 512:
            errors.append("unsupported policy_size")
    else:
        tokens = int(model.get("tokens", 0))
        dim = int(model.get("token_dim", 0))
        if tokens <= 0 or tokens > 16:
            errors.append(f"unsupported tokens {tokens}")
        if dim <= 0 or dim > 128:
            errors.append(f"unsupported dim {dim}")
    if model.get("quantization") not in ("fp32", "int8", "int16"):
        errors.append(f"unsupported quantization {model.get('quantization')}")
    return errors


def verify_target(ir):
    from training.compiler.ir_registry import VALID_ISA
    errors = []
    tgt = ir.get("target", {})
    if tgt.get("isa") not in VALID_ISA:
        errors.append(f"unsupported isa {tgt.get('isa')}")
    return errors
