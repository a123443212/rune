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

def test_ir_resnet_build():
    from training.compiler.ir import build_ir, verify_ir
    spec = {"arch": "RUNE-RESNET-01", "arch_version": "0.1.0", "tokens": 9, "token_dim": 16, "board_size": 9, "channels": 16, "num_blocks": 2, "policy_size": 82, "quantization": "fp32", "head_h2": 32}
    ir = build_ir(spec, [], {"cpu": "generic-x86-64", "isa": "portable", "vector_width": 1})
    assert ir["ir_version"] == "1.1"
    assert ir["model"]["board_size"] == 9
    kinds = [o["kind"] for o in ir["ops"]]
    for need in ["StemConv", "Conv2D", "ResidualAdd", "Relu", "Value", "WDL", "PolicyLogits", "Policy"]:
        assert need in kinds
    assert verify_ir(ir) == []


def test_ir_resnet_rejects_bad_board():
    from training.compiler.ir import build_ir, verify_ir
    spec = {"arch": "RUNE-RESNET-01", "tokens": 32, "token_dim": 16, "board_size": 32, "channels": 16, "num_blocks": 1, "policy_size": 82, "quantization": "fp32"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "portable", "vector_width": 1})
    assert any("board" in e for e in verify_ir(ir))


def test_ir_classic_still_strict():
    from training.compiler.ir import build_ir, verify_ir
    spec = {"arch": "RUNE-ATTN-GAB", "tokens": 8, "token_dim": 32, "quantization": "fp32"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "portable", "vector_width": 1})
    ir["ops"] = [o for o in ir["ops"] if o.get("kind") != "Gate"]
    assert any("Gate" in e for e in verify_ir(ir))


def test_ir_rejects_unknown_arch():
    from training.compiler.ir import build_ir, verify_ir
    spec = {"arch": "RUNE-XYZ", "tokens": 8, "token_dim": 32, "quantization": "fp32"}
    ir = build_ir(spec, [], {"cpu": "x", "isa": "portable", "vector_width": 1})
    assert any("unsupported architecture" in e for e in verify_ir(ir))
