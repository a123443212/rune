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

import sys

sys.path.insert(0, ".")

from training.models import adaptive, dense, rune_models

MACHINE = "linux-x86_64-cmake-O2-scalar"
DATE = "2026-10-05"

CANDIDATE_BUILDERS = {
    "C0-MLP": lambda: rune_models.build_model("RUNE-MLP"),
    "C1-ATTN-GAB-8x32": lambda: rune_models.build_model("RUNE-ATTN-GAB"),
    "C2-Adaptive-04": adaptive.build_adaptive_model,
    "C3-Dense-B": lambda: dense.build_dense_model(variant="B"),
}

CANDIDATES = {
    "C0-MLP": {
        "params": None,
        "flops_k": 74,
        "full_us": 45.0,
        "search": "530n/255e d2, 30k nps",
        "evidence": "baseline, all langs, fixtures",
    },
    "C1-ATTN-GAB-8x32": {
        "params": None,
        "flops_k": 132,
        "full_us": 58.3,
        "incr_us": 44.9,
        "search": "266n/123e d2, 27.5k nps, det 3/3",
        "evidence": "triangle 9/9, simd 3.7x, fixtures fp32/int8/int16",
    },
    "C2-Adaptive-04": {
        "params": 49001,
        "flops_k": 98,
        "full_us": 57.8,
        "cheap_us": 7.9,
        "search": "not wired (void)cfg",
        "evidence": "routing 100%, refine-rate unmeasured trained",
    },
    "C3-Dense-B": {
        "params": 42592,
        "flops_k": 84,
        "full_us": None,
        "search": "rejected by search tool",
        "evidence": "int8 parity 7e-6, no search bench",
    },
}


def main():
    rows = []
    for name, c in CANDIDATES.items():
        model = CANDIDATE_BUILDERS[name]()
        total = sum(p.numel() for p in model.parameters())
        emb = sum(t.numel() for t in model.embedding_tensors().values())
        c["params"] = total - emb
        c["params_true"] = total
    print(f"machine={MACHINE} date={DATE}")
    print("candidate,params_arch,params_true,flops_k,full_us,notes")
    for name, c in CANDIDATES.items():
        print(f"{name},{c['params']},{c['params_true']},{c['flops_k']},{c['full_us']},{c['evidence']}")
    print("quality: unmeasured for all (no trained weights >=25M exist)")
    print("selection: C1-ATTN-GAB-8x32")


if __name__ == "__main__":
    main()
