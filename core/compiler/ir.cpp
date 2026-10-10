/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

#include "core/compiler/ir.h"
namespace rune {
namespace vir {
bool verifyIr(const RuneIr& ir, std::string& err) {
  if (ir.irVersion != "1.0") {
    err = "bad ir_version";
    return false;
  }
  if (ir.target.isa != "portable" && ir.target.isa != "avx2" && ir.target.isa != "avx512") {
    err = "unsupported isa";
    return false;
  }
  if (ir.model.tokens <= 0 || ir.model.tokens > 16) {
    err = "unsupported tokens";
    return false;
  }
  if (ir.model.tokenDim <= 0 || ir.model.tokenDim > 128) {
    err = "unsupported dim";
    return false;
  }
  if (ir.model.quantization != "fp32" && ir.model.quantization != "int8" && ir.model.quantization != "int16") {
    err = "unsupported quantization";
    return false;
  }
  bool need[15] = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
  for (const IrOp& o : ir.ops) {
    if (o.kind == "FeatureUpdate") need[0] = true;
    if (o.kind == "AccumulatorUpdate") need[1] = true;
    if (o.kind == "Tokenize") need[2] = true;
    if (o.kind == "Q") need[3] = true;
    if (o.kind == "K") need[4] = true;
    if (o.kind == "V") need[5] = true;
    if (o.kind == "Score") need[6] = true;
    if (o.kind == "Bias") need[7] = true;
    if (o.kind == "Gate") need[8] = true;
    if (o.kind == "Mix") need[9] = true;
    if (o.kind == "Residual") need[10] = true;
    if (o.kind == "HeadH1") need[11] = true;
    if (o.kind == "HeadH2") need[12] = true;
    if (o.kind == "Value") need[13] = true;
    if (o.kind == "WDL") need[14] = true;
  }
  for (int i = 0; i < 15; ++i) {
    if (!need[i]) {
      err = "missing op";
      return false;
    }
  }
  return true;
}
std::string shapeKey(int tokens, int dim, int h1, int h2, const std::string& kind) {
  char b[64];
  if (kind == "Q" || kind == "K" || kind == "V") {
    snprintf(b, sizeof(b), "%dx%d", tokens, dim);
    return b;
  }
  if (kind == "Score" || kind == "Bias" || kind == "Gate") {
    snprintf(b, sizeof(b), "%dx%d", tokens, tokens);
    return b;
  }
  if (kind == "Mix" || kind == "Residual") {
    snprintf(b, sizeof(b), "%dx%d", tokens, dim);
    return b;
  }
  if (kind == "HeadH1") {
    snprintf(b, sizeof(b), "%dx%d", h1, tokens * dim);
    return b;
  }
  if (kind == "HeadH2") {
    snprintf(b, sizeof(b), "%dx%d", h1 == 0 ? 0 : h2, h1);
    return b;
  }
  if (kind == "Value") {
    snprintf(b, sizeof(b), "1x%d", h2);
    return b;
  }
  if (kind == "WDL") {
    snprintf(b, sizeof(b), "3x%d", h2);
    return b;
  }
  snprintf(b, sizeof(b), "%dx%d", tokens, dim);
  return b;
}
std::string kernelFor(const std::string& kind, const std::string& shape) {
  if (kind == "Q" || kind == "K" || kind == "V") return shape == "8x32" ? "qkv_fused_8x32" : "matvec_generic";
  if (kind == "Score" || kind == "Bias" || kind == "Gate") return shape == "8x8" ? "score_bias_gate_8x8" : "generic";
  if (kind == "Mix" || kind == "Residual") return shape == "8x32" ? "mix_residual_8x32" : "generic";
  if (kind == "HeadH1" || kind == "HeadH2") return "linear_bias_clip";
  if (kind == "Value") return "dot_tanh";
  if (kind == "WDL") return "matvec_generic";
  if (kind == "FeatureUpdate") return "feature_pack";
  if (kind == "AccumulatorUpdate") return "accum_grouped";
  if (kind == "Tokenize") return "clip";
  if (kind == "Route") return "route_compare";
  return "generic";
}
}
}
