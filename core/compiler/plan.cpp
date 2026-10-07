#include "core/compiler/plan.h"
namespace rune {
namespace vplan {
std::string fusionOf(const std::string& kind) {
  if (kind == "Q" || kind == "K" || kind == "V") return "f_qkv";
  if (kind == "Score" || kind == "Bias" || kind == "Gate") return "f_score_bias_gate";
  if (kind == "Mix" || kind == "Residual") return "f_mix_residual";
  if (kind == "HeadH1") return "f_head_h1";
  if (kind == "HeadH2") return "f_head_h2";
  return "";
}
void selectKernels(vir::RuneIr& ir) {
  ir.kernels.clear();
  ir.fusion.clear();
  bool hQ = false, hS = false, hM = false, hH1 = false, hH2 = false;
  for (const vir::IrOp& o : ir.ops) {
    vir::KernelEntry e;
    e.op = o.id;
    e.kind = o.kind;
    e.shape = vir::shapeKey(ir.model.tokens, ir.model.tokenDim, ir.model.headH1, ir.model.headH2, o.kind);
    e.kernelId = vir::kernelFor(o.kind, e.shape);
    if (o.kind == "Tokenize" && (ir.model.quantization == "int8" || ir.model.quantization == "int16")) e.kernelId = "dequant_clip";
    e.dtype = ir.model.dtype;
    e.packing = "row-major-aligned32";
    e.isa = ir.target.isa;
    e.fusionGroup = fusionOf(o.kind);
    ir.kernels.push_back(e);
    if (e.fusionGroup == "f_qkv") hQ = true;
    if (e.fusionGroup == "f_score_bias_gate") hS = true;
    if (e.fusionGroup == "f_mix_residual") hM = true;
    if (e.fusionGroup == "f_head_h1") hH1 = true;
    if (e.fusionGroup == "f_head_h2") hH2 = true;
  }
  bool fixed = ir.model.tokens == 8 && ir.model.tokenDim == 32;
  if (fixed && hQ) ir.fusion.push_back({"f_qkv", {"Q", "K", "V"}, "qkv_fused_8x32"});
  if (fixed && hS) ir.fusion.push_back({"f_score_bias_gate", {"Score", "Bias", "Gate"}, "score_bias_gate_8x8"});
  if (fixed && hM) ir.fusion.push_back({"f_mix_residual", {"Mix", "Residual"}, "mix_residual_8x32"});
  if (hH1) ir.fusion.push_back({"f_head_h1", {"HeadH1"}, "linear_bias_clip"});
  if (hH2) ir.fusion.push_back({"f_head_h2", {"HeadH2"}, "linear_bias_clip"});
  ir.arenaBytes = planArenaBytes(ir.model.tokens, ir.model.tokenDim, ir.model.headH1, ir.model.headH2);
}
size_t planArenaBytes(int tokens, int dim, int h1, int h2) {
  size_t live[10] = {
    (size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * dim,
    (size_t)tokens * dim, (size_t)tokens * tokens, (size_t)tokens * tokens, (size_t)tokens * dim,
    (size_t)h1, (size_t)h2, (size_t)tokens * dim};
  size_t total = 0;
  for (size_t e : live) total += e * 4 + 32;
  return (total + 31) / 32 * 32;
}
double kernelCost(const std::string& shape, const std::string& kernelId, const std::string& isa) {
  double ops = 1.0;
  size_t x = shape.find('x');
  if (x != std::string::npos) {
    double a = atof(shape.substr(0, x).c_str());
    double b = atof(shape.substr(x + 1).c_str());
    ops = a * b;
  }
  double mem = ops * 4.0;
  if (kernelId.find("fused") != std::string::npos || kernelId.rfind("score", 0) == 0 || kernelId.rfind("mix", 0) == 0 || kernelId.rfind("qkv", 0) == 0) mem *= 0.6;
  double f = 1.0;
  if (isa == "avx2") f = 0.35;
  if (isa == "avx512") f = 0.28;
  return ops * f + mem * 0.05;
}
}
}
