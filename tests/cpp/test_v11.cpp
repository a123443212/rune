#include "tests/cpp/test_v11.h"
#include "tests/cpp/test_framework.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include "core/compiler/artifact.h"
#include "core/compiler/ir.h"
#include "core/compiler/plan.h"
#include "core/accumulators/grouped_accumulator.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/kernels/accum_specialized.h"
#include "core/kernels/fused.h"
#include "core/kernels/quant_specialized.h"
#include "core/kernels/ref_kernels.h"
#include "core/kernels/smallmat.h"
#include "core/runtime/arena.h"
using namespace rune;
static void testSmallMatVec() {
  float mat[1024];
  float vec[32];
  float bias[32];
  float a[32];
  float b[32];
  for (int i = 0; i < 1024; ++i) mat[i] = (float)(i % 17) * 0.01f - 0.08f;
  for (int i = 0; i < 32; ++i) {
    vec[i] = (float)(i % 7) * 0.1f;
    bias[i] = (float)(i % 5) * 0.01f;
  }
  ref::matVec(mat, vec, bias, a, 32, 32);
  small::matVec32(mat, vec, bias, b);
  for (int i = 0; i < 32; ++i) CHECK(std::fabs(a[i] - b[i]) < 1e-6);
}
static void testFusedQkv() {
  float w[1024];
  float b[32];
  float x[256];
  for (int i = 0; i < 1024; ++i) w[i] = (float)(i % 13) * 0.005f - 0.03f;
  for (int i = 0; i < 32; ++i) b[i] = 0.01f;
  for (int i = 0; i < 256; ++i) x[i] = (float)(i % 11) * 0.05f;
  float q[256];
  float k[256];
  float v[256];
  fused::qkvFused8x32(w, b, w, b, w, b, x, q, k, v);
  float e[32];
  ref::matVec(w, x, b, e, 32, 32);
  for (int i = 0; i < 32; ++i) CHECK(std::fabs(q[i] - e[i]) < 1e-5);
}
static void testScoreGate() {
  float q[256];
  float k[256];
  float gab[64];
  for (int i = 0; i < 256; ++i) {
    q[i] = (float)(i % 9) * 0.02f;
    k[i] = (float)(i % 5) * 0.03f;
  }
  for (int i = 0; i < 64; ++i) gab[i] = 0.01f;
  float s[64];
  float g[64];
  fused::scoreBiasGate8x8(q, k, gab, GateFn::Clip, s, g);
  float s2[64];
  ref::matMulTT(q, k, s2, 8, 8, 32);
  for (int i = 0; i < 64; ++i) CHECK(std::fabs(s[i] - (s2[i] + gab[i])) < 1e-4);
  for (int i = 0; i < 64; ++i) CHECK(g[i] >= 0.0f && g[i] <= 1.0f);
}
static void testQuantSpec() {
  int32_t acc[8] = {10, -20, 0, 127, -127, 50, -50, 5};
  float out[8];
  qspec::dequantClip(acc, 0.01f, out, 8);
  for (int i = 0; i < 8; ++i) {
    float e = ref::clippedRelu((float)acc[i] * 0.01f);
    CHECK(std::fabs(out[i] - e) < 1e-9);
  }
}
static void testAccumSpec() {
  EmbeddingTables t;
  t.init(11);
  Board start;
  std::vector<ActiveFeature> f;
  GroupedFeatureSet::extract(start, f);
  GroupedAccumulator ref(&t);
  ref.refresh(f);
  float tr[256];
  ref.tokens(tr);
  float got[256];
  aspec::GroupOffsets g = aspec::offsetsFor(32);
  aspec::refreshGrouped(got, &t, f, g);
  float clipped[256];
  for (int i = 0; i < 256; ++i) {
    float v = got[i];
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    clipped[i] = v;
  }
  for (int i = 0; i < 256; ++i) CHECK(std::fabs(tr[i] - clipped[i]) < 1e-6);
}
static void testIrPlan() {
  vir::RuneIr ir;
  ir.irVersion = "1.0";
  ir.model.tokens = 8;
  ir.model.tokenDim = 32;
  ir.model.quantization = "fp32";
  ir.model.dtype = "fp32";
  ir.target.isa = "avx2";
  const char* kinds[15] = {"FeatureUpdate", "AccumulatorUpdate", "Tokenize", "Q", "K", "V", "Score", "Bias", "Gate", "Mix", "Residual", "HeadH1", "HeadH2", "Value", "WDL"};
  for (int i = 0; i < 15; ++i) {
    vir::IrOp o;
    o.id = "op" + std::to_string(i);
    o.kind = kinds[i];
    ir.ops.push_back(o);
  }
  std::string err;
  CHECK(vir::verifyIr(ir, err));
  vplan::selectKernels(ir);
  CHECK(ir.kernels.size() == 15);
  CHECK(ir.arenaBytes > 0);
  bool found = false;
  for (auto& k : ir.kernels) {
    if (k.kernelId == "qkv_fused_8x32") found = true;
  }
  CHECK(found);
}
static void testArena() {
  rt::Arena a(4096);
  CHECK(a.bytes() >= 4096);
  float* p = a.at(0, 8);
  for (int i = 0; i < 8; ++i) p[i] = (float)i;
  CHECK(a.at(0, 8)[3] == 3.0f);
  CHECK(rt::arenaBytesFor(8, 32, 128, 32) > 0);
  bool thrown = false;
  try {
    float* q = a.at(0, 4096);
    for (int i = 0; i < 4096; ++i) q[i] = 1.0f;
  } catch (const std::out_of_range&) {
    thrown = true;
  }
  CHECK(thrown);
}
static void testClipPropagatesNan() {
  float nan = std::nanf("");
  CHECK(!(ref::clippedRelu(nan) == ref::clippedRelu(nan)));
  CHECK(!(ref::hardSigmoid(nan) == ref::hardSigmoid(nan)));
  CHECK(ref::clippedRelu(-0.0f) == 0.0f);
  CHECK(std::signbit(ref::clippedRelu(-0.0f)) != 0);
}
static void testAccumSpecWideGroup() {
  EmbeddingTables t;
  t.init(11);
  std::vector<ActiveFeature> f;
  for (int i = 0; i < 300; ++i) f.push_back({5, static_cast<uint16_t>(i)});
  GroupedAccumulator ref(&t);
  ref.refresh(f);
  float want[256];
  ref.tokens(want);
  float got[256];
  aspec::GroupOffsets g = aspec::offsetsFor(32);
  aspec::refreshGrouped(got, &t, f, g);
  for (int i = 0; i < 256; ++i) {
    float v = got[i] < 0.0f ? 0.0f : (got[i] > 1.0f ? 1.0f : got[i]);
    CHECK(std::fabs(want[i] - v) < 1e-6);
  }
}
static void testArtifactReject() {
  vart::CompiledInfo info;
  std::string header;
  std::string err;
  CHECK(!vart::readCompiledHeader("spec/test-vectors/models/tiny-mlp-fp32.rune", header, info, err));
  CHECK(!err.empty());
}
void runV11Tests() {
  testSmallMatVec();
  testFusedQkv();
  testScoreGate();
  testQuantSpec();
  testAccumSpec();
  testAccumSpecWideGroup();
  testClipPropagatesNan();
  testIrPlan();
  testArena();
  testArtifactReject();
}
