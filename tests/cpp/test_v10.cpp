#include "tests/cpp/test_v10.h"
#include "tests/cpp/test_framework.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/adaptive/adaptive.h"
#include "core/architectures/dense/dense.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/kernels/ref_kernels.h"
#include "core/model_io/model_io.h"
using namespace rune;

namespace {

bool readWhole(const std::string& path, std::string& bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  bytes = ss.str();
  return true;
}

bool writeWhole(const std::string& path, const std::string& bytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return true;
}

void replaceAll(std::string& s, const std::string& from, const std::string& to) {
  size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
}

bool craftMutantFrom(const std::string& src, const std::string& dst, const std::string& from,
                     const std::string& to) {
  std::string bytes;
  if (!readWhole(src, bytes)) return false;
  if (bytes.size() < 8) return false;
  uint32_t hlen = 0;
  for (int i = 0; i < 4; ++i) hlen |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[4 + i])) << (8 * i);
  if (8 + hlen > bytes.size()) return false;
  std::string header = bytes.substr(8, hlen);
  std::string payload = bytes.substr(8 + hlen);
  replaceAll(header, from, to);
  std::string out = bytes.substr(0, 4);
  uint32_t nh = static_cast<uint32_t>(header.size());
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((nh >> (8 * i)) & 0xFF));
  out += header;
  out += payload;
  return writeWhole(dst, out);
}

bool craftMutant(const std::string& dst, const std::string& from, const std::string& to) {
  return craftMutantFrom("spec/test-vectors/models/tiny-mlp-fp32.rune", dst, from, to);
}

bool expectReject(const std::string& path) {
  RuneFile rf;
  std::string err;
  bool ok = loadRuneFile(path, rf, err);
  CHECK(!ok);
  CHECK(!err.empty());
  return !ok;
}

}

static void testMalformedLoader() {
  CHECK(craftMutant("/tmp/rune_evil_shape.rune",
                    "{\"name\":\"emb0\",\"shape\":[256,32],\"dtype\":\"float32\"}",
                    "{\"name\":\"emb0\",\"shape\":[\"x\",32],\"dtype\":\"float32\"}"));
  CHECK(expectReject("/tmp/rune_evil_shape.rune"));
  CHECK(craftMutant("/tmp/rune_evil_emb9.rune", "\"name\":\"emb7\"", "\"name\":\"emb9\""));
  CHECK(expectReject("/tmp/rune_evil_emb9.rune"));
  CHECK(craftMutant("/tmp/rune_evil_embempty.rune", "\"name\":\"emb7\"", "\"name\":\"emb\""));
  CHECK(expectReject("/tmp/rune_evil_embempty.rune"));
  CHECK(craftMutant("/tmp/rune_evil_embneg.rune", "\"name\":\"emb7\"", "\"name\":\"emb-1\""));
  CHECK(expectReject("/tmp/rune_evil_embneg.rune"));
  CHECK(craftMutant("/tmp/rune_evil_dtype.rune",
                    "{\"name\":\"emb0\",\"shape\":[256,32],\"dtype\":\"float32\"}",
                    "{\"name\":\"emb0\",\"shape\":[256,32],\"dtype\":\"int4\"}"));
  CHECK(expectReject("/tmp/rune_evil_dtype.rune"));
  CHECK(craftMutant("/tmp/rune_evil_shapeoob.rune",
                    "{\"name\":\"emb0\",\"shape\":[256,32],\"dtype\":\"float32\"}",
                    "{\"name\":\"emb0\",\"shape\":[300,32],\"dtype\":\"float32\"}"));
  CHECK(expectReject("/tmp/rune_evil_shapeoob.rune"));
  CHECK(craftMutant("/tmp/rune_evil_tokens.rune", "\"tokens\":8", "\"tokens\":16"));
  CHECK(expectReject("/tmp/rune_evil_tokens.rune"));
  CHECK(craftMutantFrom("spec/test-vectors/models/dense-b-fp32.rune", "/tmp/rune_evil_hyper.rune",
                        "\"head_h1\":128", "\"head_h1\":1000000000"));
  CHECK(expectReject("/tmp/rune_evil_hyper.rune"));
  {
    std::string bytes;
    CHECK(readWhole("spec/test-vectors/models/tiny-mlp-fp32.rune", bytes));
    CHECK(writeWhole("/tmp/rune_evil_trunc.rune", bytes.substr(0, bytes.size() / 2)));
    CHECK(expectReject("/tmp/rune_evil_trunc.rune"));
  }
  {
    std::string bytes;
    CHECK(readWhole("spec/test-vectors/models/tiny-mlp-fp32.rune", bytes));
    bytes[0] = 'X';
    CHECK(writeWhole("/tmp/rune_evil_magic.rune", bytes));
    CHECK(expectReject("/tmp/rune_evil_magic.rune"));
  }
  {
    std::string bytes;
    CHECK(readWhole("spec/test-vectors/models/tiny-mlp-fp32.rune", bytes));
    bytes[bytes.size() - 1] ^= 0xFF;
    CHECK(writeWhole("/tmp/rune_evil_hash.rune", bytes));
    CHECK(expectReject("/tmp/rune_evil_hash.rune"));
  }
  {
    RuneFile rf;
    std::string err;
    CHECK(loadRuneFile("spec/test-vectors/models/tiny-mlp-fp32.rune", rf, err));
  }
}
#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/adaptive/adaptive.h"
#include "core/architectures/dense/dense.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/kernels/ref_kernels.h"
#include "core/model_io/model_io.h"
using namespace rune;
static void testQuantHalfAway() {
  float scale = 0.05f;
  CHECK(ref::quantizeHalfAway(0.0f, scale, 127) == 0);
  CHECK(ref::quantizeHalfAway(0.5f, scale, 127) == 10);
  CHECK(ref::quantizeHalfAway(-0.5f, scale, 127) == -10);
  CHECK(ref::quantizeHalfAway(1.0f, scale, 127) == 20);
  CHECK(ref::quantizeHalfAway(-1.0f, scale, 127) == -20);
  CHECK(ref::quantizeHalfAway(0.123f, scale, 127) == 2);
  CHECK(ref::quantizeHalfAway(-0.987f, scale, 127) == -20);
  CHECK(ref::quantizeHalfAway(127.0f, scale, 127) == 127);
  CHECK(ref::quantizeHalfAway(-127.0f, scale, 127) == -127);
  float nan = std::nanf("");
  CHECK(ref::quantizeHalfAway(nan, scale, 127) == 0);
  CHECK(ref::quantizeHalfAway(0.5f, scale, 32767) == 10);
}
static void testStartposFeatures() {
  Board b;
  std::vector<ActiveFeature> f;
  GroupedFeatureSet::extract(b, f);
  CHECK(f.size() == 214);
  for (size_t i = 1; i < f.size(); ++i) CHECK(!(f[i] < f[i - 1]));
  for (auto& af : f) {
    CHECK(af.group < 9);
    CHECK(af.index < GroupedFeatureSet::vocabSize(af.group));
  }
  CHECK(std::string(GroupedFeatureSet::version()) == "grouped_hkav2_fullthreats_v02");
}
static void testRefreshVsIncremental() {
  Board a;
  std::vector<Move> lm;
  a.generateLegalMoves(lm);
  Board b = a;
  b.makeMove(lm[0]);
  std::vector<ActiveFeature> f0, f1, added, removed;
  GroupedFeatureSet::extract(a, f0);
  GroupedFeatureSet::extract(b, f1);
  GroupedFeatureSet::diffFeatures(f0, f1, added, removed);
  EmbeddingTables t;
  t.init(7);
  GroupedAccumulator r(&t);
  r.refresh(f1);
  float tr[256];
  r.tokens(tr);
  GroupedAccumulator inc(&t);
  inc.refresh(f0);
  inc.applyDiff(added, removed);
  float ti[256];
  inc.tokens(ti);
  for (int i = 0; i < 256; ++i) CHECK(std::fabs(tr[i] - ti[i]) < 1e-6f);
}
static void testRouting() {
  CHECK(ref::routingRefine(0.9f, 0.5f, 0.5f, false, 0.5f) == true);
  CHECK(ref::routingRefine(0.1f, 0.5f, 0.5f, false, 0.5f) == false);
  CHECK(ref::routingRefine(std::nanf(""), 0.5f, 0.5f, false, 0.5f) == true);
  CHECK(ref::routingRefine(0.4f, 0.5f, 0.8f, true, 0.2f) == false);
  CHECK(ref::routingRefine(0.1f, 0.5f, 0.8f, true, 0.2f) == false);
}
static void testFixtureLoads() {
  const char* files[] = {
    "spec/test-vectors/models/tiny-mlp-fp32.rune",
    "spec/test-vectors/models/small-gab-fp32.rune",
    "spec/test-vectors/models/small-gab-int8.rune",
    "spec/test-vectors/models/small-gab-int16.rune",
    "spec/test-vectors/models/rel-08x32-fp32.rune",
    "spec/test-vectors/models/dense-b-fp32.rune",
    "spec/test-vectors/models/dense-b-int8.rune",
    "spec/test-vectors/models/dense-b-int16.rune",
    "spec/test-vectors/models/adaptive-fp32.rune",
  };
  Board start;
  for (auto* p : files) {
    RuneFile rf;
    std::string err;
    bool ok = loadRuneFile(p, rf, err);
    CHECK(ok);
    if (!ok) continue;
    CHECK(rf.spec.featureSet == "grouped_hkav2_fullthreats_v02");
    const EmbeddingTables* tab = &rf.embeddings;
    if (rf.isFlex) {
      RelationalEvaluator rev;
      std::string e2;
      CHECK(rev.configure(&rf.flexEmbeddings, &rf.layout, static_cast<RelationalModel*>(rf.arch.get()), e2));
      auto r = rev.evaluateBoard(start);
      CHECK(r.value >= -1.0f && r.value <= 1.0f);
    } else if (rf.isDense) {
      DenseEvaluator dev;
      std::string e2;
      CHECK(dev.configure(&rf.varEmbeddings, static_cast<DenseModel*>(rf.arch.get()), e2));
      float v;
      float w[3];
      dev.evaluateBoard(start, v, w);
      CHECK(v >= -1.0f && v <= 1.0f);
    } else if (rf.isAdaptive || rf.isUncertainty) {
      AdaptiveEvaluator aev;
      std::string e2;
      CHECK(aev.configure(&rf.varEmbeddings, static_cast<AdaptiveModel*>(rf.arch.get()), e2));
      auto r = aev.evaluateBoard(start, AdaptiveMode::Adaptive, 0.0f, false, false);
      CHECK(r.value >= -1.0f && r.value <= 1.0f);
      CHECK(r.difficulty == r.difficulty);
    } else {
      Evaluator ev(tab, rf.arch.get());
      auto r = ev.evaluateBoard(start);
      CHECK(r.value >= -1.0f && r.value <= 1.0f);
    }
  }
}
static void testDenseQuantCloseness() {
  Board start;
  float vfp = 0.0f, vi8 = 0.0f;
  for (int k = 0; k < 2; ++k) {
    const char* p = k == 0 ? "spec/test-vectors/models/dense-b-fp32.rune"
                           : "spec/test-vectors/models/dense-b-int8.rune";
    RuneFile rf;
    std::string err;
    CHECK(loadRuneFile(p, rf, err));
    CHECK(rf.isDense);
    DenseEvaluator dev;
    CHECK(dev.configure(&rf.varEmbeddings, static_cast<DenseModel*>(rf.arch.get()), err));
    float v;
    float w[3];
    dev.evaluateBoard(start, v, w);
    if (k == 0) vfp = v; else vi8 = v;
  }
  CHECK(std::fabs(vfp - vi8) < 0.01f);
}
static void testMakeUnmakeParity() {
  Board b;
  std::string k0 = b.toFen();
  std::vector<Move> lm;
  b.generateLegalMoves(lm);
  Board c = b;
  std::vector<ActiveFeature> f0;
  GroupedFeatureSet::extract(b, f0);
  c.makeMove(lm[0]);
  std::vector<ActiveFeature> f1;
  GroupedFeatureSet::extract(c, f1);
  c.unmakeMove();
  CHECK(c.toFen() == k0);
  std::vector<ActiveFeature> f2;
  GroupedFeatureSet::extract(c, f2);
  CHECK(f0.size() == f2.size());
  for (size_t i = 0; i < f0.size(); ++i) CHECK(f0[i] == f2[i]);
}
void runV10Tests() {
  testQuantHalfAway();
  testStartposFeatures();
  testRefreshVsIncremental();
  testRouting();
  testFixtureLoads();
  testDenseQuantCloseness();
  testMalformedLoader();
  testMakeUnmakeParity();
}
