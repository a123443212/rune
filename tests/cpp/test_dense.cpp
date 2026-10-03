#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "core/architectures/dense/dense.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/model_io/model_io.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

namespace {

std::vector<int> kWidths = {24, 40, 32, 28, 20, 44, 36, 12};
std::vector<int> kWidthsD = {24, 28, 32, 20, 16, 30, 26, 12};

void makeEmbeddings(VarEmbeddings& emb, const std::vector<int>& dims) {
  VarWidths w;
  std::string err;
  CHECK(VarWidths::make(dims, w, err));
  emb.configure(w);
  emb.init(7);
}

void extract(Board& b, std::vector<ActiveFeature>& feats) {
  GroupedFeatureSet::extract(b, feats);
}

bool tokensEqual(const float* a, const float* b, int n, float tol) {
  for (int i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

}  // namespace

void testVarWidths() {
  VarWidths w;
  std::string err;
  CHECK(VarWidths::make({24, 40, 32, 28, 20, 44, 36, 12}, w, err));
  CHECK(w.total() == 236);
  CHECK(!VarWidths::make({32, 32}, w, err));
  CHECK(!VarWidths::make({32, 32, 32, 32, 32, 32, 32, 4}, w, err));
  CHECK(!VarWidths::make({32, 32, 32, 32, 32, 32, 32, 128}, w, err));
}

void testVarAccumulatorIncremental() {
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1",
      "8/2P5/1K6/8/8/1k6/8/8 w - - 0 1",
  };
  for (const std::vector<int>& dims :
       std::vector<std::vector<int>>{kWidths, {16, 16, 16, 16, 16, 16, 16, 16}}) {
    VarEmbeddings emb;
    makeEmbeddings(emb, dims);
    int total = 0;
    for (int d : dims) total += d;
    for (const char* fen : fens) {
      Board b(fen);
      VarAccumulator inc;
      inc.configure(&emb);
      std::vector<ActiveFeature> prev;
      extract(b, prev);
      inc.refresh(prev);
      std::mt19937 rng(99);
      for (int step = 0; step < 60; ++step) {
        std::vector<Move> moves;
        b.generateLegalMoves(moves);
        if (moves.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        CHECK(b.makeMove(moves[pick(rng)]));
        std::vector<ActiveFeature> cur;
        extract(b, cur);
        std::vector<ActiveFeature> added, removed;
        GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
        inc.applyDiff(added, removed);
        VarAccumulator ref;
        ref.configure(&emb);
        ref.refresh(cur);
        std::vector<float> ti(total), tr(total);
        inc.tokens(ti.data());
        ref.tokens(tr.data());
        CHECK(tokensEqual(ti.data(), tr.data(), total, 1e-5f));
        prev = cur;
      }
      for (int step = 0; step < 30; ++step) {
        if (b.historySize() == 0) break;
        b.unmakeMove();
        std::vector<ActiveFeature> cur;
        extract(b, cur);
        std::vector<ActiveFeature> added, removed;
        GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
        inc.applyDiff(added, removed);
        VarAccumulator ref;
        ref.configure(&emb);
        ref.refresh(cur);
        std::vector<float> ti(total), tr(total);
        inc.tokens(ti.data());
        ref.tokens(tr.data());
        CHECK(tokensEqual(ti.data(), tr.data(), total, 1e-5f));
        prev = cur;
      }
    }
  }
}

void testPoolGateMath() {
  VarWidths w;
  std::string err;
  CHECK(VarWidths::make(kWidths, w, err));
  TokenPool pool;
  CHECK(pool.configure(w, PoolMode::PerToken, true, 32, err));
  for (int t = 0; t < 8; ++t) {
    for (size_t i = 0; i < pool.perW[t].size(); ++i) pool.perW[t][i] = 0.01f * (i % 7);
    for (size_t i = 0; i < pool.perB[t].size(); ++i) pool.perB[t][i] = -0.2f + 0.05f * (i % 5);
  }
  int total = w.total();
  std::vector<float> in(total), out(total);
  for (int i = 0; i < total; ++i) in[i] = 0.05f * (i % 11);
  pool.forward(in.data(), out.data());
  int off = 0;
  for (int t = 0; t < 8; ++t) {
    int d = kWidths[t];
    for (int r = 0; r < d; ++r) {
      float acc = pool.perB[t][r];
      for (int c = 0; c < d; ++c) acc += pool.perW[t][r * d + c] * in[off + c];
      float want = acc < 0 ? 0 : (acc > 1 ? 1 : acc);
      CHECK_CLOSE(out[off + r], want, 1e-5);
    }
    off += d;
  }
  TokenPool shared;
  CHECK(shared.configure(w, PoolMode::Shared, true, 32, err) == false);
  VarWidths wsh;
  CHECK(VarWidths::make(kWidthsD, wsh, err));
  CHECK(shared.configure(wsh, PoolMode::Shared, true, 32, err));
  VarEmbeddings emb;
  emb.configure([&] {
    VarWidths gw;
    for (int g = 0; g < 8; ++g) gw.w[g] = 32;
    return gw;
  }());
  emb.init(3);
  std::vector<float> acc32(256);
  for (int i = 0; i < 256; ++i) acc32[i] = 0.04f * (i % 9);
  int totalSh = 0;
  for (int d : kWidthsD) totalSh += d;
  std::vector<float> sout(totalSh);
  shared.forward(acc32.data(), sout.data());
  int inBase = 0;
  int outBase = 0;
  for (int t = 0; t < 8; ++t) {
    int d = kWidthsD[t];
    for (int r = 0; r < d; ++r) {
      float acc = 0.0f;
      for (int c = 0; c < 32; ++c) acc += shared.sharedS[r * 32 + c] * acc32[inBase + c];
      float want = acc * shared.tokS[t][r] + shared.tokB[t][r];
      want = want < 0 ? 0 : (want > 1 ? 1 : want);
      CHECK_CLOSE(sout[outBase + r], want, 1e-5);
    }
    inBase += 32;
    outBase += d;
  }
  ChannelGate gate;
  CHECK(gate.configure(w, true, err));
  for (size_t i = 0; i < gate.ga.size(); ++i) {
    gate.ga[i] = 0.5f;
    gate.gb[i] = 0.1f;
  }
  std::vector<float> gout(total);
  gate.forward(out.data(), gout.data());
  for (int i = 0; i < total; ++i) {
    float g = 0.5f * out[i] + 0.1f;
    if (g < 0) g = 0;
    if (g > 1) g = 1;
    CHECK_CLOSE(gout[i], out[i] * g, 1e-6);
  }
  ChannelGate off_gate;
  CHECK(off_gate.configure(w, false, err));
  CHECK(off_gate.parameterCount() == 0);
  std::vector<float> pass(total);
  off_gate.forward(out.data(), pass.data());
  CHECK(tokensEqual(pass.data(), out.data(), total, 0.0f));
}

void testDenseModelIO() {
  for (const std::string& variant : {"B", "C", "D"}) {
    std::vector<int> dims = (variant == "D") ? kWidthsD : kWidths;
    DenseBuildSpec spec;
    spec.variant = variant;
    spec.dims = dims;
    spec.pooling = (variant == "D") ? "shared" : "per_token";
    spec.poolClip = true;
    spec.gateOn = (variant == "C");
    spec.sharedWidth = 32;
    DenseModel model;
    std::string err;
    CHECK(model.configure(spec, err));
    VarEmbeddings emb;
    if (variant == "D") {
      VarWidths gw;
      for (int g = 0; g < 8; ++g) gw.w[g] = 32;
      emb.configure(gw);
    } else {
      VarWidths w;
      CHECK(VarWidths::make(dims, w, err));
      emb.configure(w);
    }
    emb.init(11);
    ModelSpec mspec = model.spec();
    for (const std::string& quant : {"fp32", "int8", "int16"}) {
      std::string path = "/tmp/rune_dense_" + variant + "_" + quant + ".rune";
      CHECK(saveDenseRuneFile(path, mspec, emb, model, quant, err));
      RuneFile loaded;
      CHECK(loadRuneFile(path, loaded, err));
      CHECK(loaded.isDense);
      CHECK(loaded.spec.variant == variant);
      CHECK(loaded.spec.tokenDims == dims);
      Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
      DenseEvaluator ref;
      CHECK(ref.configure(&emb, &model, err));
      DenseEvaluator got;
      DenseModel* lm = static_cast<DenseModel*>(loaded.arch.get());
      CHECK(got.configure(&loaded.varEmbeddings, lm, err));
      float v1, v2, w1[3], w2[3];
      ref.evaluateBoard(b, v1, w1);
      got.evaluateBoard(b, v2, w2);
      float tol = (quant == "fp32") ? 1e-6f : 0.02f;
      CHECK_CLOSE(v1, v2, tol);
    }
    std::string path = "/tmp/rune_dense_" + variant + "_fp32.rune";
    {
      std::ifstream f(path, std::ios::binary);
      std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      bytes[bytes.size() - 1] ^= 0xFF;
      std::ofstream o(path + ".bad", std::ios::binary);
      o.write(bytes.data(), bytes.size());
    }
    RuneFile bad;
    CHECK(!loadRuneFile(path + ".bad", bad, err));
    CHECK(err == "checksum mismatch");
  }
  RuneFile rej;
  std::string err;
  CHECK(!loadRuneFile("/tmp/rune_dense_B_fp32.rune.nonexistent", rej, err));
  {
    std::ofstream o("/tmp/rune_dense_bogus.rune", std::ios::binary);
    std::string header = "{\"format\":1,\"arch\":\"NOPE\",\"arch_version\":\"9.9.9\"}";
    o.write("RUNE", 4);
    uint32_t hlen = header.size();
    o.write(reinterpret_cast<const char*>(&hlen), 4);
    o.write(header.data(), header.size());
  }
  CHECK(!loadRuneFile("/tmp/rune_dense_bogus.rune", rej, err));
}

void testDenseParamAccounting() {
  DenseBuildSpec spec;
  spec.variant = "C";
  spec.dims = kWidths;
  spec.pooling = "per_token";
  spec.gateOn = true;
  DenseModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  model.getTensors(names, shapes, data);
  size_t n = 0;
  for (auto& sh : shapes) {
    size_t c = 1;
    for (int s : sh) c *= s;
    n += c;
  }
  CHECK(model.parameterCount() == n);
  DenseModel plain;
  DenseBuildSpec pspec;
  pspec.variant = "A";
  CHECK(plain.configure(pspec, err));
  CHECK(plain.pool.parameterCount() == 0);
  CHECK(plain.gate.parameterCount() == 0);
}

void testDenseStudentWidths() {
  DenseBuildSpec spec;
  spec.variant = "A";
  spec.dims = {16, 16, 16, 16, 16, 16, 16, 16};
  spec.headH1 = 64;
  spec.headH2 = 16;
  DenseModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  ModelSpec ms = model.spec();
  CHECK(ms.headH1 == 64 && ms.headH2 == 16);
  DenseBuildSpec bad = spec;
  bad.headH1 = 2;
  CHECK(!model.configure(bad, err));
  VarEmbeddings emb;
  VarWidths w;
  CHECK(VarWidths::make(spec.dims, w, err));
  emb.configure(w);
  emb.init(5);
  std::string path = "/tmp/rune_dense_s2_fp32.rune";
  CHECK(saveDenseRuneFile(path, ms, emb, model, "fp32", err));
  RuneFile loaded;
  CHECK(loadRuneFile(path, loaded, err));
  CHECK(loaded.spec.headH1 == 64 && loaded.spec.headH2 == 16);
  DenseEvaluator ref;
  CHECK(ref.configure(&emb, &model, err));
  DenseEvaluator got;
  DenseModel* lm = static_cast<DenseModel*>(loaded.arch.get());
  CHECK(got.configure(&loaded.varEmbeddings, lm, err));
  Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  float v1, v2, w1[3], w2[3];
  ref.evaluateBoard(b, v1, w1);
  got.evaluateBoard(b, v2, w2);
  CHECK_CLOSE(v1, v2, 1e-6f);
}
