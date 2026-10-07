#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "core/architectures/adaptive/adaptive.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/model_io/model_io.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

namespace {

void makeAdaptiveEmbeddings(VarEmbeddings& emb, int dim) {
  VarWidths w;
  for (int g = 0; g < 8; ++g) w.w[g] = dim;
  emb.configure(w);
  emb.init(7);
}

AdaptiveEvalResult evalMode(AdaptiveEvaluator& ev, AdaptiveMode mode, float threshold) {
  return ev.evaluate(mode, threshold, true, false);
}

}  // namespace

void testAdaptiveConfigure() {
  AdaptiveBuildSpec spec;
  AdaptiveModel m;
  std::string err;
  CHECK(m.configure(spec, err));
  spec.dim = 4;
  CHECK(!m.configure(spec, err));
  spec.dim = 128;
  CHECK(!m.configure(spec, err));
  spec.dim = 16;
  spec.cheapPooling = "shared";
  CHECK(!m.configure(spec, err));
  spec.dim = 32;
  spec.cheapPooling = "shared";
  CHECK(m.configure(spec, err));
  spec.cheapPooling = "bogus";
  CHECK(!m.configure(spec, err));
  spec.cheapPooling = "none";
  spec.prunedPairs.push_back({0, 8});
  CHECK(!m.configure(spec, err));
  spec.prunedPairs.clear();
  spec.prunedPairs.push_back({2, 5});
  CHECK(m.configure(spec, err));
  ModelSpec s = m.spec();
  CHECK(s.arch == "RUNE-04");
  CHECK(s.prunedPairs.size() == 1);
}

void testAdaptiveRoutingModes() {
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
  };
  VarEmbeddings emb;
  makeAdaptiveEmbeddings(emb, 32);
  AdaptiveModel model;
  std::string err;
  AdaptiveBuildSpec spec;
  CHECK(model.configure(spec, err));
  AdaptiveEvaluator ev;
  CHECK(ev.configure(&emb, &model, err));
  for (const char* fen : fens) {
    Board b(fen);
    AdaptiveEvalResult cheap = ev.evaluateBoard(b, AdaptiveMode::Cheap, 0.0f, true, false);
    AdaptiveEvalResult always = ev.evaluateBoard(b, AdaptiveMode::Always, 0.0f, true, false);
    AdaptiveEvalResult allRef = ev.evaluateBoard(b, AdaptiveMode::Adaptive, -1e30f, true, false);
    AdaptiveEvalResult noneRef = ev.evaluateBoard(b, AdaptiveMode::Adaptive, 1e30f, true, false);
    CHECK(cheap.refined == false);
    CHECK(always.refined == true);
    CHECK(allRef.refined == true);
    CHECK(noneRef.refined == false);
    CHECK_CLOSE(allRef.value, always.value, 1e-6);
    CHECK_CLOSE(noneRef.value, cheap.value, 1e-6);
    AdaptiveEvalResult above =
        ev.evaluateBoard(b, AdaptiveMode::Adaptive, cheap.difficulty + 1e-3f, true, false);
    AdaptiveEvalResult below =
        ev.evaluateBoard(b, AdaptiveMode::Adaptive, cheap.difficulty - 1e-3f, true, false);
    CHECK(above.refined == false);
    CHECK(below.refined == true);
    CHECK_CLOSE(above.value, cheap.value, 1e-6);
    AdaptiveEvalResult rep = ev.evaluateBoard(b, AdaptiveMode::Adaptive, 0.25f, true, false);
    AdaptiveEvalResult rep2 = ev.evaluateBoard(b, AdaptiveMode::Adaptive, 0.25f, true, false);
    CHECK_CLOSE(rep.value, rep2.value, 0.0);
    CHECK(rep.refined == rep2.refined);
  }
}

void testAdaptiveHysteresis() {
  AdaptiveModel m;
  std::string err;
  AdaptiveBuildSpec spec;
  spec.hasTLow = true;
  spec.tLow = 0.3f;
  spec.threshold = 0.6f;
  CHECK(m.configure(spec, err));
  CHECK(m.route(0.5f, true, 0.6f) == true);
  CHECK(m.route(0.2f, true, 0.6f) == false);
  CHECK(m.route(0.5f, false, 0.6f) == false);
  CHECK(m.route(0.7f, false, 0.6f) == true);
  AdaptiveBuildSpec plain;
  AdaptiveModel m2;
  CHECK(m2.configure(plain, err));
  CHECK(m2.route(0.5f, true) == true);
  CHECK(m2.route(0.4f, false) == false);
  float nan = std::nanf("");
  CHECK(m2.route(nan, true) == true);
  CHECK(m2.route(nan, false) == true);
}

void testRouteSearchHigh() {
  AdaptiveBuildSpec spec;
  AdaptiveModel m;
  std::string err;
  CHECK(m.configure(spec, err));
  RoutingThresholds t;
  t.diffT = 0.5f;
  t.uncT = 0.5f;
  t.stabT = 0.5f;
  t.tHigh = 0.1f;
  CHECK(m.routeSearch(SearchRoute::Difficulty, 0.2f, 0.0f, 0.0f, false, t) == true);
  float nan = std::nanf("");
  RoutingThresholds t2;
  CHECK(m.routeSearch(SearchRoute::Difficulty, nan, 0.0f, 0.0f, false, t2) == true);
  CHECK(m.routeSearch(SearchRoute::Difficulty, 0.2f, 0.0f, 0.0f, false, t2) == false);
}

void testAdaptiveIncremental() {
  VarEmbeddings emb;
  makeAdaptiveEmbeddings(emb, 32);
  AdaptiveModel model;
  std::string err;
  AdaptiveBuildSpec spec;
  CHECK(model.configure(spec, err));
  Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  AdaptiveEvaluator inc;
  CHECK(inc.configure(&emb, &model, err));
  inc.refresh(b);
  std::vector<ActiveFeature> prev;
  GroupedFeatureSet::extract(b, prev);
  std::mt19937 rng(7);
  for (int step = 0; step < 40; ++step) {
    std::vector<Move> moves;
    b.generateLegalMoves(moves);
    if (moves.empty()) break;
    std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
    CHECK(b.makeMove(moves[pick(rng)]));
    std::vector<ActiveFeature> cur;
    GroupedFeatureSet::extract(b, cur);
    std::vector<ActiveFeature> added, removed;
    GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
    inc.updateIncremental(added, removed);
    AdaptiveEvalResult a = inc.evaluate(AdaptiveMode::Always, 0.0f, true, false);
    AdaptiveEvaluator ref;
    CHECK(ref.configure(&emb, &model, err));
    AdaptiveEvalResult want = ref.evaluateBoard(b, AdaptiveMode::Always, 0.0f, true, false);
    CHECK_CLOSE(a.value, want.value, 1e-5);
    CHECK_CLOSE(a.difficulty, want.difficulty, 1e-5);
    prev = cur;
  }
}

void testAdaptiveModelIO() {
  VarEmbeddings emb;
  makeAdaptiveEmbeddings(emb, 32);
  AdaptiveBuildSpec spec;
  spec.threshold = 0.25f;
  spec.prunedPairs.push_back({0, 4});
  AdaptiveModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  ModelSpec mspec = model.spec();
  for (const std::string& quant : {"fp32", "int8", "int16"}) {
    std::string path = "/tmp/rune_adaptive_" + quant + ".rune";
    CHECK(saveAdaptiveRuneFile(path, mspec, emb, model, quant, err));
    RuneFile loaded;
    CHECK(loadRuneFile(path, loaded, err));
    CHECK(loaded.isAdaptive);
    CHECK(loaded.spec.arch == "RUNE-04");
    CHECK(loaded.spec.archVersion == "0.4.0");
    CHECK(loaded.spec.prunedPairs.size() == 1);
    Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    AdaptiveEvaluator ref;
    CHECK(ref.configure(&emb, &model, err));
    AdaptiveEvaluator got;
    AdaptiveModel* lm = static_cast<AdaptiveModel*>(loaded.arch.get());
    CHECK(got.configure(&loaded.varEmbeddings, lm, err));
    AdaptiveEvalResult v1 = ref.evaluateBoard(b, AdaptiveMode::Always, 0.0f, true, false);
    AdaptiveEvalResult v2 = got.evaluateBoard(b, AdaptiveMode::Always, 0.0f, true, false);
    float tol = (quant == "fp32") ? 1e-6f : 0.02f;
    CHECK_CLOSE(v1.value, v2.value, tol);
  }
  std::string path = "/tmp/rune_adaptive_fp32.rune";
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
  {
    std::ofstream o("/tmp/rune_adaptive_bogus.rune", std::ios::binary);
    std::string header = "{\"format\":1,\"arch\":\"RUNE-04\",\"arch_version\":\"9.9.9\"}";
    o.write("RUNE", 4);
    uint32_t hlen = header.size();
    o.write(reinterpret_cast<const char*>(&hlen), 4);
    o.write(header.data(), header.size());
  }
  RuneFile rej;
  CHECK(!loadRuneFile("/tmp/rune_adaptive_bogus.rune", rej, err));
}

void testAdaptiveParamAccounting() {
  AdaptiveBuildSpec spec;
  AdaptiveModel model;
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
  CHECK(model.cheapParameterCount() < model.parameterCount());
  std::vector<float> flat(n, 0.1f);
  CHECK(model.setTensors(names, flat));
  std::vector<std::string> wrong = names;
  wrong[0] = "bogus";
  CHECK(!model.setTensors(wrong, flat));
}

void testUncertaintyBounds() {
  AdaptiveBuildSpec spec;
  spec.hasUncertainty = true;
  spec.hasStabilityHead = true;
  AdaptiveModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  CHECK(std::string(model.archId()) == "RUNE-05");
  CHECK(std::string(model.archVersion()) == "0.5.0");
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
  };
  VarEmbeddings emb;
  makeAdaptiveEmbeddings(emb, 32);
  AdaptiveEvaluator ev;
  CHECK(ev.configure(&emb, &model, err));
  for (const char* fen : fens) {
    Board b(fen);
    AdaptiveEvalResult r = ev.evaluateSearch(b, SearchRoute::Full, RoutingThresholds(), false);
    CHECK(r.uncertainty >= 0.0f && r.uncertainty <= 1.0f);
    CHECK(r.stability >= 0.0f);
    AdaptiveEvalResult r2 = ev.evaluateSearch(b, SearchRoute::Full, RoutingThresholds(), false);
    CHECK(r2.refined == r.refined);
    CHECK_CLOSE(r2.uncertainty, r.uncertainty, 0.0);
  }
  AdaptiveBuildSpec plain;
  AdaptiveModel m2;
  CHECK(m2.configure(plain, err));
  CHECK(std::string(m2.archId()) == "RUNE-04");
  CHECK(m2.cheapParameterCount() < model.cheapParameterCount());
}

void testSearchRoutingModes() {
  SearchRoute r;
  CHECK(searchRouteFromString("difficulty", r) && r == SearchRoute::Difficulty);
  CHECK(searchRouteFromString("uncertainty", r) && r == SearchRoute::Uncertainty);
  CHECK(searchRouteFromString("both", r) && r == SearchRoute::Both);
  CHECK(searchRouteFromString("full", r) && r == SearchRoute::Full);
  CHECK(!searchRouteFromString("bogus", r));
  CHECK(std::string(searchRouteName(SearchRoute::Both)) == "both");
  AdaptiveBuildSpec spec;
  spec.hasUncertainty = true;
  AdaptiveModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  RoutingThresholds t;
  t.diffT = 0.5f;
  t.uncT = 0.5f;
  t.stabT = 0.5f;
  CHECK(model.routeSearch(SearchRoute::Difficulty, 0.2f, 0.9f, 0.9f, false, t) == false);
  CHECK(model.routeSearch(SearchRoute::Uncertainty, 0.2f, 0.9f, 0.0f, false, t) == true);
  CHECK(model.routeSearch(SearchRoute::Both, 0.2f, 0.9f, 0.0f, false, t) == true);
  CHECK(model.routeSearch(SearchRoute::Full, 0.2f, 0.1f, 0.9f, false, t) == true);
  CHECK(model.routeSearch(SearchRoute::Full, 0.2f, 0.1f, 0.1f, false, t) == false);
  RoutingThresholds h = t;
  h.hasTLow = true;
  h.tLow = 0.3f;
  CHECK(model.routeSearch(SearchRoute::Difficulty, 0.4f, 0.0f, 0.0f, true, h) == true);
  CHECK(model.routeSearch(SearchRoute::Difficulty, 0.4f, 0.0f, 0.0f, false, h) == false);
}

void testUncertaintyModelIO() {
  VarEmbeddings emb;
  makeAdaptiveEmbeddings(emb, 32);
  AdaptiveBuildSpec spec;
  spec.hasUncertainty = true;
  spec.hasStabilityHead = true;
  AdaptiveModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  ModelSpec mspec = model.spec();
  std::string path = "/tmp/rune_search_fp32.rune";
  CHECK(saveAdaptiveRuneFile(path, mspec, emb, model, "fp32", err));
  RuneFile loaded;
  CHECK(loadRuneFile(path, loaded, err));
  CHECK(loaded.isUncertainty);
  CHECK(loaded.spec.arch == "RUNE-05");
  CHECK(loaded.spec.archVersion == "0.5.0");
  CHECK(loaded.spec.hasUncertainty == true);
  Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  AdaptiveEvaluator ref;
  CHECK(ref.configure(&emb, &model, err));
  AdaptiveEvaluator got;
  AdaptiveModel* lm = static_cast<AdaptiveModel*>(loaded.arch.get());
  CHECK(got.configure(&loaded.varEmbeddings, lm, err));
  AdaptiveEvalResult v1 = ref.evaluateSearch(b, SearchRoute::Both, RoutingThresholds(), false);
  AdaptiveEvalResult v2 = got.evaluateSearch(b, SearchRoute::Both, RoutingThresholds(), false);
  CHECK_CLOSE(v1.value, v2.value, 1e-6f);
  CHECK_CLOSE(v1.uncertainty, v2.uncertainty, 1e-6f);
  CHECK(v1.refined == v2.refined);
  {
    std::ifstream f(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    bytes[bytes.size() - 1] ^= 0xFF;
    std::ofstream o(path + std::string(".bad"), std::ios::binary);
    o.write(bytes.data(), bytes.size());
  }
  RuneFile bad;
  CHECK(!loadRuneFile(path + std::string(".bad"), bad, err));
  CHECK(err == "checksum mismatch");
}

void testUncertaintyParamAccounting() {
  AdaptiveBuildSpec spec;
  spec.hasUncertainty = true;
  AdaptiveModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  model.getTensors(names, shapes, data);
  CHECK(names[names.size() - 4] == "uw");
  CHECK(names[names.size() - 1] == "sb");
  size_t n = 0;
  for (auto& sh : shapes) {
    size_t c = 1;
    for (int s : sh) c *= s;
    n += c;
  }
  CHECK(model.parameterCount() == n);
  AdaptiveBuildSpec plain;
  AdaptiveModel m2;
  CHECK(m2.configure(plain, err));
  CHECK(model.parameterCount() == m2.parameterCount() + 2 * (256 + 1));
  std::vector<float> flat(n, 0.1f);
  CHECK(model.setTensors(names, flat));
  CHECK(!model.setTensors(names, std::vector<float>(n - 1, 0.1f)));
}

void testAdaptiveStudentWidths() {
  AdaptiveBuildSpec spec;
  spec.dim = 16;
  spec.cheapHidden = 16;
  spec.refH1 = 64;
  spec.refH2 = 16;
  AdaptiveModel model;
  std::string err;
  CHECK(model.configure(spec, err));
  ModelSpec ms = model.spec();
  CHECK(ms.cheapHidden == 16 && ms.refH1 == 64 && ms.refH2 == 16);
  AdaptiveBuildSpec bad = spec;
  bad.refH2 = 2;
  CHECK(!model.configure(bad, err));
  VarEmbeddings emb;
  VarWidths w;
  for (int g = 0; g < 8; ++g) w.w[g] = 16;
  emb.configure(w);
  emb.init(9);
  std::string path = "/tmp/rune_adaptive_s2_fp32.rune";
  CHECK(saveAdaptiveRuneFile(path, ms, emb, model, "fp32", err));
  RuneFile loaded;
  CHECK(loadRuneFile(path, loaded, err));
  CHECK(loaded.spec.cheapHidden == 16);
  AdaptiveEvaluator ref;
  CHECK(ref.configure(&emb, &model, err));
  AdaptiveEvaluator got;
  AdaptiveModel* lm = static_cast<AdaptiveModel*>(loaded.arch.get());
  CHECK(got.configure(&loaded.varEmbeddings, lm, err));
  Board b("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
  AdaptiveEvalResult v1 = ref.evaluateBoard(b, AdaptiveMode::Always, 0.0f, true, false);
  AdaptiveEvalResult v2 = got.evaluateBoard(b, AdaptiveMode::Always, 0.0f, true, false);
  CHECK_CLOSE(v1.value, v2.value, 1e-6f);
}
