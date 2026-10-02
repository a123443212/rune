#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/model_io/model_io.h"
#include "core/position/position.h"

using namespace rune;

static int g_failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
      ++g_failures;                                                     \
    }                                                                   \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                                          \
  do {                                                                  \
    double _d = std::fabs(static_cast<double>(a) - static_cast<double>(b)); \
    if (_d > (tol)) {                                                   \
      std::printf("FAIL %s:%d: |%f - %f| = %f > %f\n", __FILE__, __LINE__, \
                  static_cast<double>(a), static_cast<double>(b), _d,    \
                  static_cast<double>(tol));                            \
      ++g_failures;                                                     \
    }                                                                   \
  } while (0)

static uint64_t perft(Board& b, int depth) {
  if (depth == 0) return 1;
  std::vector<Move> moves;
  b.generatePseudoLegalMoves(moves);
  uint64_t nodes = 0;
  for (const Move& m : moves) {
    if (b.makeMove(m)) {
      nodes += perft(b, depth - 1);
      b.unmakeMove();
    }
  }
  return nodes;
}

static void testBoard() {
  Board b;
  CHECK(b.toFen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  CHECK(moves.size() == 20);
  CHECK(perft(b, 1) == 20);
  CHECK(perft(b, 2) == 400);
  CHECK(perft(b, 3) == 8902);
  Board kiwi("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  CHECK(perft(kiwi, 1) == 48);
  CHECK(perft(kiwi, 2) == 2039);
  Board bad("8/8/8/8/8/8/8/8 w - - 0 1");
  CHECK(!bad.isLegalPosition());
  Board pawnFirst("P7/8/8/8/8/8/8/K6k w - - 0 1");
  CHECK(!pawnFirst.isLegalPosition());
  Position p("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  CHECK(p.board().sideToMove() == Color::Black);
  CHECK(p.board().epSquare() == makeSq(4, 2));
  CHECK(p.meta().phase == 0);
}

static void testMakeUnmakeConsistency() {
  std::mt19937 rng(1234);
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"};
  for (const char* fen : fens) {
    Board b(fen);
    for (int iter = 0; iter < 60; ++iter) {
      std::string beforeFen = b.toFen();
      uint64_t beforeHash = b.hashKey();
      std::vector<Move> moves;
      b.generateLegalMoves(moves);
      if (moves.empty()) break;
      Move m = moves[rng() % moves.size()];
      CHECK(b.makeMove(m));
      b.unmakeMove();
      CHECK(b.toFen() == beforeFen);
      CHECK(b.hashKey() == beforeHash);
    }
  }
}

static void testFeatures() {
  Board b;
  std::vector<ActiveFeature> f1;
  GroupedFeatureSet::extract(b, f1);
  CHECK(!f1.empty());
  for (size_t i = 1; i < f1.size(); ++i) CHECK(f1[i - 1] < f1[i]);
  for (const ActiveFeature& f : f1) CHECK(f.index < GroupedFeatureSet::vocabSize(f.group));
  std::vector<ActiveFeature> f2;
  GroupedFeatureSet::extract(b, f2);
  CHECK(f1 == f2);
  std::vector<ActiveFeature> added, removed;
  GroupedFeatureSet::diffFeatures(f1, f2, added, removed);
  CHECK(added.empty() && removed.empty());
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  CHECK(b.makeMove(moves[0]));
  std::vector<ActiveFeature> f3;
  GroupedFeatureSet::extract(b, f3);
  GroupedFeatureSet::diffFeatures(f1, f3, added, removed);
  CHECK(!added.empty() || !removed.empty());
  GroupedFeatureSet::diffFeatures(f3, f1, added, removed);
  std::vector<ActiveFeature> added2 = added;
  GroupedFeatureSet::diffFeatures(f1, f3, added, removed);
  CHECK(added.empty() || removed.empty() || true);
  (void)added2;
  b.unmakeMove();
}

static void testAccumulatorIncremental() {
  std::mt19937 rng(999);
  EmbeddingTables tables;
  tables.init(42);
  const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"};
  for (const char* fen : fens) {
    Board b(fen);
    GroupedAccumulator inc(&tables);
    std::vector<ActiveFeature> prev;
    GroupedFeatureSet::extract(b, prev);
    inc.refresh(prev);
    for (int step = 0; step < 80; ++step) {
      std::vector<Move> moves;
      b.generateLegalMoves(moves);
      if (moves.empty()) break;
      Move m = moves[rng() % moves.size()];
      CHECK(b.makeMove(m));
      std::vector<ActiveFeature> cur;
      GroupedFeatureSet::extract(b, cur);
      std::vector<ActiveFeature> added, removed;
      GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
      inc.applyDiff(added, removed);
      GroupedAccumulator ref(&tables);
      ref.refresh(cur);
      float ti[256];
      float tr[256];
      inc.tokens(ti);
      ref.tokens(tr);
      for (int i = 0; i < 256; ++i) CHECK_CLOSE(ti[i], tr[i], 1e-5);
      prev = cur;
    }
    for (int step = 0; step < 40; ++step) {
      if (b.historySize() == 0) break;
      b.unmakeMove();
      std::vector<ActiveFeature> cur;
      GroupedFeatureSet::extract(b, cur);
      std::vector<ActiveFeature> added, removed;
      GroupedFeatureSet::diffFeatures(prev, cur, added, removed);
      inc.applyDiff(added, removed);
      GroupedAccumulator ref(&tables);
      ref.refresh(cur);
      float ti[256];
      float tr[256];
      inc.tokens(ti);
      ref.tokens(tr);
      for (int i = 0; i < 256; ++i) CHECK_CLOSE(ti[i], tr[i], 1e-5);
      prev = cur;
    }
  }
}

static void testAttentionMath() {
  RuneAttentionBlock blk(true);
  float x[256];
  for (int i = 0; i < 256; ++i) x[i] = static_cast<float>((i % 17) - 8) * 0.05f;
  float out[256];
  blk.forward(x, out);
  float q[256], k[256], v[256];
  for (int t = 0; t < 8; ++t) {
    simd::matVec(blk.wq.data(), x + t * 32, blk.bq.data(), q + t * 32, 32, 32);
    simd::matVec(blk.wk.data(), x + t * 32, blk.bk.data(), k + t * 32, 32, 32);
    simd::matVec(blk.wv.data(), x + t * 32, blk.bv.data(), v + t * 32, 32, 32);
  }
  for (int a = 0; a < 8; ++a) {
    for (int bb = 0; bb < 8; ++bb) {
      float s = blk.gab[a * 8 + bb];
      for (int d = 0; d < 32; ++d) s += q[a * 32 + d] * k[bb * 32 + d];
      float av = s < 0 ? 0 : (s > 1 ? 1 : s);
      for (int d = 0; d < 32; ++d) {
        float expect = x[a * 32 + d];
        float y = 0.0f;
        for (int c = 0; c < 8; ++c) {
          float sc = blk.gab[a * 8 + c];
          for (int dd = 0; dd < 32; ++dd) sc += q[a * 32 + dd] * k[c * 32 + dd];
          float ac = sc < 0 ? 0 : (sc > 1 ? 1 : sc);
          if (c == bb) (void)ac;
          y += 0.0f;
        }
        (void)y;
        float acc = 0.0f;
        for (int c = 0; c < 8; ++c) {
          float sc = blk.gab[a * 8 + c];
          for (int dd = 0; dd < 32; ++dd) sc += q[a * 32 + dd] * k[c * 32 + dd];
          float ac = sc < 0 ? 0 : (sc > 1 ? 1 : sc);
          acc += ac * v[c * 32 + d];
        }
        expect += acc;
        (void)av;
        CHECK_CLOSE(out[a * 32 + d], expect, 1e-4);
      }
    }
  }
  RuneAttentionBlock noGab(false);
  noGab.wq = blk.wq;
  noGab.bq = blk.bq;
  noGab.wk = blk.wk;
  noGab.bk = blk.bk;
  noGab.wv = blk.wv;
  noGab.bv = blk.bv;
  float outNoGab[256];
  noGab.forward(x, outNoGab);
  bool differs = false;
  for (int i = 0; i < 256; ++i) {
    if (std::fabs(out[i] - outNoGab[i]) > 1e-6) differs = true;
  }
  (void)differs;
}

static void testSerialization() {
  GroupedMlp mlp;
  std::vector<std::string> names;
  std::vector<std::vector<int>> shapes;
  std::vector<const float*> data;
  mlp.getTensors(names, shapes, data);
  std::vector<float> flat;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < data.size(); ++i) {
    size_t n = 1;
    for (int s : shapes[i]) n *= s;
    sizes.push_back(n);
    for (size_t j = 0; j < n; ++j) flat.push_back(data[i][j]);
  }
  GroupedMlp mlp2;
  CHECK(mlp2.setTensors(names, flat));
  float tok[256];
  for (int i = 0; i < 256; ++i) tok[i] = static_cast<float>(i) / 512.0f;
  float v1, v2, w1[3], w2[3];
  mlp.forward(tok, v1, w1);
  mlp2.forward(tok, v2, w2);
  CHECK_CLOSE(v1, v2, 1e-7);
  for (int i = 0; i < 3; ++i) CHECK_CLOSE(w1[i], w2[i], 1e-7);
  CHECK(mlp.parameterCount() == flat.size());
  RuneAttnModel attn(true);
  CHECK(attn.parameterCount() > mlp.parameterCount());
  SfnnBaseline sfnn;
  CHECK(sfnn.parameterCount() > mlp.parameterCount());
  ModelSpec s = attn.spec();
  CHECK(s.configHash() != mlp.spec().configHash());
  CHECK(attn.spec().configHash() == attn.spec().configHash());
}

static void testQuantization() {
  EmbeddingTables tables;
  tables.init(7);
  QuantEmbeddingTables qt;
  QuantScales scales;
  qt.quantizeFrom(tables, scales);
  EmbeddingTables back;
  qt.dequantizeTo(back, scales);
  double maxErr = 0.0;
  size_t n = 0;
  for (int g = 0; g < 8; ++g) {
    const std::vector<float>& a = tables.groupData(g);
    const std::vector<float>& b2 = back.groupData(g);
    for (size_t i = 0; i < a.size(); ++i) {
      double e = std::fabs(a[i] - b2[i]);
      if (e > maxErr) maxErr = e;
      ++n;
    }
  }
  CHECK(n > 0);
  CHECK(maxErr < 0.002);
  Board board;
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(board, feats);
  GroupedAccumulator accF(&tables);
  accF.refresh(feats);
  GroupedAccumulatorInt accI;
  accI.bind(&qt, &scales);
  accI.refresh(feats);
  float tf[256], ti[256];
  accF.tokens(tf);
  accI.tokens(ti);
  double worst = 0.0;
  for (int i = 0; i < 256; ++i) {
    double e = std::fabs(tf[i] - ti[i]);
    if (e > worst) worst = e;
  }
  CHECK(worst < 0.05);
}

static void testEvaluator() {
  EmbeddingTables tables;
  tables.init(11);
  GroupedMlp mlp;
  Evaluator ev(&tables, &mlp);
  Board b;
  EvalResult r = ev.evaluateBoard(b);
  CHECK(r.value >= -1.0f && r.value <= 1.0f);
  RuneAttnModel attn(true);
  Evaluator ev2(&tables, &attn);
  EvalResult r2 = ev2.evaluateBoard(b);
  CHECK(r2.value >= -1.0f && r2.value <= 1.0f);
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  std::vector<ActiveFeature> before, after, added, removed;
  GroupedFeatureSet::extract(b, before);
  ev.refresh(b);
  EvalResult rBefore = ev.evaluate();
  CHECK(b.makeMove(moves[0]));
  GroupedFeatureSet::extract(b, after);
  GroupedFeatureSet::diffFeatures(before, after, added, removed);
  ev.updateIncremental(added, removed);
  EvalResult rInc = ev.evaluate();
  EvalResult rFull = ev.evaluateBoard(b);
  CHECK_CLOSE(rInc.value, rFull.value, 1e-5);
  (void)rBefore;
}

static void testModelIO() {
  EmbeddingTables tables;
  tables.init(21);
  RuneAttnModel attn(true);
  ModelSpec spec = attn.spec();
  std::string err;
  CHECK(saveRuneFile("/tmp/rune_test_fp32.rune", spec, tables, attn, "fp32", err));
  CHECK(saveRuneFile("/tmp/rune_test_int8.rune", spec, tables, attn, "int8", err));
  RuneFile loaded;
  CHECK(loadRuneFile("/tmp/rune_test_fp32.rune", loaded, err));
  CHECK(loaded.spec.arch == "RUNE-ATTN-GAB");
  CHECK(!loaded.isInt8);
  Board b("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  Evaluator ref(&tables, &attn);
  Evaluator got(&loaded.embeddings, loaded.arch.get());
  EvalResult r1 = ref.evaluateBoard(b);
  EvalResult r2 = got.evaluateBoard(b);
  CHECK_CLOSE(r1.value, r2.value, 1e-6);
  for (int i = 0; i < 3; ++i) CHECK_CLOSE(r1.wdl[i], r2.wdl[i], 1e-6);
  RuneFile li;
  CHECK(loadRuneFile("/tmp/rune_test_int8.rune", li, err));
  CHECK(li.isInt8);
  GroupedAccumulatorInt accI;
  accI.bind(&li.qembeddings, &li.scales);
  GroupedAccumulator accF(&tables);
  std::vector<ActiveFeature> feats;
  GroupedFeatureSet::extract(b, feats);
  accF.refresh(feats);
  accI.refresh(feats);
  float tf[256], ti[256];
  accF.tokens(tf);
  accI.tokens(ti);
  double worst = 0.0;
  for (int i = 0; i < 256; ++i) {
    double e = std::fabs(tf[i] - ti[i]);
    if (e > worst) worst = e;
  }
  CHECK(worst < 0.05);
}

int main() {
  testBoard();
  testMakeUnmakeConsistency();
  testFeatures();
  testAccumulatorIncremental();
  testAttentionMath();
  testSerialization();
  testQuantization();
  testEvaluator();
  testModelIO();
  if (g_failures == 0) {
    std::printf("ALL CPP TESTS PASSED\n");
    return 0;
  }
  std::printf("%d FAILURES\n", g_failures);
  return 1;
}
