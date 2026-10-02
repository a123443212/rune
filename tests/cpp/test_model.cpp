#include <string>
#include <vector>

#include "core/accumulators/grouped_accumulator.h"
#include "core/architectures/attention/attention.h"
#include "core/architectures/mlp/mlp.h"
#include "core/board/board.h"
#include "core/features/feature_set.h"
#include "core/inference/evaluator.h"
#include "core/model_io/model_io.h"
#include "tests/cpp/test_framework.h"

using namespace rune;

void testEvaluator() {
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
  ev.evaluate();
  CHECK(b.makeMove(moves[0]));
  GroupedFeatureSet::extract(b, after);
  GroupedFeatureSet::diffFeatures(before, after, added, removed);
  ev.updateIncremental(added, removed);
  EvalResult rInc = ev.evaluate();
  EvalResult rFull = ev.evaluateBoard(b);
  CHECK_CLOSE(rInc.value, rFull.value, 1e-5);
}

void testModelIO() {
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
