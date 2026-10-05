#include <cstdio>
#include <string>
#include <vector>
#include "core/board/board.h"
#include "core/engine/eval_cache.h"
#include "core/engine/eval_contract.h"
#include "core/engine/search.h"
#include "core/inference/evaluator.h"
#include "core/model_io/model_io.h"
int main(int argc, char** argv) {
  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  std::string model;
  int depth = 3;
  std::string mode = "L0";
  for (int i = 1; i + 1 < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fen") fen = argv[i + 1];
    if (a == "--model") model = argv[i + 1];
    if (a == "--depth") depth = std::atoi(argv[i + 1]);
    if (a == "--lazy") mode = argv[i + 1];
  }
  rune::Board board(fen);
  rune::eng::LazyConfig cfg;
  if (mode == "L1") cfg.mode = rune::eng::LazyMode::L1;
  if (mode == "L2") cfg.mode = rune::eng::LazyMode::L2;
  long evals = 0;
  rune::RuneFile rf;
  std::string err;
  bool haveModel = !model.empty() && rune::loadRuneFile(model, rf, err) && !rf.isFlex && !rf.isDense && !rf.isAdaptive && !rf.isUncertainty;
  if (!model.empty() && !haveModel && !err.empty()) {
    std::printf("model note %s\n", err.c_str());
  }
  rune::Evaluator* evp = nullptr;
  rune::Evaluator ev(&rf.embeddings, rf.arch.get());
  if (haveModel) evp = &ev;
  rune::eng::EvalFn evfn = [&](rune::Board& b) -> float {
    ++evals;
    if (evp) return evp->evaluateBoard(b).value;
    std::vector<rune::Move> lm;
    b.generateLegalMoves(lm);
    return (float)((int)lm.size() % 7) * 0.05f;
  };
  rune::eng::SearchStats st;
  float s = rune::eng::searchRoot(board, depth, evfn, cfg, st, -1e9f, 1e9f);
  std::printf("score %.4f nodes %ld evals %ld nps %.0f cutoffs %ld root %ld pv %ld cut %ld leaf %ld\n", s, st.nodes, evals, st.nps(), st.cutoffs, st.rootCount, st.pvCount, st.cutCount, st.leafCount);
  std::printf("engine_score %d\n", rune::eng::engineScore(s));
  if (st.hasMove) std::printf("move %d %d\n", (int)st.rootMove.from, (int)st.rootMove.to);
  rune::eng::EvalCache cache(model.empty() ? "stub" : model, mode, 64);
  cache.put(fen, s);
  float v = 0.0f;
  bool hit = cache.get(fen, v);
  std::printf("cache_hit %d value %.4f rate %.3f\n", (int)hit, v, cache.hitRate());
  return 0;
}
