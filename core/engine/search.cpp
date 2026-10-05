#include "core/engine/search.h"
#include <chrono>
namespace rune {
namespace eng {
double SearchStats::nps() const {
  return seconds > 0 ? (double)nodes / seconds : 0.0;
}
namespace {
float negamax(Board& b, int depth, int ply, float alpha, float beta, bool root, EvalFn& ev, const LazyConfig& cfg, SearchStats& stats) {
  (void)cfg;
  if (depth <= 0) {
    ++stats.nodes;
    ++stats.evals;
    ++stats.leafCount;
    return ev(b);
  }
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  if (moves.empty()) {
    ++stats.nodes;
    ++stats.evals;
    ++stats.leafCount;
    return ev(b);
  }
  float best = -1e9f;
  for (const Move& m : moves) {
    Board child = b;
    if (!child.makeMove(m)) continue;
    float v = -negamax(child, depth - 1, ply + 1, -beta, -alpha, false, ev, cfg, stats);
    ++stats.nodes;
    if (root) ++stats.rootCount;
    else if (beta - alpha > 1.0f) ++stats.pvCount;
    else if (ply % 2 == 0) ++stats.cutCount;
    else ++stats.leafCount;
    if (v > best) best = v;
    if (v > alpha) alpha = v;
    if (alpha >= beta) {
      ++stats.cutoffs;
      break;
    }
  }
  return best;
}
}
float searchRoot(Board& board, int depth, EvalFn ev, const LazyConfig& cfg, SearchStats& stats, float alpha, float beta) {
  auto t0 = std::chrono::high_resolution_clock::now();
  stats = SearchStats();
  std::vector<Move> moves;
  board.generateLegalMoves(moves);
  float best = -1e9f;
  bool first = true;
  for (const Move& m : moves) {
    Board child = board;
    if (!child.makeMove(m)) continue;
    float v = -negamax(child, depth - 1, 1, -beta, -alpha, false, ev, cfg, stats);
    ++stats.nodes;
    ++stats.rootCount;
    if (first || v > best) {
      best = v;
      stats.rootMove = m;
      stats.hasMove = true;
      first = false;
    }
    if (v > alpha) alpha = v;
    if (alpha >= beta) {
      ++stats.cutoffs;
      break;
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  stats.seconds = std::chrono::duration<double>(t1 - t0).count();
  stats.rootScore = first ? 0.0f : best;
  return stats.rootScore;
}
}
}
