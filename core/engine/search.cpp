#include "core/engine/search.h"
#include <chrono>
namespace rune {
namespace eng {
double SearchStats::nps() const {
  return seconds > 0 ? (double)nodes / seconds : 0.0;
}
namespace {
constexpr float kMateScore = 10000.0f;
float terminalScore(Board& b, int ply) {
  if (b.inCheck(b.sideToMove())) {
    return -(kMateScore - static_cast<float>(ply));
  }
  return 0.0f;
}
bool isDrawn(const Board& b, const std::vector<uint64_t>& hist, uint64_t key) {
  if (b.halfmoveClock() >= 100) {
    return true;
  }
  int reps = 0;
  for (uint64_t h : hist) {
    if (h == key) ++reps;
  }
  return reps >= 2;
}
float negamax(Board& b, int depth, int ply, float alpha, float beta, bool root, EvalFn& ev, const LazyConfig& cfg, SearchStats& stats, std::vector<uint64_t>& hist) {
  (void)cfg;
  uint64_t key = b.hashKey();
  if (isDrawn(b, hist, key)) {
    ++stats.nodes;
    return 0.0f;
  }
  hist.push_back(key);
  float out;
  if (depth <= 0) {
    ++stats.nodes;
    ++stats.evals;
    ++stats.leafCount;
    out = ev(b);
  } else {
    std::vector<Move> moves;
    b.generateLegalMoves(moves);
    if (moves.empty()) {
      ++stats.nodes;
      ++stats.evals;
      ++stats.leafCount;
      out = terminalScore(b, ply);
    } else {
      float best = -1e9f;
      for (const Move& m : moves) {
        if (!b.makeMove(m)) continue;
        float v = -negamax(b, depth - 1, ply + 1, -beta, -alpha, false, ev, cfg, stats, hist);
        b.unmakeMove();
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
      out = best;
    }
  }
  hist.pop_back();
  return out;
}
}
float searchRoot(Board& board, int depth, EvalFn ev, const LazyConfig& cfg, SearchStats& stats, float alpha, float beta) {
  auto t0 = std::chrono::high_resolution_clock::now();
  stats = SearchStats();
  std::vector<Move> moves;
  board.generateLegalMoves(moves);
  float best = -1e9f;
  bool first = true;
  if (moves.empty()) {
    stats.nodes += 1;
    stats.evals += 1;
    stats.leafCount += 1;
    stats.rootScore = terminalScore(board, 0);
    return stats.rootScore;
  }
  std::vector<uint64_t> hist;
  hist.push_back(board.hashKey());
  for (const Move& m : moves) {
    if (!board.makeMove(m)) continue;
    float v = -negamax(board, depth - 1, 1, -beta, -alpha, false, ev, cfg, stats, hist);
    board.unmakeMove();
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
