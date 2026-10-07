#include "core/engine/search.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
namespace rune {
namespace eng {
double SearchStats::nps() const {
  return seconds > 0 ? (double)nodes / seconds : 0.0;
}
namespace {
constexpr float kMateScore = 10000.0f;
int pieceValue(PieceType t) {
  if (t == PieceType::Pawn) return 100;
  if (t == PieceType::Knight) return 320;
  if (t == PieceType::Bishop) return 330;
  if (t == PieceType::Rook) return 500;
  if (t == PieceType::Queen) return 900;
  return 0;
}
int moveScore(const Board& b, const Move& m) {
  Piece moving = b.at(m.from);
  Piece victim = b.at(m.to);
  if (victim.empty() && moving.type == PieceType::Pawn && m.to == b.epSquare()) {
    victim = Piece{PieceType::Pawn, opposite(moving.color)};
  }
  int score = 0;
  if (!victim.empty()) {
    score = pieceValue(victim.type) * 16 - pieceValue(moving.type);
  }
  if (m.promotion != PieceType::None) {
    score += pieceValue(m.promotion);
  }
  return score;
}
}
void orderMoves(Board& b, std::vector<Move>& moves) {
  std::stable_sort(moves.begin(), moves.end(), [&b](const Move& a, const Move& c) {
    return moveScore(b, a) > moveScore(b, c);
  });
}
namespace {
float terminalScore(Board& b, int ply) {
  if (b.inCheck(b.sideToMove())) {
    return -(kMateScore - static_cast<float>(ply));
  }
  return 0.0f;
}
struct TtEntry {
  float score = 0.0f;
  int depth = 0;
  uint8_t flag = 0;
};
const size_t kTtCap = 1 << 20;
const int kMaxQPly = 64;
using TtMap = std::unordered_map<uint64_t, TtEntry>;
bool ttProbe(TtMap& tt, uint64_t key, int depth, float& alpha, float& beta, SearchStats& stats, float& out) {
  auto it = tt.find(key);
  if (it == tt.end() || it->second.depth < depth) {
    return false;
  }
  ++stats.ttHits;
  const TtEntry& e = it->second;
  if (e.flag == 0) {
    out = e.score;
    return true;
  }
  if (e.flag == 1 && e.score > alpha) {
    alpha = e.score;
  }
  if (e.flag == 2 && e.score < beta) {
    beta = e.score;
  }
  if (alpha >= beta) {
    out = e.score;
    return true;
  }
  return false;
}
void ttStore(TtMap& tt, uint64_t key, int depth, float score, float alpha, float beta) {
  if (tt.size() >= kTtCap) {
    tt.clear();
  }
  uint8_t flag = 0;
  if (score <= alpha) {
    flag = 2;
  } else if (score >= beta) {
    flag = 1;
  }
  auto it = tt.find(key);
  if (it != tt.end() && it->second.depth > depth) {
    return;
  }
  tt[key] = TtEntry{score, depth, flag};
}
bool shouldRefine(float cheap, float alpha, float beta, const LazyConfig& cfg) {
  if (cfg.mode == LazyMode::L0) {
    return true;
  }
  if (!(cheap == cheap)) {
    return true;
  }
  if (cfg.mode == LazyMode::L1) {
    return cheap >= cfg.threshold;
  }
  float lo = alpha - cfg.margin;
  float hi = beta + cfg.margin;
  if (cheap <= lo || cheap >= hi) {
    return false;
  }
  return cheap >= cfg.threshold;
}
float leafValue(Board& b, float alpha, float beta, EvalFn& ev, EvalFn* cheap, const LazyConfig& cfg, SearchStats& stats) {
  if (!cheap) {
    ++stats.evals;
    return ev(b);
  }
  float c = (*cheap)(b);
  if (shouldRefine(c, alpha, beta, cfg)) {
    ++stats.refined;
    ++stats.evals;
    return ev(b);
  }
  return c;
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
float qsearch(Board& b, float stand, float alpha, float beta, int ply, int qply, EvalFn& ev, EvalFn* cheap, const LazyConfig& cfg, SearchStats& stats, std::vector<uint64_t>& hist);
float negamax(Board& b, int depth, int ply, float alpha, float beta, bool root, EvalFn& ev, EvalFn* cheap, const LazyConfig& cfg, SearchStats& stats, std::vector<uint64_t>& hist, TtMap& tt) {
  uint64_t key = b.hashKey();
  if (isDrawn(b, hist, key)) {
    ++stats.nodes;
    return 0.0f;
  }
  hist.push_back(key);
  float betaMut = beta;
  float probed = 0.0f;
  if (ttProbe(tt, key, depth, alpha, betaMut, stats, probed)) {
    hist.pop_back();
    ++stats.nodes;
    return probed;
  }
  float origAlpha = alpha;
  float origBeta = betaMut;
  float out;
  if (depth <= 0) {
    float stand = leafValue(b, alpha, betaMut, ev, cheap, cfg, stats);
    out = qsearch(b, stand, alpha, betaMut, ply, 0, ev, cheap, cfg, stats, hist);
  } else {
    std::vector<Move> moves;
    b.generateLegalMoves(moves);
    orderMoves(b, moves);
    if (moves.empty()) {
      ++stats.nodes;
      ++stats.evals;
      ++stats.leafCount;
      out = terminalScore(b, ply);
    } else {
      float best = -1e9f;
      for (const Move& m : moves) {
        if (!b.makeMove(m)) continue;
        float v = -negamax(b, depth - 1, ply + 1, -betaMut, -alpha, false, ev, cheap, cfg, stats, hist, tt);
        b.unmakeMove();
        ++stats.nodes;
        if (root) ++stats.rootCount;
        else if (betaMut - alpha > 1.0f) ++stats.pvCount;
        else if (ply % 2 == 0) ++stats.cutCount;
        else ++stats.leafCount;
        if (v > best) best = v;
        if (v > alpha) alpha = v;
        if (alpha >= betaMut) {
          ++stats.cutoffs;
          break;
        }
      }
      out = best;
    }
  }
  ttStore(tt, key, depth, out, origAlpha, origBeta);
  hist.pop_back();
  return out;
}
bool isCaptureMove(const Board& b, const Move& m) {
  if (!b.at(m.to).empty()) {
    return true;
  }
  Piece moving = b.at(m.from);
  return moving.type == PieceType::Pawn && m.to == b.epSquare();
}
float qsearch(Board& b, float stand, float alpha, float beta, int ply, int qply, EvalFn& ev, EvalFn* cheap, const LazyConfig& cfg, SearchStats& stats, std::vector<uint64_t>& hist) {
  ++stats.nodes;
  ++stats.leafCount;
  ++stats.qnodes;
  if (qply >= kMaxQPly) {
    return stand;
  }
  uint64_t key = b.hashKey();
  hist.push_back(key);
  int reps = 0;
  for (uint64_t h : hist) {
    if (h == key) ++reps;
  }
  if (reps >= 3) {
    hist.pop_back();
    return 0.0f;
  }
  std::vector<Move> moves;
  b.generateLegalMoves(moves);
  orderMoves(b, moves);
  float best = stand;
  if (best > alpha) {
    alpha = best;
  }
  if (alpha >= beta) {
    hist.pop_back();
    return best;
  }
  for (const Move& m : moves) {
    if (!isCaptureMove(b, m)) {
      continue;
    }
    if (!b.makeMove(m)) continue;
    float childStand = leafValue(b, -beta, -alpha, ev, cheap, cfg, stats);
    float v = -qsearch(b, childStand, -beta, -alpha, ply + 1, qply + 1, ev, cheap, cfg, stats, hist);
    b.unmakeMove();
    ++stats.nodes;
    ++stats.leafCount;
    ++stats.qnodes;
    if (v > best) {
      best = v;
    }
    if (v > alpha) {
      alpha = v;
    }
    if (alpha >= beta) {
      break;
    }
  }
  hist.pop_back();
  return best;
}
}
float searchRoot(Board& board, int depth, EvalFn ev, const LazyConfig& cfg, SearchStats& stats, float alpha, float beta) {
  auto t0 = std::chrono::high_resolution_clock::now();
  stats = SearchStats();
  std::vector<Move> moves;
  board.generateLegalMoves(moves);
  orderMoves(board, moves);
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
  TtMap tt;
  for (const Move& m : moves) {
    if (!board.makeMove(m)) continue;
    float v = -negamax(board, depth - 1, 1, -beta, -alpha, false, ev, nullptr, cfg, stats, hist, tt);
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
float searchRootLazy(Board& board, int depth, EvalFn cheap, EvalFn full, const LazyConfig& cfg, SearchStats& stats, float alpha, float beta) {
  auto t0 = std::chrono::high_resolution_clock::now();
  stats = SearchStats();
  std::vector<Move> moves;
  board.generateLegalMoves(moves);
  orderMoves(board, moves);
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
  TtMap tt;
  for (const Move& m : moves) {
    if (!board.makeMove(m)) continue;
    float v = -negamax(board, depth - 1, 1, -beta, -alpha, false, full, &cheap, cfg, stats, hist, tt);
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
