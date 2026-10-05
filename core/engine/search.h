#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/board/board.h"
namespace rune {
namespace eng {
enum class LazyMode { L0, L1, L2 };
struct LazyConfig {
  LazyMode mode = LazyMode::L0;
  float margin = 0.08f;
  int maxRefine = 1;
};
struct SearchStats {
  long nodes = 0;
  long evals = 0;
  long refined = 0;
  long cutoffs = 0;
  double seconds = 0.0;
  float rootScore = 0.0f;
  bool hasMove = false;
  Move rootMove;
  long rootCount = 0;
  long pvCount = 0;
  long cutCount = 0;
  long leafCount = 0;
  double nps() const;
};
using EvalFn = std::function<float(Board&)>;
float searchRoot(Board& board, int depth, EvalFn ev, const LazyConfig& cfg, SearchStats& stats, float alpha, float beta);
}
}
