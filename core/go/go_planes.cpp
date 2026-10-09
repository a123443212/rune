#include "core/go/go_planes.h"

namespace rune {
namespace go {

void extractPlanesInto(const GoBoard& board, float* out) {
  int n = board.size();
  uint8_t stm = board.sideToMove();
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      int8_t v = board.at(r, c);
      float rel = 0.0f;
      if (v != 0) {
        bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
        rel = mine ? 1.0f : -1.0f;
      }
      out[r * n + c] = rel;
    }
  }
}

void extractPlanes(const GoBoard& board, std::vector<float>& out) {
  int n = board.size();
  out.assign(static_cast<size_t>(n * n), 0.0f);
  extractPlanesInto(board, out.data());
}

}
}
