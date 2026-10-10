#include "core/go/go_planes_v02.h"

#include "core/go/go_rules.h"

namespace rune {
namespace go {

void extractPlanesV02Into(const GoState& st, float* out) {
  int n = st.board.size();
  uint8_t stm = st.board.sideToMove();
  std::vector<int8_t> raw(static_cast<size_t>(n * n));
  for (int sq = 0; sq < n * n; ++sq) raw[static_cast<size_t>(sq)] = st.board.atSq(sq);
  std::vector<int> libs;
  goLibertyMap(raw, n, libs);
  for (int p = 0; p < kPlanesV02; ++p) {
    for (int sq = 0; sq < n * n; ++sq) out[static_cast<size_t>(p * n * n + sq)] = 0.0f;
  }
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      int sq = r * n + c;
      int8_t v = raw[static_cast<size_t>(sq)];
      bool mine = (v == 1 && stm == kBlack) || (v == -1 && stm == kWhite);
      bool theirs = (v == 1 || v == -1) && !mine && v != 0;
      if (mine) out[sq] = 1.0f;
      if (theirs) out[n * n + sq] = 1.0f;
      if (v == 0) {
        out[2 * n * n + sq] = 1.0f;
      } else {
        int k = libs[static_cast<size_t>(sq)];
        if (k <= 1) out[3 * n * n + sq] = 1.0f;
        else if (k == 2) out[4 * n * n + sq] = 1.0f;
        else out[5 * n * n + sq] = 1.0f;
      }
      if (st.hasKo && sq == st.ko) out[6 * n * n + sq] = 1.0f;
      out[7 * n * n + sq] = static_cast<float>(stm);
    }
  }
}

void extractPlanesV02(const GoState& st, std::vector<float>& out) {
  int n = st.board.size();
  out.assign(static_cast<size_t>(kPlanesV02 * n * n), 0.0f);
  extractPlanesV02Into(st, out.data());
}

}
}
