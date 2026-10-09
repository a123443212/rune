#include "core/go/go_scratch.h"

namespace rune {
namespace go {

void GoResnetScratch::ensure(int board, int channels, int h2, int policy) {
  size_t conv = static_cast<size_t>(channels) * board * board;
  if (a.size() < conv) {
    a.assign(conv, 0.0f);
    b.assign(conv, 0.0f);
    c.assign(conv, 0.0f);
  }
  if (pooled.size() < static_cast<size_t>(channels)) pooled.assign(channels, 0.0f);
  if (h.size() < static_cast<size_t>(h2)) h.assign(h2, 0.0f);
  if (logits.size() < static_cast<size_t>(policy)) logits.assign(policy, 0.0f);
}

}
}
