#pragma once

#include <cstddef>
#include <vector>

namespace rune {
namespace go {

struct GoResnetScratch {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c;
  std::vector<float> pooled;
  std::vector<float> h;
  std::vector<float> logits;

  void ensure(int board, int channels, int h2, int policy);
};

}
}
