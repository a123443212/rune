#pragma once
#include <cstddef>
#include <vector>
namespace rune {
namespace rt {
class Arena {
 public:
  explicit Arena(size_t bytes = 8192);
  float* at(size_t offBytes, size_t elems);
  const float* at(size_t offBytes, size_t elems) const;
  size_t bytes() const;
  void clear();
 private:
  std::vector<float> buf_;
};
size_t arenaBytesFor(int tokens, int dim, int h1, int h2);
}
}
