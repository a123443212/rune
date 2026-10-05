#include "core/runtime/arena.h"
namespace rune {
namespace rt {
Arena::Arena(size_t bytes) {
  size_t elems = (bytes + 3) / 4;
  if (elems == 0) elems = 1;
  buf_.assign(elems, 0.0f);
}
float* Arena::at(size_t offBytes, size_t elems) {
  size_t o = offBytes / 4;
  return buf_.data() + o;
}
const float* Arena::at(size_t offBytes, size_t elems) const {
  size_t o = offBytes / 4;
  return buf_.data() + o;
}
size_t Arena::bytes() const {
  return buf_.size() * 4;
}
void Arena::clear() {
  for (float& v : buf_) v = 0.0f;
}
size_t arenaBytesFor(int tokens, int dim, int h1, int h2) {
  size_t live[9] = {(size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * dim, (size_t)tokens * tokens, (size_t)tokens * dim, (size_t)h1, (size_t)h2, (size_t)tokens * dim};
  size_t t = 0;
  for (size_t e : live) t += e * 4 + 32;
  return (t + 31) / 32 * 32;
}
}
}
