#include "core/architectures/base/architecture.h"

namespace rune {

uint64_t fnv1aHash(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

std::string ModelSpec::canonicalString() const {
  return arch + "|" + archVersion + "|" + featureSet + "|t" + std::to_string(tokens) + "x" +
         std::to_string(tokenDim) + "|" + attention + "|" + geometricBias + "|" + head + "|" +
         quantization;
}

uint64_t ModelSpec::configHash() const { return fnv1aHash(canonicalString()); }

}
