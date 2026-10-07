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

uint64_t fnv1aHash(const uint8_t* data, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

std::string ModelSpec::canonicalString() const {
  std::string dims;
  for (size_t i = 0; i < tokenDims.size(); ++i) {
    if (i > 0) dims += ",";
    dims += std::to_string(tokenDims[i]);
  }
  return arch + "|" + archVersion + "|" + featureSet + "|t" + std::to_string(tokens) + "x" +
         std::to_string(tokenDim) + "|" + attention + "|" + geometricBias + "|" + head + "|" +
         quantization + "|" + variant + "|[" + dims + "]|" + pooling + "|" +
         (gateOn ? "gate" : "nogate") + "|" + (poolClip ? "clip" : "noclip") +
         "|sw" + std::to_string(sharedWidth);
}

uint64_t ModelSpec::configHash() const { return fnv1aHash(canonicalString()); }

}
