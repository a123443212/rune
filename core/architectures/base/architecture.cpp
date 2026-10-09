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
  return arch + "|" + archVersion + "|" + game + "|" + featureSet + "|t" + std::to_string(tokens) + "x" +
         std::to_string(tokenDim) + "|" + attention + "|" + geometricBias + "|" + head + "|" +
         quantization + "|" + variant + "|[" + dims + "]|" + pooling + "|" +
         (gateOn ? "gate" : "nogate") + "|" + (poolClip ? "clip" : "noclip") +
         "|sw" + std::to_string(sharedWidth) + "|hb" + std::to_string(headBuckets);
}

uint64_t ModelSpec::configHash() const { return fnv1aHash(canonicalString()); }

bool gateFromString(const std::string& name, GateFn& out) {
  if (name == "clip") {
    out = GateFn::Clip;
    return true;
  }
  if (name == "hard_sigmoid") {
    out = GateFn::HardSigmoid;
    return true;
  }
  if (name == "screlu") {
    out = GateFn::Screlu;
    return true;
  }
  return false;
}

const char* gateName(GateFn fn) {
  if (fn == GateFn::HardSigmoid) return "hard_sigmoid";
  if (fn == GateFn::Screlu) return "screlu";
  return "clip";
}

float applyGate(GateFn fn, float s) {
  if (fn == GateFn::HardSigmoid) {
    float v = 0.2f * s + 0.5f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
  }
  if (fn == GateFn::Screlu) {
    float c = s;
    if (c < 0.0f) c = 0.0f;
    if (c > 1.0f) c = 1.0f;
    return c * c;
  }
  if (s < 0.0f) return 0.0f;
  if (s > 1.0f) return 1.0f;
  return s;
}

}
