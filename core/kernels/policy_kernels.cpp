#include "core/kernels/policy_kernels.h"

#include <cmath>
#include <cfloat>

namespace rune {
namespace policy {

void softmax(const float* logits, float* out, size_t n) {
  float m = -FLT_MAX;
  for (size_t i = 0; i < n; ++i) {
    if (logits[i] > m) m = logits[i];
  }
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float e = std::exp(logits[i] - m);
    out[i] = e;
    s += e;
  }
  if (!(s > 0.0f) || !std::isfinite(s)) {
    float u = 1.0f / static_cast<float>(n ? n : 1);
    for (size_t i = 0; i < n; ++i) out[i] = u;
    return;
  }
  for (size_t i = 0; i < n; ++i) out[i] /= s;
}

void normalizeMasked(const float* logits, const bool* legal, float* out, size_t n) {
  float m = -FLT_MAX;
  bool any = false;
  for (size_t i = 0; i < n; ++i) {
    if (legal[i]) {
      any = true;
      if (logits[i] > m) m = logits[i];
    }
  }
  if (!any) {
    for (size_t i = 0; i < n; ++i) out[i] = 0.0f;
    return;
  }
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    if (legal[i]) {
      float e = std::exp(logits[i] - m);
      out[i] = e;
      s += e;
    } else {
      out[i] = 0.0f;
    }
  }
  if (!(s > 0.0f) || !std::isfinite(s)) {
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i) {
      if (legal[i]) ++cnt;
    }
    float u = cnt ? 1.0f / static_cast<float>(cnt) : 0.0f;
    for (size_t i = 0; i < n; ++i) out[i] = legal[i] ? u : 0.0f;
    return;
  }
  for (size_t i = 0; i < n; ++i) out[i] /= s;
}

float entropy(const float* probs, size_t n) {
  float h = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    if (probs[i] > 0.0f) h -= probs[i] * std::log(probs[i]);
  }
  return h;
}

}
}
