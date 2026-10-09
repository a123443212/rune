#pragma once

#include <cstddef>

namespace rune {
namespace policy {

void softmax(const float* logits, float* out, size_t n);
void normalizeMasked(const float* logits, const bool* legal, float* out, size_t n);
float entropy(const float* probs, size_t n);

}
}
