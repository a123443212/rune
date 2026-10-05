#pragma once
#include <cstdint>
namespace rune {
namespace qspec {
void dequantClip(const int32_t* acc, float scale, float* out, int n);
void dequantClipBatched(const float* scales8, const int32_t* acc, float* out, int dim);
void clampRow(float* out, int n, float lo, float hi);
int32_t quantizeHalfAwaySpec(float w, float scale, int bound);
}
}
