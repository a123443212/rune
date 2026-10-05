#include "core/kernels/quant_specialized.h"
#include <cmath>
#include "core/kernels/ref_kernels.h"
namespace rune {
namespace qspec {
void dequantClip(const int32_t* acc, float scale, float* out, int n) {
  for (int i = 0; i < n; ++i) out[i] = ref::clippedRelu((float)acc[i] * scale);
}
void dequantClipBatched(const float* scales8, const int32_t* acc, float* out, int dim) {
  for (int g = 0; g < 8; ++g) {
    float s = scales8[g];
    for (int d = 0; d < dim; ++d) out[g * dim + d] = ref::clippedRelu((float)acc[g * dim + d] * s);
  }
}
void clampRow(float* out, int n, float lo, float hi) {
  for (int i = 0; i < n; ++i) {
    if (out[i] < lo) out[i] = lo;
    else if (out[i] > hi) out[i] = hi;
  }
}
int32_t quantizeHalfAwaySpec(float w, float scale, int bound) {
  return ref::quantizeHalfAway(w, scale, bound);
}
}
}
