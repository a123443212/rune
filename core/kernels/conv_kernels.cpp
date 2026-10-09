#include "core/kernels/conv_kernels.h"

namespace rune {
namespace conv {

void conv2dNchw(const float* input, const float* weight, const float* bias, float* out, int n,
                int cin, int cout, int h, int w, int kh, int kw, int padH, int padW) {
  int oh = h + 2 * padH - kh + 1;
  int ow = w + 2 * padW - kw + 1;
  for (int nn = 0; nn < n; ++nn) {
    for (int co = 0; co < cout; ++co) {
      for (int oy = 0; oy < oh; ++oy) {
        for (int ox = 0; ox < ow; ++ox) {
          float acc = bias ? bias[co] : 0.0f;
          for (int ci = 0; ci < cin; ++ci) {
            for (int ky = 0; ky < kh; ++ky) {
              for (int kx = 0; kx < kw; ++kx) {
                int iy = oy + ky - padH;
                int ix = ox + kx - padW;
                if (iy < 0 || ix < 0 || iy >= h || ix >= w) continue;
                float iv = input[((nn * cin + ci) * h + iy) * w + ix];
                float wv = weight[((co * cin + ci) * kh + ky) * kw + kx];
                acc += iv * wv;
              }
            }
          }
          out[((nn * cout + co) * oh + oy) * ow + ox] = acc;
        }
      }
    }
  }
}

void conv3x3Pad1(const float* input, const float* weight, const float* bias, float* out, int cin,
                 int cout, int h, int w) {
  for (int co = 0; co < cout; ++co) {
    float b = bias ? bias[co] : 0.0f;
    const float* kw = weight + co * cin * 9;
    float* oplane = out + co * h * w;
    for (int oy = 0; oy < h; ++oy) {
      for (int ox = 0; ox < w; ++ox) {
        float acc = b;
        for (int ci = 0; ci < cin; ++ci) {
          const float* iplane = input + ci * h * w;
          const float* k = kw + ci * 9;
          if (oy > 0) {
            const float* row = iplane + (oy - 1) * w;
            if (ox > 0) acc += row[ox - 1] * k[0];
            acc += row[ox] * k[1];
            if (ox + 1 < w) acc += row[ox + 1] * k[2];
          }
          {
            const float* row = iplane + oy * w;
            if (ox > 0) acc += row[ox - 1] * k[3];
            acc += row[ox] * k[4];
            if (ox + 1 < w) acc += row[ox + 1] * k[5];
          }
          if (oy + 1 < h) {
            const float* row = iplane + (oy + 1) * w;
            if (ox > 0) acc += row[ox - 1] * k[6];
            acc += row[ox] * k[7];
            if (ox + 1 < w) acc += row[ox + 1] * k[8];
          }
        }
        oplane[oy * w + ox] = acc;
      }
    }
  }
}

void conv3x3Pad1Relu(const float* input, const float* weight, const float* bias, float* out,
                     int cin, int cout, int h, int w) {
  for (int co = 0; co < cout; ++co) {
    float b = bias ? bias[co] : 0.0f;
    const float* kw = weight + co * cin * 9;
    float* oplane = out + co * h * w;
    for (int oy = 0; oy < h; ++oy) {
      for (int ox = 0; ox < w; ++ox) {
        float acc = b;
        for (int ci = 0; ci < cin; ++ci) {
          const float* iplane = input + ci * h * w;
          const float* k = kw + ci * 9;
          if (oy > 0) {
            const float* row = iplane + (oy - 1) * w;
            if (ox > 0) acc += row[ox - 1] * k[0];
            acc += row[ox] * k[1];
            if (ox + 1 < w) acc += row[ox + 1] * k[2];
          }
          {
            const float* row = iplane + oy * w;
            if (ox > 0) acc += row[ox - 1] * k[3];
            acc += row[ox] * k[4];
            if (ox + 1 < w) acc += row[ox + 1] * k[5];
          }
          if (oy + 1 < h) {
            const float* row = iplane + (oy + 1) * w;
            if (ox > 0) acc += row[ox - 1] * k[6];
            acc += row[ox] * k[7];
            if (ox + 1 < w) acc += row[ox + 1] * k[8];
          }
        }
        oplane[oy * w + ox] = acc < 0.0f ? 0.0f : acc;
      }
    }
  }
}

void reluInplace(float* buf, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (buf[i] < 0.0f) buf[i] = 0.0f;
  }
}

void residualAddRelu(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float s = a[i] + b[i];
    out[i] = s < 0.0f ? 0.0f : s;
  }
}

void residualAddReluInplace(const float* base, float* delta, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float s = base[i] + delta[i];
    delta[i] = s < 0.0f ? 0.0f : s;
  }
}

void globalAvgPool(const float* input, float* out, int n, int c, int h, int w) {
  float hw = static_cast<float>(h * w);
  for (int nn = 0; nn < n; ++nn) {
    for (int cc = 0; cc < c; ++cc) {
      float acc = 0.0f;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          acc += input[((nn * c + cc) * h + y) * w + x];
        }
      }
      out[nn * c + cc] = acc / hw;
    }
  }
}

}
}
