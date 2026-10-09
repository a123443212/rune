#include "core/go/go_resnet.h"

#include <cmath>

#include "core/kernels/conv_kernels.h"
#include "core/kernels/policy_kernels.h"
#include "core/kernels/ref_kernels.h"

namespace rune {
namespace go {

GoResnetOutput forwardGoResnet(const GoResnetWeights& wt, const GoResnetSizes& sz,
                               const float* planes) {
  int b = sz.board;
  int c = sz.channels;
  int hw = b * b;
  std::vector<float> cur(static_cast<size_t>(c * hw), 0.0f);
  conv::conv2dNchw(planes, wt.stemW.data(), wt.stemB.data(), cur.data(), 1, 1, c, b, b, 3, 3,
                   1, 1);
  conv::reluInplace(cur.data(), cur.size());
  std::vector<float> tmp(static_cast<size_t>(c * hw), 0.0f);
  std::vector<float> tmp2(static_cast<size_t>(c * hw), 0.0f);
  std::vector<float> out(static_cast<size_t>(c * hw), 0.0f);
  for (int i = 0; i < sz.blocks; ++i) {
    conv::conv2dNchw(cur.data(), wt.blockW1[static_cast<size_t>(i)].data(),
                     wt.blockB1[static_cast<size_t>(i)].data(), tmp.data(), 1, c, c, b, b, 3,
                     3, 1, 1);
    conv::reluInplace(tmp.data(), tmp.size());
    conv::conv2dNchw(tmp.data(), wt.blockW2[static_cast<size_t>(i)].data(),
                     wt.blockB2[static_cast<size_t>(i)].data(), tmp2.data(), 1, c, c, b, b, 3,
                     3, 1, 1);
    conv::residualAddRelu(cur.data(), tmp2.data(), out.data(), out.size());
    cur = out;
  }
  std::vector<float> pooled(static_cast<size_t>(c), 0.0f);
  conv::globalAvgPool(cur.data(), pooled.data(), 1, c, b, b);
  std::vector<float> h(static_cast<size_t>(sz.valueH2), 0.0f);
  ref::matVec(wt.vh1.data(), pooled.data(), wt.bh1.data(), h.data(), sz.valueH2, c);
  for (float& v : h) {
    if (v < 0.0f) v = 0.0f;
    else if (v > 1.0f) v = 1.0f;
  }
  float vv = wt.bv;
  for (int i = 0; i < sz.valueH2; ++i) vv += wt.wv[static_cast<size_t>(i)] * h[static_cast<size_t>(i)];
  GoResnetOutput r;
  r.value = std::tanh(vv);
  ref::matVec(wt.wwdl.data(), h.data(), wt.bwdl.data(), r.wdl, 3, sz.valueH2);
  std::vector<float> logits(static_cast<size_t>(sz.policySize), 0.0f);
  ref::matVec(wt.wpol.data(), cur.data(), wt.bpol.data(), logits.data(), sz.policySize,
              c * hw);
  r.policy.assign(static_cast<size_t>(sz.policySize), 0.0f);
  policy::softmax(logits.data(), r.policy.data(), r.policy.size());
  return r;
}

}
}
