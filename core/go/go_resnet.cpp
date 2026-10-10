/*
RUNE — Relational Unified Neural Evaluator
Copyright (C) 2026 a123443212

SPDX-License-Identifier: MIT OR Apache-2.0

This project is dual-licensed under the MIT License and the
Apache License, Version 2.0. You may choose either license
when using, copying, modifying, or distributing this software.

MIT License: https://opensource.org/license/mit
Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, this
software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
*/

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
  conv::conv2dNchw(planes, wt.stemW.data(), wt.stemB.data(), cur.data(), 1, sz.inPlanes, c, b, b, 3, 3,
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

GoResnetOutput forwardGoResnetFast(const GoResnetWeights& wt, const GoResnetSizes& sz,
                                   const float* planes, GoResnetScratch& sc) {
  int b = sz.board;
  int c = sz.channels;
  int hw = b * b;
  sc.ensure(b, c, sz.valueH2, sz.policySize);
  conv::conv3x3Pad1Relu(planes, wt.stemW.data(), wt.stemB.data(), sc.a.data(), sz.inPlanes, c, b, b);
  float* cur = sc.a.data();
  float* nxt = sc.b.data();
  float* tmp = sc.c.data();
  for (int i = 0; i < sz.blocks; ++i) {
    conv::conv3x3Pad1Relu(cur, wt.blockW1[static_cast<size_t>(i)].data(),
                          wt.blockB1[static_cast<size_t>(i)].data(), tmp, c, c, b, b);
    conv::conv2dNchw(tmp, wt.blockW2[static_cast<size_t>(i)].data(),
                     wt.blockB2[static_cast<size_t>(i)].data(), nxt, 1, c, c, b, b, 3, 3, 1,
                     1);
    conv::residualAddReluInplace(cur, nxt, static_cast<size_t>(c * hw));
    float* t = cur;
    cur = nxt;
    nxt = t;
  }
  float* fin = (sz.blocks % 2 == 0) ? sc.a.data() : sc.b.data();
  conv::globalAvgPool(fin, sc.pooled.data(), 1, c, b, b);
  ref::matVec(wt.vh1.data(), sc.pooled.data(), wt.bh1.data(), sc.h.data(), sz.valueH2, c);
  for (int i = 0; i < sz.valueH2; ++i) {
    if (sc.h[static_cast<size_t>(i)] < 0.0f) sc.h[static_cast<size_t>(i)] = 0.0f;
    else if (sc.h[static_cast<size_t>(i)] > 1.0f) sc.h[static_cast<size_t>(i)] = 1.0f;
  }
  float vv = wt.bv;
  for (int i = 0; i < sz.valueH2; ++i) vv += wt.wv[static_cast<size_t>(i)] * sc.h[static_cast<size_t>(i)];
  GoResnetOutput r;
  r.value = std::tanh(vv);
  ref::matVec(wt.wwdl.data(), sc.h.data(), wt.bwdl.data(), r.wdl, 3, sz.valueH2);
  sc.logits.assign(static_cast<size_t>(sz.policySize), 0.0f);
  ref::matVec(wt.wpol.data(), fin, wt.bpol.data(), sc.logits.data(), sz.policySize, c * hw);
  r.policy.assign(static_cast<size_t>(sz.policySize), 0.0f);
  policy::softmax(sc.logits.data(), r.policy.data(), r.policy.size());
  (void)cur;
  return r;
}

}
}
