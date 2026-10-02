#include "core/architectures/dense/pooling.h"

#include "core/simd/simd.h"

namespace rune {

bool poolModeFromString(const std::string& name, PoolMode& out) {
  if (name == "none") {
    out = PoolMode::None;
    return true;
  }
  if (name == "per_token") {
    out = PoolMode::PerToken;
    return true;
  }
  if (name == "shared") {
    out = PoolMode::Shared;
    return true;
  }
  return false;
}

const char* poolModeName(PoolMode mode) {
  if (mode == PoolMode::PerToken) return "per_token";
  if (mode == PoolMode::Shared) return "shared";
  return "none";
}

float TokenPool::clip01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

TokenPool::TokenPool() {
  VarWidths dflt;
  std::string err;
  configure(dflt, PoolMode::None, true, 32, err);
}

bool TokenPool::configure(const VarWidths& widths, PoolMode mode, bool clipOut, int sharedWidth,
                          std::string& err) {
  if (mode == PoolMode::Shared && sharedWidth < 8) {
    err = "shared width too small";
    return false;
  }
  if (mode == PoolMode::Shared) {
    for (int t = 0; t < 8; ++t) {
      if (widths.w[t] > sharedWidth) {
        err = "token dim exceeds shared width";
        return false;
      }
    }
  }
  widths_ = widths;
  mode_ = mode;
  clip_ = clipOut;
  sharedWidth_ = sharedWidth;
  for (int t = 0; t < 8; ++t) {
    int w = widths_.w[t];
    perW[t].assign(w * w, 0.0f);
    perB[t].assign(w, 0.0f);
    for (int i = 0; i < w; ++i) perW[t][i * w + i] = 1.0f;
    tokS[t].assign(w, 1.0f);
    tokB[t].assign(w, 0.0f);
  }
  sharedS.assign(sharedWidth_ * sharedWidth_, 0.0f);
  for (int i = 0; i < sharedWidth_; ++i) sharedS[i * sharedWidth_ + i] = 1.0f;
  return true;
}

void TokenPool::forward(const float* in, float* out) const {
  int inOff = 0;
  int outOff = 0;
  for (int t = 0; t < 8; ++t) {
    int w = widths_.w[t];
    if (mode_ == PoolMode::None) {
      for (int d = 0; d < w; ++d) {
        float v = in[inOff + d];
        out[outOff + d] = clip_ ? clip01(v) : v;
      }
    } else if (mode_ == PoolMode::PerToken) {
      simd::matVec(perW[t].data(), in + inOff, perB[t].data(), out + outOff, w, w);
      if (clip_) {
        for (int d = 0; d < w; ++d) out[outOff + d] = clip01(out[outOff + d]);
      }
    } else {
      if (scratch_.size() < static_cast<size_t>(sharedWidth_)) scratch_.assign(sharedWidth_, 0.0f);
      float* tmp = scratch_.data();
      for (int d = 0; d < sharedWidth_; ++d) tmp[d] = 0.0f;
      simd::matVec(sharedS.data(), in + inOff, nullptr, tmp, sharedWidth_, sharedWidth_);
      for (int d = 0; d < w; ++d) {
        float v = tmp[d] * tokS[t][d] + tokB[t][d];
        out[outOff + d] = clip_ ? clip01(v) : v;
      }
    }
    inOff += (mode_ == PoolMode::Shared) ? sharedWidth_ : w;
    outOff += w;
  }
}

size_t TokenPool::parameterCount() const {
  if (mode_ == PoolMode::None) return 0;
  if (mode_ == PoolMode::PerToken) {
    size_t n = 0;
    for (int t = 0; t < 8; ++t) n += perW[t].size() + perB[t].size();
    return n;
  }
  size_t n = sharedS.size();
  for (int t = 0; t < 8; ++t) n += tokS[t].size() + tokB[t].size();
  return n;
}

float ChannelGate::clip01(float v) {
  if (v < 0.0f) return 0.0f;
  if (v > 1.0f) return 1.0f;
  return v;
}

ChannelGate::ChannelGate() {
  VarWidths dflt;
  std::string err;
  configure(dflt, false, err);
}

bool ChannelGate::configure(const VarWidths& widths, bool enabled, std::string& err) {
  (void)err;
  widths_ = widths;
  enabled_ = enabled;
  offsets_.assign(9, 0);
  for (int t = 0; t < 8; ++t) offsets_[t + 1] = offsets_[t] + widths_.w[t];
  ga.assign(offsets_[8], 0.0f);
  gb.assign(offsets_[8], 0.0f);
  return true;
}

void ChannelGate::forward(const float* in, float* out) const {
  if (!enabled_) {
    for (int i = 0; i < offsets_[8]; ++i) out[i] = in[i];
    return;
  }
  for (int i = 0; i < offsets_[8]; ++i) {
    float g = clip01(ga[i] * in[i] + gb[i]);
    out[i] = in[i] * g;
  }
}

size_t ChannelGate::parameterCount() const {
  if (!enabled_) return 0;
  return ga.size() + gb.size();
}

}
