#include "tests/cpp/test_multihead.h"
#include "tests/cpp/test_framework.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <string>
#include "core/architectures/attention/multi_head.h"
#include "core/board/board.h"
#include "core/inference/evaluator.h"
#include "core/model_io/model_io.h"

using namespace rune;

namespace {

double scanNumber(const std::string& s, size_t& p) {
  while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t' ||
                          s[p] == ':' || s[p] == ',' || s[p] == '[')) ++p;
  size_t e = p;
  while (e < s.size() && (std::isdigit(s[e]) || s[e] == '.' || s[e] == '-' || s[e] == '+' ||
                          s[e] == 'e' || s[e] == 'E')) ++e;
  double v = std::stod(s.substr(p, e - p));
  p = e;
  return v;
}

void testMixerGeometry() {
  CHECK(MultiHeadMixer::kHeads == 4);
  CHECK(MultiHeadMixer::kHeadDim == 8);
  CHECK(MultiHeadMixer::kTokens == 8);
  CHECK(MultiHeadMixer::kDim == 32);
  CHECK(MultiHeadMixer::kHeads * MultiHeadMixer::kHeadDim == MultiHeadMixer::kDim);
  MultiHeadMixer m;
  CHECK(m.parameterCount() == 4 * (3 * (8 * 32 + 8) + 64) + (32 * 32 + 32));
}

void testMixerResidualZeroWeights() {
  MultiHeadMixer m;
  for (int h = 0; h < 4; ++h) {
    std::fill(m.wq[h].begin(), m.wq[h].end(), 0.0f);
    std::fill(m.bq[h].begin(), m.bq[h].end(), 0.0f);
    std::fill(m.wk[h].begin(), m.wk[h].end(), 0.0f);
    std::fill(m.bk[h].begin(), m.bk[h].end(), 0.0f);
    std::fill(m.wv[h].begin(), m.wv[h].end(), 0.0f);
    std::fill(m.bv[h].begin(), m.bv[h].end(), 0.0f);
    std::fill(m.gab[h].begin(), m.gab[h].end(), 0.0f);
  }
  std::fill(m.wo.begin(), m.wo.end(), 0.0f);
  std::fill(m.bwo.begin(), m.bwo.end(), 0.0f);
  float x[256];
  for (int i = 0; i < 256; ++i) x[i] = 0.01f * (i % 17);
  float out[256] = {0.0f};
  m.forward(x, out);
  for (int i = 0; i < 256; ++i) CHECK_CLOSE(out[i], x[i], 1e-6);
}

void testPerHeadGabIsolation() {
  MultiHeadMixer a;
  MultiHeadMixer b;
  b.wq[0] = a.wq[0];
  b.bq[0] = a.bq[0];
  b.wk[0] = a.wk[0];
  b.bk[0] = a.bk[0];
  b.wv[0] = a.wv[0];
  b.bv[0] = a.bv[0];
  b.gab[0] = a.gab[0];
  b.wo = a.wo;
  b.bwo = a.bwo;
  for (int h = 1; h < 4; ++h) {
    std::fill(b.wq[h].begin(), b.wq[h].end(), 0.0f);
    std::fill(b.wk[h].begin(), b.wk[h].end(), 0.0f);
    std::fill(b.wv[h].begin(), b.wv[h].end(), 0.0f);
  }
  float x[256];
  for (int i = 0; i < 256; ++i) x[i] = 0.01f * ((i * 7) % 13);
  float oa[256] = {0.0f};
  float ob[256] = {0.0f};
  a.forward(x, oa);
  b.forward(x, ob);
  bool differs = false;
  for (int i = 0; i < 256; ++i) {
    if (std::fabs(oa[i] - ob[i]) > 1e-6) differs = true;
  }
  CHECK(differs);
}

void testMultiHeadGolden() {
  std::ifstream f("spec/test-vectors/v10/multi_head.json");
  CHECK(!!f);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  RuneFile rf;
  std::string err;
  CHECK(loadRuneFile("spec/test-vectors/models/small-mh4-fp32.rune", rf, err));
  CHECK(rf.spec.arch == "RUNE-ATTN-MH4");
  CHECK(rf.spec.attention == "multi_head");
  CHECK(rf.spec.headBuckets == 1);
  size_t p = 0;
  int count = 0;
  while (true) {
    size_t fp = s.find("\"fen\"", p);
    if (fp == std::string::npos) break;
    size_t q0 = s.find('"', fp + 5) + 1;
    size_t q1 = s.find('"', q0);
    std::string fen = s.substr(q0, q1 - q0);
    size_t vp = s.find("\"value\"", q1);
    size_t vv = vp + 8;
    double value = scanNumber(s, vv);
    size_t wp = s.find("\"wdl\"", vv);
    size_t wv = wp + 6;
    double wdl[3];
    for (int i = 0; i < 3; ++i) wdl[i] = scanNumber(s, wv);
    p = wv;
    Board b;
    CHECK(b.setFen(fen));
    Evaluator ev(&rf.embeddings, rf.arch.get());
    auto r = ev.evaluateBoard(b);
    CHECK(std::fabs(r.value - value) < 1e-5);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(r.wdl[i] - wdl[i]) < 1e-5);
    ++count;
  }
  CHECK(count == 3);
}

}  // namespace

void runMultiHeadTests() {
  testMixerGeometry();
  testMixerResidualZeroWeights();
  testPerHeadGabIsolation();
  testMultiHeadGolden();
}
