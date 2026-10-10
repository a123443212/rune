#include <fstream>
#include <string>
#include <vector>

#include "core/go/go_model.h"
#include "core/go/go_planes_v02.h"
#include "core/go/go_resnet.h"
#include "core/go/go_rules.h"
#include "core/go/go_state.h"
#include "core/go/go_token_v02.h"
#include "tests/cpp/test_framework.h"

using namespace rune;
using namespace rune::go;

namespace {

bool hasFeature(const std::vector<ActiveFeature>& feats, int g, int idx) {
  for (const auto& f : feats) {
    if (f.group == g && f.index == idx) return true;
  }
  return false;
}

std::string emptyV02(int n) {
  std::string grid;
  for (int r = 0; r < n; ++r) {
    if (r) grid += "/";
    grid += std::string(static_cast<size_t>(n), '.');
  }
  return grid + " b - 7.5 1 0";
}

void testParseV02() {
  GoState st;
  CHECK(st.setState(emptyV02(9)));
  CHECK(st.board.size() == 9);
  CHECK(!st.hasKo);
  CHECK(st.moveNo == 1);
  GoState ko;
  std::string grid;
  for (int r = 0; r < 9; ++r) {
    if (r) grid += "/";
    grid += std::string(9, '.');
  }
  CHECK(ko.setState(grid + " w 5 7.5 20 1"));
  CHECK(ko.hasKo);
  CHECK(ko.ko == 5);
  CHECK(ko.moveNo == 20);
  CHECK(ko.passNo == 1);
  CHECK(ko.encode() == grid + " w 5 7.500000 20 1");
  GoState old;
  CHECK(old.setState(grid + " b"));
  CHECK(!old.hasKo);
  GoState bad;
  CHECK(!bad.setState(grid + " b 9999 7.5 1 0"));
}

void testPlanesV02() {
  GoState st;
  CHECK(st.setState(emptyV02(9)));
  std::vector<float> p;
  extractPlanesV02(st, p);
  CHECK(p.size() == 8 * 81);
  float empty = 0.0f;
  float own = 0.0f;
  for (int sq = 0; sq < 81; ++sq) {
    own += p[static_cast<size_t>(sq)];
    empty += p[static_cast<size_t>(2 * 81 + sq)];
  }
  CHECK(own == 0.0f);
  CHECK(empty == 81.0f);
  GoState one;
  CHECK(one.setState("........./........./........./........./....X..../........./........./........./......... b - 7.5 10 0"));
  std::vector<float> q;
  extractPlanesV02(one, q);
  CHECK(q[4 * 9 + 4] == 1.0f);
  CHECK(q[5 * 81 + 4 * 9 + 4] == 1.0f);
  CHECK(q[3 * 81 + 4 * 9 + 4] == 0.0f);
}

void testContextV02() {
  GoState st;
  CHECK(st.setState(emptyV02(9)));
  float ctx[12];
  st.contextV02(ctx);
  CHECK(ctx[0] == 0.0f);
  CHECK(ctx[6] == 0.0f);
  CHECK_CLOSE(ctx[3], 0.5, 1e-6);
  CHECK_CLOSE(ctx[10], 1.0, 1e-6);
  CHECK(st.phaseV02() == 0);
  GoState late;
  std::string grid;
  for (int r = 0; r < 9; ++r) {
    if (r) grid += "/";
    grid += std::string(9, '.');
  }
  CHECK(late.setState(grid + " b - 7.5 70 0"));
  CHECK(late.phaseV02() == 2);
}

void testRulesV02() {
  int n = 9;
  std::vector<int8_t> board(static_cast<size_t>(n * n), 0);
  board[1] = 1;
  board[9] = 1;
  std::vector<int8_t> next;
  std::vector<int> captured;
  bool ok = false;
  int oko = -1;
  CHECK(!goPlayStone(board, n, 0, -1, false, -1, next, ok, oko, captured));
  std::vector<int> legal;
  goLegalMoves(board, n, 1, false, -1, legal);
  bool found = false;
  for (int m : legal) {
    if (m == -1) found = true;
  }
  CHECK(found);
  float s = goAreaScore(std::vector<int8_t>(81, 0), 9, 7.5f);
  CHECK_CLOSE(s, -7.5, 1e-6);
  GoState st;
  CHECK(st.setState(emptyV02(9)));
  std::vector<int8_t> raw(81);
  for (int i = 0; i < 81; ++i) raw[static_cast<size_t>(i)] = st.board.atSq(i);
  goLegalMoves(raw, 9, 0, false, -1, legal);
  CHECK(legal.size() == 82);
}

void testTokenV02() {
  GoState st;
  CHECK(st.setState(emptyV02(9)));
  std::vector<ActiveFeature> feats;
  GoTokenV02::extract(st, feats);
  CHECK(!feats.empty());
  for (size_t i = 1; i < feats.size(); ++i) CHECK(feats[i - 1] < feats[i]);
  CHECK(hasFeature(feats, 7, 0));
  CHECK(GoTokenV02::phaseFromGroup7(feats) == 0);
  GoState one;
  CHECK(one.setState("........./........./........./........./....X..../........./........./........./......... b - 7.5 10 0"));
  std::vector<ActiveFeature> f2;
  GoTokenV02::extract(one, f2);
  CHECK(!f2.empty());
}

void testResnetV02() {
  GoResnetSizes sz;
  sz.board = 9;
  sz.channels = 2;
  sz.blocks = 1;
  sz.policySize = 82;
  sz.valueH2 = 4;
  sz.inPlanes = 8;
  GoResnetWeights wt;
  wt.stemW.assign(static_cast<size_t>(2 * 8 * 3 * 3), 0.05f);
  wt.stemB.assign(2, 0.0f);
  wt.blockW1.assign(1, std::vector<float>(static_cast<size_t>(2 * 2 * 3 * 3), 0.02f));
  wt.blockB1.assign(1, std::vector<float>(2, 0.0f));
  wt.blockW2.assign(1, std::vector<float>(static_cast<size_t>(2 * 2 * 3 * 3), 0.02f));
  wt.blockB2.assign(1, std::vector<float>(2, 0.0f));
  wt.vh1.assign(4 * 2, 0.1f);
  wt.bh1.assign(4, 0.0f);
  wt.wv.assign(4, 0.25f);
  wt.bv = 0.0f;
  wt.wwdl.assign(3 * 4, 0.1f);
  wt.bwdl.assign(3, 0.0f);
  wt.wpol.assign(static_cast<size_t>(82 * 2 * 81), 0.001f);
  wt.bpol.assign(82, 0.0f);
  GoState st;
  CHECK(st.setState(emptyV02(9)));
  std::vector<float> planes;
  extractPlanesV02(st, planes);
  CHECK(planes.size() == 8 * 81);
  GoResnetOutput r1 = forwardGoResnet(wt, sz, planes.data());
  GoResnetOutput r2 = forwardGoResnet(wt, sz, planes.data());
  CHECK_CLOSE(r1.value, r2.value, 1e-9);
  GoResnetScratch sc;
  GoResnetOutput r3 = forwardGoResnetFast(wt, sz, planes.data(), sc);
  CHECK_CLOSE(r1.value, r3.value, 1e-5);
  for (size_t i = 0; i < r1.policy.size(); ++i) CHECK_CLOSE(r1.policy[i], r3.policy[i], 1e-5);
  CHECK(r1.policy.size() == 82);
  float s = 0.0f;
  for (float v : r1.policy) s += v;
  CHECK_CLOSE(s, 1.0, 1e-5);
}

bool readGoldenValue(const std::string& path, std::string& state, double& value) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t p = text.find("\"state\"");
  if (p == std::string::npos) return false;
  size_t a = text.find('"', p + 7);
  if (a == std::string::npos) return false;
  size_t b = text.find('"', a + 1);
  if (b == std::string::npos) return false;
  state = text.substr(a + 1, b - a - 1);
  size_t q = text.find("\"value\"", b);
  if (q == std::string::npos) return false;
  size_t c = text.find(':', q);
  if (c == std::string::npos) return false;
  size_t d = text.find_first_of(",}", c + 1);
  if (d == std::string::npos) return false;
  try {
    value = std::stod(text.substr(c + 1, d - c - 1));
    return true;
  } catch (...) {
    return false;
  }
}

void testFixtureV02() {
  GoResnetFile file;
  std::string err;
  CHECK(loadGoResnetFile("spec/test-vectors/models/resnet9-8plane-fp32.rune", file, err));
  CHECK(file.sizes.board == 9);
  CHECK(file.sizes.channels == 8);
  CHECK(file.sizes.blocks == 2);
  CHECK(file.sizes.inPlanes == 8);
  CHECK(file.featureVersion == "go_planes_v02");
  CHECK(file.weights.stemW.size() == 8 * 8 * 9);
  std::string state;
  double want = 0.0;
  CHECK(readGoldenValue("spec/test-vectors/resnet/eval_v02.json", state, want));
  GoState st;
  CHECK(st.setState(state));
  std::vector<float> planes;
  extractPlanesV02(st, planes);
  CHECK(planes.size() == 8 * 81);
  GoResnetOutput r = forwardGoResnet(file.weights, file.sizes, planes.data());
  CHECK_CLOSE(r.value, want, 1e-4);
  GoResnetScratch sc;
  GoResnetOutput rf = forwardGoResnetFast(file.weights, file.sizes, planes.data(), sc);
  CHECK_CLOSE(rf.value, want, 1e-4);
}

}

void testGoV02() {
  testParseV02();
  testPlanesV02();
  testContextV02();
  testRulesV02();
  testTokenV02();
  testResnetV02();
  testFixtureV02();
}
