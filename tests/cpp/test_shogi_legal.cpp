#include <string>
#include <vector>

#include "core/shogi/shogi_legal.h"
#include "tests/cpp/test_framework.h"

using namespace rune;
using namespace rune::shogi;

namespace {

bool hasMove(const std::vector<std::string>& moves, const std::string& mv) {
  for (const auto& m : moves) {
    if (m == mv) return true;
  }
  return false;
}

std::string startSfen() {
  return "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
}

void testStartposCounts() {
  std::vector<std::string> black;
  CHECK(shogiLegalMoves(startSfen(), black));
  CHECK(black.size() == 30);
  for (size_t i = 1; i < black.size(); ++i) CHECK(black[i - 1] < black[i]);
  std::string nxt;
  CHECK(shogiApplyMove(startSfen(), black[0], nxt));
  std::vector<std::string> white;
  CHECK(shogiLegalMoves(nxt, white));
  CHECK(white.size() == 30);
}

void testApplyRejects() {
  std::string out;
  CHECK(!shogiApplyMove(startSfen(), "D99p", out));
  CHECK(!shogiApplyMove(startSfen(), "8080", out));
}

void testPromo() {
  std::vector<std::string> moves;
  CHECK(shogiLegalMoves("9/1P7/9/9/9/9/9/9/4K4 b - 1", moves));
  CHECK(hasMove(moves, "1001+"));
  CHECK(!hasMove(moves, "1001"));
  std::vector<std::string> opt;
  CHECK(shogiLegalMoves("9/9/9/4P4/9/9/9/9/4K4 b - 1", opt));
  CHECK(hasMove(opt, "3122"));
  CHECK(hasMove(opt, "3122+"));
}

void testDropRules() {
  std::vector<std::string> moves;
  CHECK(shogiLegalMoves("9/9/9/9/4P4/9/9/9/4K4 b P 1", moves));
  for (const auto& m : moves) {
    if (!m.empty() && m[0] == 'D' && m.back() == 'p') {
      int to = std::stoi(m.substr(1, m.size() - 2));
      CHECK(to % 9 != 4);
      CHECK(to / 9 != 0);
    }
  }
  std::vector<std::string> mate;
  CHECK(shogiLegalMoves("4k4/9/9/9/9/9/9/9/4K4 b P 1", mate));
  CHECK(!hasMove(mate, "D76p"));
}

void testMateIsLoss() {
  std::vector<std::string> moves;
  CHECK(shogiLegalMoves("4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1", moves));
  CHECK(moves.empty());
  CHECK(shogiGameResult("4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1") == "1-0");
  CHECK(shogiGameResult(startSfen()) == "*");
}

}

void testShogiLegal() {
  testStartposCounts();
  testApplyRejects();
  testPromo();
  testDropRules();
  testMateIsLoss();
}
